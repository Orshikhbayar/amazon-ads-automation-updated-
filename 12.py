#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12.py — OpenAI-only (requires OPENAI_API_KEY)

- Embeddings: utils.embedding.get_embedding (set EMBEDDING_BACKEND=openai)
- Retrieval: FAISS cosine (index from ingest_index_json.py)
- Output 1: Scored matches (from YOUR data) with cosine, match %, est CTR
- Output 2: ONE formatted "Proposed Target Segments" section (grounded on your data)
  * Uses ONLY retrieved segment names (no inventing)
- Output 3: Automatically saves generated segments to generated_segments.jsonl

Usage:
  export OPENAI_API_KEY=sk-...
  export EMBEDDING_BACKEND=openai
  export EMBEDDING_MODEL="text-embedding-3-small"
  python3 12.py --brief "Target SMB owners buying routers and labelers"
  python3 12.py --debug  # Show debug output

Flags:
  --no-extract       Disable AI keyword extraction (use brief only)
  --kw-weight 0.4    Blend weight for keyword embedding (0..1)
  --retrieval-only   Only print matches (no LLM output)
  --debug            Show debug output
"""

import os, sys, json, argparse, re
import numpy as np
import httpx
import traceback
import time
from datetime import datetime, timezone
from typing import List, Tuple

from packaging import version
from openai import OpenAI, BadRequestError
from utils.embedding import get_embedding  # uses EMBEDDING_BACKEND

# Try to import faiss, fall back to lightweight vector search
try:
    import faiss
    USE_FAISS = True
except ImportError:
    from utils.vector_search import VectorSearch
    USE_FAISS = False

# ---------------------------
# Config
# ---------------------------
GEN_MODEL = os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini")
DEBUG = False  # set from args

def _path(*parts):
    p1 = os.path.join("data", *parts)
    p2 = os.path.join("Data", *parts)
    return p1 if os.path.exists(p1) else p2

INDEX_PATH = _path("faiss.index")
DOCS_PATH  = _path("docs.jsonl")
JAPAN_MAP_PATH = _path("japan.json")

# ---------------------------
# Sanity: index & docs
# ---------------------------
missing = [p for p in (INDEX_PATH, DOCS_PATH) if not os.path.exists(p)]
if missing:
    sys.exit("❌ Missing files:\n  " + "\n  ".join(missing) +
             "\nTip: re-run `ingest_index_json.py` with your current EMBEDDING_MODEL.")

# Load index and docs based on available backend
if USE_FAISS:
    index = faiss.read_index(INDEX_PATH)
    docs = [json.loads(l) for l in open(DOCS_PATH, "r", encoding="utf-8")]
else:
    # Use lightweight vector search (loads embeddings from docs.jsonl)
    vector_search = VectorSearch(DOCS_PATH, INDEX_PATH)
    docs = vector_search.docs
    index = vector_search  # Use same interface

# Load Japanese name mapping (optional)
japanese_names = {}
if os.path.exists(JAPAN_MAP_PATH):
    with open(JAPAN_MAP_PATH, "r", encoding="utf-8") as f:
        japanese_names = json.load(f)
    if DEBUG:
        print(f"✅ Loaded {len(japanese_names)} Japanese name mappings")
else:
    print(f"⚠️  WARNING: Japanese mapping file not found at {JAPAN_MAP_PATH}")
    print(f"    Segment names will remain in English.")

# Translation cache to reduce API calls
_translation_cache = {}

# ---------------------------
# HTTPX client helper (handles 0.27 vs 0.28+)
# ---------------------------
def _make_httpx_client(proxy: str | None, timeout: float = 30.0) -> httpx.Client:
    """
    Create an httpx.Client that works across httpx versions.
    httpx 0.28+ uses 'proxy='; 0.27- uses 'proxies='.
    """
    kw = "proxy" if version.parse(httpx.__version__) >= version.parse("0.28.0") else "proxies"
    kwargs = {"timeout": timeout}
    if proxy:
        kwargs[kw] = proxy
    try:
        return httpx.Client(**kwargs)
    except TypeError:
        # Fallback if the current httpx doesn't support the chosen kw
        kwargs.pop(kw, None)
        alt_kw = "proxies" if kw == "proxy" else "proxy"
        if proxy:
            kwargs[alt_kw] = proxy
        return httpx.Client(**kwargs)

# ---------------------------
# Helpers
# ---------------------------
def debug_print(msg: str):
    if DEBUG:
        print(msg)

def has_japanese(text: str) -> bool:
    if not text:
        return False
    return any('\u3040' <= c <= '\u309F' or 
               '\u30A0' <= c <= '\u30FF' or 
               '\u4E00' <= c <= '\u9FAF' for c in text)

def get_japanese_name(english_name: str) -> str:
    japanese_name = japanese_names.get(english_name, english_name)
    debug_print(f"Mapping '{english_name}' → '{japanese_name}'")
    return japanese_name

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _percent_from_cos(cos_val: float) -> float:
    # cosine (-1..1) → 0..100%
    return max(0.0, min(1.0, (cos_val + 1.0) / 2.0)) * 100.0

def _tokenize_lower(s: str) -> set:
    if not s:
        return set()
    english_terms = set(re.findall(r"[a-zA-Z0-9\-\+/#\.]+", s.lower()))
    japanese_terms = set(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", s))
    return english_terms | japanese_terms

def estimate_ctr_percent(row: dict, base_ctr_pct: float = 1.0) -> float:
    """
    Heuristic CTR estimator:
      est_ctr% = base_ctr_pct * score_factor * kw_bonus
      score_factor = 0.5 + 0.5 * (match_pct/100)
      kw_bonus = 1 + 0.02 * hit_count, capped at 1.25
    """
    score_factor = 0.5 + 0.5 * (row["match_pct"] / 100.0)
    hit_count = len(row.get("hits_brief", []))
    kw_bonus = min(1.25, 1.0 + 0.02 * hit_count)
    return round(base_ctr_pct * score_factor * kw_bonus, 2)

# ---------------------------
# OpenAI Client (v1.x compatible, proxy via env if needed)
# ---------------------------
def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")
    http_client = _make_httpx_client(proxy, timeout=90.0)  # Increased timeout for better reliability

    # Ensure the SDK doesn't pick proxy env itself (we already provide client)
    # Not strictly necessary, but avoids double-proxy surprises.
    for k in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"):
        os.environ.pop(k, None)

    return OpenAI(api_key=api_key, http_client=http_client)  # no 'proxies' kw in v1.x

# ---------------------------
# Translation Functions
# ---------------------------
def translate_japanese_to_english(text: str) -> str:
    """Translate Japanese text to English for better embedding matching (with caching)."""
    cache_key = text[:100]
    if cache_key in _translation_cache:
        debug_print("Using cached translation")
        return _translation_cache[cache_key]
    if not has_japanese(text):
        debug_print("Text is already English, skipping translation")
        return text
    
    # Skip translation for very short text to save time
    if len(text.strip()) < 15:
        debug_print("Text too short, skipping translation")
        return text

    debug_print(f"Translating Japanese to English: '{text[:50]}...'")
    
    try:
        client = _openai_client()
        sys_msg = (
            "Translate the following Japanese text to English. "
            "Keep the meaning accurate and preserve marketing/business terminology. "
            "Return ONLY the English translation."
        )
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": text}]

        # Use shorter timeout for translation to prevent blocking
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=0.0,  # Completely deterministic translation
            max_completion_tokens=200,  # Reduced token limit
            timeout=10  # 10 second timeout
        )
        english_text = resp.choices[0].message.content.strip()
        debug_print(f"Translation result: '{english_text[:50]}...'")
        _translation_cache[cache_key] = english_text
        return english_text
    except Exception as e:
        debug_print(f"⚠️  Translation failed: {e}, using original text")
        _translation_cache[cache_key] = text  # Cache the original to avoid retrying
        return text

def translate_keywords_to_japanese(keywords: List[str]) -> List[str]:
    """Translate a list of English keywords to Japanese using OpenAI."""
    if not keywords:
        return keywords
    client = _openai_client()
    keywords_text = ", ".join(keywords)

    sys_msg = (
        "Translate the following English keywords to Japanese. "
        "Return ONLY a JSON array of Japanese translations in the same order. "
        "Keep marketing and product terms natural for Japanese Amazon users."
    )
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": keywords_text}]
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=0.5,
            max_completion_tokens=400
        )
        content = resp.choices[0].message.content.strip()
        translated = json.loads(content)
        if isinstance(translated, list) and len(translated) == len(keywords):
            return translated
    except Exception as e:
        debug_print(f"Keyword translation error: {e}")
    return keywords  # fallback

# ---------------------------
# Step 1: AI keyword extraction
# ---------------------------
def extract_keywords_ai(brief: str, max_terms: int = 10) -> List[str]:
    """
    Extract compact keywords (English and Japanese) for better retrieval.
    Returns a list of strings. Falls back to naive split if parsing fails.
    """
    client = _openai_client()
    sys_msg = (
        "Extract up to 10 concise keywords (English and/or Japanese) from the campaign brief that are "
        "useful for matching Amazon audience/product segments. "
        "Return ONLY a JSON array of strings. Include both English and Japanese terms when relevant."
    )
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": brief}]
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=200
        )
    except BadRequestError:
        resp = client.chat.completions.create(
            model=GEN_MODEL, 
            messages=messages,
            temperature=0.0
        )

    content = resp.choices[0].message.content.strip()
    try:
        arr = json.loads(content)
        out, seen = [], set()
        for x in arr:
            if isinstance(x, str):
                t = x.strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower()); out.append(t)
        return out[:max_terms]
    except Exception:
        toks = [t for t in _tokenize_lower(brief) if len(t) > 1]
        out, seen = [], set()
        for t in list(toks)[:max_terms]:
            if t not in seen:
                seen.add(t); out.append(t)
        return out

# ---------------------------
# Step 2: Retrieval
# ---------------------------
def retrieve_segments_detailed(
    brief: str,
    top_k: int = 10,
    use_extract: bool = True,
    kw_weight: float = 0.4
) -> Tuple[List[dict], List[str], str]:
    """
    Retrieve segments with error handling and validation.
    Returns: (rows, ai_kws, error_msg)
    """
    try:
        start_time = time.time()
        if not brief or len(brief.strip()) < 10:
            return [], [], "キャンペーンの詳細は10文字以上で入力してください。現在の文字数: " + str(len(brief.strip()) if brief else 0) + "文字"
        if top_k < 1 or top_k > 10:
            return [], [], "top_k must be between 1 and 10"

        debug_print(f"Processing brief: '{brief[:100]}...'")

        # Translate JP → EN for embedding with timeout protection
        try:
            english_brief = translate_japanese_to_english(brief)
            # Check if we're taking too long
            if time.time() - start_time > 6:
                debug_print("Translation taking too long, using original text")
                english_brief = brief
        except Exception as e:
            debug_print(f"Translation error: {e}, using original text")
            english_brief = brief
            
        debug_print(f"Using brief for embedding: '{english_brief[:100]}...'")

        emb_brief = _normalize(get_embedding(english_brief))
        debug_print(f"Brief embedding shape: {emb_brief.shape}")

        # Disable AI keyword extraction - use only brief terms
        ai_kws = []
        debug_print("Using brief terms only (no AI keyword extraction)")

        if ai_kws:
            kw_text = ", ".join(ai_kws)
            emb_kw = _normalize(get_embedding(kw_text))
            q_vec = _normalize((1.0 - kw_weight) * emb_brief + kw_weight * emb_kw)
        else:
            q_vec = emb_brief

        brief_terms = _tokenize_lower(english_brief)
        ai_terms = set([t.lower() for t in ai_kws])
        
        # Show extracted brief terms for user visibility
        if brief_terms and not DEBUG:
            brief_terms_list = sorted(list(brief_terms))
            if len(brief_terms_list) <= 20:
                # Show all terms if 20 or fewer
                print(f"🔑 Brief terms: {', '.join(brief_terms_list)}")
            else:
                # Show first 20 terms if more than 20
                print(f"🔑 Brief terms: {', '.join(brief_terms_list[:20])}")
                print(f"    (and {len(brief_terms) - 20} more...)")

        # Use a more comprehensive search approach
        search_size = min(max(top_k * 5, 50), len(docs))  # Start with larger search
        debug_print(f"Searching with size: {search_size}")
        D, I = index.search(np.array([q_vec], dtype="float32"), search_size)
        debug_print(f"First 5 cosine distances: {D[0][:5]}")

        rows, seen = [], set()

        for rank, idx in enumerate(I[0]):
            if len(rows) >= top_k:
                break

            rec = docs[idx]
            key = rec.get("keyword") or f"seg_{idx}"
            if key in seen:
                continue
            seen.add(key)

            text = (rec.get("text") or rec.get("answer") or "")
            text_terms = _tokenize_lower(key + " " + text)
            hits_from_ai = sorted(list(ai_terms.intersection(text_terms)))
            hits_from_brief = sorted(list(brief_terms.intersection(text_terms)))

            # For cosine/IP FAISS, D is similarity. Clamp to [0,1] just in case.
            cos = float(D[0][rank])
            cos = max(0.0, min(1.0, cos))

            debug_print(f"Segment '{key}' cosine: {cos:.3f}")

            if cos >= 0.4:
                pct = _percent_from_cos(cos)
                rows.append({
                    "keyword": key,
                    "text": text,
                    "cosine": cos,
                    "match_pct": pct,
                    "hits_ai": hits_from_ai,
                    "hits_brief": hits_from_brief,
                })
                debug_print(f"✅ Accepted segment '{key}' ({pct:.1f}%)")
            else:
                debug_print(f"❌ Rejected segment '{key}' (score too low)")

        # If still not enough, lower threshold to 0.3
        if len(rows) < top_k:
            debug_print(f"Only found {len(rows)}/{top_k} segments, using lower threshold")
            for rank, idx in enumerate(I[0]):
                if len(rows) >= top_k:
                    break
                rec = docs[idx]
                key = rec.get("keyword") or f"seg_{idx}"
                if key in seen:
                    continue
                seen.add(key)
                
                cos = float(D[0][rank])
                cos = max(0.0, min(1.0, cos))
                
                if cos >= 0.2:  # Lower threshold for fallback
                    text = (rec.get("text") or rec.get("answer") or "")
                    text_terms = _tokenize_lower(key + " " + text)
                    hits_from_ai = sorted(list(ai_terms.intersection(text_terms)))
                    hits_from_brief = sorted(list(brief_terms.intersection(text_terms)))
                    pct = _percent_from_cos(cos)
                    rows.append({
                        "keyword": key,
                        "text": text,
                        "cosine": cos,
                        "match_pct": pct,
                        "hits_ai": hits_from_ai,
                        "hits_brief": hits_from_brief,
                    })
                    debug_print(f"✅ Fallback accepted segment '{key}' ({pct:.1f}%)")

        if len(rows) == 0:
            debug_print("No segments found, checking search parameters...")
            debug_print(f"Search size: {search_size}, docs available: {len(docs)}")
            debug_print(f"Query vector shape: {q_vec.shape if hasattr(q_vec, 'shape') else 'no shape'}")
            return [], [], "No matching segments found. Try different keywords or lower the match threshold."

        debug_print(f"Final result: {len(rows)} segments retrieved")
        return rows, ai_kws, None

    except Exception as e:
        traceback.print_exc()  # full stacktrace
        return [], [], f"Retrieval error: {str(e)}"

# ---------------------------
# Step 3: Prompt / Generation
# ---------------------------
STRICT_RULES = """
HARD RULES:
- Use ONLY segment names from Allowed Segment Names (verbatim).
- Do NOT invent or rephrase segment names.
- For **Keywords**, use ONLY terms and facts found in each segment's Text.
- Keep outputs concise and ad-ready.
"""

INSTRUCTIONS = """
You are an Amazon Ads strategist.
Always respond entirely in Japanese.

Propose relevant target segments with:
- a brief 'Why it fits' (1–2 lines) in Japanese,
- 10 Keywords in Japanese taken FROM the segment Text.

IMPORTANT: Base all content strictly on the provided segment Text. Do not invent information.
"""

def build_prompt_strict(campaign_brief: str, retrieved_rows: List[dict], allowed_names: List[str]) -> str:
    blocks = []
    for i, r in enumerate(retrieved_rows, 1):
        preview = r["text"][:320]
        japanese_name = get_japanese_name(r['keyword'])
        blocks.append(f"Segment {i}: {japanese_name}\nText: {preview}")
    context = "\n\n".join(blocks)
    allowed_block = ", ".join(allowed_names)

    # Generate template based on actual number of segments
    segment_templates = []
    for i, name in enumerate(allowed_names, 1):
        template = f"""**Segment {i}: {name}**  
**Why it fits:** 1–2 lines in Japanese explaining relevance to the campaign.  
**Keywords:** 10 Japanese terms FROM its Text."""
        segment_templates.append(template)

    return f"""{INSTRUCTIONS}
{STRICT_RULES}

=== Campaign Brief ===
{campaign_brief}

=== Retrieved Segments (from your data) ===
{context}

=== Allowed Segment Names (use ALL {len(allowed_names)} segments below) ===
{allowed_block}

CRITICAL REQUIREMENTS:
1. You MUST create exactly {len(allowed_names)} complete segments using ALL the names above
2. Each segment MUST have ALL fields: Why it fits, Keywords
3. DO NOT stop or truncate - complete ALL segments fully

Return ONLY the section below in clean markdown:

💡 Proposed Target Segments

{chr(10).join(segment_templates)}
""".strip()

def generate_with_openai(prompt: str, model: str = GEN_MODEL) -> str:
    client = _openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=1000,  # Appropriate for 10 segments
            timeout=60  # 60 second timeout
        )
        result = resp.choices[0].message.content.strip()
        
        # Check if output seems complete (should end with description, not cut off)
        if not result.endswith('.') and not result.endswith('。') and not result.endswith('す'):
            debug_print("⚠️ Output may be truncated, trying with higher token limit")
            # Retry with higher limit
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=1500,  # Higher limit for retry
                timeout=90  # Longer timeout for retry
            )
            result = resp.choices[0].message.content.strip()
        
        return result
    except BadRequestError:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=60
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Generation failed: {str(e)}\nPlease try again or contact support."

# ---------------------------
# Pretty printing
# ---------------------------
def print_matches(rows: List[dict], ai_kws: List[str], base_ctr_pct: float = 1.0):
    print("\n🔎 Matched segments (from YOUR data):\n")
    for i, r in enumerate(rows, 1):
        est_ctr = estimate_ctr_percent(r, base_ctr_pct=base_ctr_pct)
        segment_name = get_japanese_name(r['keyword'])
        print(f"{i}) {segment_name}")
        print(f"   • score: {r['cosine']:.3f}  |  match: {r['match_pct']:.1f}%  |  est CTR: {est_ctr:.2f}%")
        if r["hits_brief"]:
            print(f"   • matching terms: {', '.join(r['hits_brief'])}")

# ---------------------------
# Save generation output
# ---------------------------
def save_generation(brief: str, ai_kws: List[str], rows: List[dict], md_output: str, path="generated_segments.jsonl"):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "brief": brief,
            "ai_keywords": ai_kws,
            "retrieved_segments": [r["keyword"] for r in rows],
            "scores": [
                {"keyword": r["keyword"], "match_pct": r["match_pct"], "cosine": r["cosine"]}
                for r in rows
            ],
            "output_markdown": md_output
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        debug_print(f"💾 Saved generation → {path}")
    except Exception as e:
        debug_print(f"⚠️  Could not save to {path}: {e}")

# ---------------------------
# CLI
# ---------------------------
def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="Amazon Ads Automation - Segment Generator")

    ap.add_argument("--brief", type=str, default=None, help="Inline campaign brief.")
    ap.add_argument("--no-extract", action="store_true", help="Disable AI keyword extraction.")
    ap.add_argument("--kw-weight", type=float, default=0.5, help="Blend weight for keyword embedding (0-1).")
    ap.add_argument("--retrieval-only", action="store_true", help="Only print matches, skip generation.")
    ap.add_argument("--base-ctr", type=float, default=1.0, help="Base CTR %% prior (default 1.0).")
    ap.add_argument("--debug", action="store_true", help="Show debug output.")
    args = ap.parse_args()

    DEBUG = args.debug

    brief = args.brief or input("Enter campaign brief: ").strip()
    if not brief:
        print("❌ Campaign brief is required")
        return

    rows, ai_kws, error = retrieve_segments_detailed(
        brief,
        top_k=10,
        use_extract=not args.no_extract,
        kw_weight=max(0.0, min(1.0, args.kw_weight)),
    )

    if error:
        print(f"\n❌ {error}")
        return

    print_matches(rows, ai_kws, base_ctr_pct=args.base_ctr)

    if args.retrieval_only:
        return

    try:
        allowed = [get_japanese_name(r["keyword"]) for r in rows]
        prompt = build_prompt_strict(brief, rows, allowed)
        md = generate_with_openai(prompt, model=GEN_MODEL)

        print("\n" + md + "\n")
        save_generation(brief, ai_kws, rows, md_output=md)

    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        if DEBUG:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
