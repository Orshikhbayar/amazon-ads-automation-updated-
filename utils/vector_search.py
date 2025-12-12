"""
Lightweight vector search using numpy (FAISS replacement for serverless)
Works with pre-computed embeddings stored in docs.jsonl
"""
import numpy as np
import json
import os

class VectorSearch:
    def __init__(self, docs_path: str, index_path: str = None):
        """
        Initialize vector search from docs.jsonl
        Each doc should have an 'embedding' field with the vector
        """
        self.docs = []
        self.embeddings = None
        
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.docs = [json.loads(line) for line in f]
            
            # Extract embeddings if they exist in docs
            if self.docs and 'embedding' in self.docs[0]:
                self.embeddings = np.array([doc['embedding'] for doc in self.docs], dtype=np.float32)
                # Normalize for cosine similarity
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                self.embeddings = self.embeddings / norms
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> tuple:
        """
        Search for most similar documents using cosine similarity
        
        Args:
            query_embedding: Query vector (1D or 2D array)
            top_k: Number of results to return
            
        Returns:
            (distances, indices) - similar to FAISS interface
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.array([[]]), np.array([[]])
        
        # Ensure query is 2D
        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize query
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        if query_norm[0, 0] > 0:
            query = query / query_norm
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.embeddings, query.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return np.array([top_scores]), np.array([top_indices])
    
    def get_doc(self, idx: int) -> dict:
        """Get document by index"""
        if 0 <= idx < len(self.docs):
            return self.docs[idx]
        return {}


def load_index_and_docs(index_path: str, docs_path: str):
    """
    Compatibility function - returns VectorSearch instance
    that mimics FAISS interface
    """
    return VectorSearch(docs_path, index_path)
