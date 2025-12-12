from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Path to public directory
PUBLIC_DIR = os.path.join(parent_dir, 'public')

# Override the root route to serve index.html directly
@app.route('/')
def serve_index():
    index_path = os.path.join(PUBLIC_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html')
    return jsonify({'error': 'index.html not found', 'path': index_path}), 404

# Try to import API routes from main app
try:
    from app import app as main_app
    
    # Copy only API routes from main app (skip root route)
    for rule in main_app.url_map.iter_rules():
        if rule.endpoint != 'static' and rule.rule != '/':
            app.add_url_rule(
                rule.rule,
                rule.endpoint,
                main_app.view_functions[rule.endpoint],
                methods=rule.methods
            )
except Exception as e:
    import traceback
    error_msg = str(e)
    error_trace = traceback.format_exc()
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'error', 
            'message': 'Main app import failed',
            'error': error_msg,
            'trace': error_trace
        })
