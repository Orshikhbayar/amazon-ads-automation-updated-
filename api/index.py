from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Create Flask app
app = Flask(__name__)
CORS(app)

# Try to import routes from main app
try:
    from app import app as main_app
    
    # Copy routes from main app
    for rule in main_app.url_map.iter_rules():
        if rule.endpoint != 'static':
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
    
    # Fallback routes if import fails
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Amazon Ads Automation API',
            'status': 'running',
            'error': error_msg,
            'trace': error_trace
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'ok', 
            'message': 'Serverless function is running',
            'import_error': error_msg
        })

# Vercel requires the app to be exposed at module level
# The @vercel/python runtime will automatically use this
