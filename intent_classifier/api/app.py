"""
Flask API for the intent classification system.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

from intent_classifier.main import IntentClassificationPipeline
from intent_classifier.models.schemas import ClassificationConfig
from intent_classifier.config import API_CONFIG


# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=API_CONFIG["cors_origins"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the classification pipeline
pipeline = IntentClassificationPipeline()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "intent-classifier"
    })


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify a single query.
    
    Expected JSON payload:
    {
        "query": "What is the status of RFI #1838?",
        "user_id": "optional-user-id",
        "query_id": "optional-query-id"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query']
        user_id = data.get('user_id')
        query_id = data.get('query_id')
        
        # Classify the query
        result = pipeline.classify(query, user_id=user_id, query_id=query_id)
        
        # Convert to dict for JSON serialization
        return jsonify(result.dict())
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple queries in batch.
    
    Expected JSON payload:
    {
        "queries": [
            {"query": "...", "query_id": "..."},
            {"query": "...", "query_id": "..."}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({
                "error": "Missing 'queries' in request body"
            }), 400
        
        queries = data['queries']
        if not isinstance(queries, list):
            return jsonify({
                "error": "'queries' must be a list"
            }), 400
        
        results = []
        for item in queries:
            if isinstance(item, str):
                query = item
                query_id = None
            elif isinstance(item, dict) and 'query' in item:
                query = item['query']
                query_id = item.get('query_id')
            else:
                continue
            
            result = pipeline.classify(query, query_id=query_id)
            results.append(result.dict())
        
        return jsonify({
            "results": results,
            "total": len(results)
        })
    
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/taxonomy', methods=['GET'])
def get_taxonomy():
    """Get the taxonomy structure."""
    try:
        taxonomy_data = pipeline.taxonomy.get_taxonomy_summary()
        return jsonify(taxonomy_data)
    except Exception as e:
        logger.error(f"Error fetching taxonomy: {e}")
        return jsonify({
            "error": "Failed to fetch taxonomy"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error"
    }), 500


def create_app():
    """Factory function to create the Flask app."""
    return app


if __name__ == '__main__':
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )
