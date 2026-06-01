"""
Striga Detection Backend API
Flask-based REST API for YOLO model inference on sorghum plants
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import base64
import os
import sys
import logging
import torch
from datetime import datetime
from functools import wraps
from io import BytesIO
from PIL import Image

from inference import StrigaDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global detector instance
detector = None

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
INFERENCE_CONFIDENCE_THRESHOLD = 0.7


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_model():
    """Load YOLO model at startup"""
    global detector
    try:
        model_path = os.getenv('MODEL_PATH', './striga/yolo_training/my_model/weights/best.pt')
        logger.info(f"Loading model from: {model_path}")
        detector = StrigaDetector(model_path)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


def require_detector(f):
    """Decorator to ensure detector is initialized"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if detector is None:
            return jsonify({"error": "Model not loaded"}), 500
        return f(*args, **kwargs)
    return decorated_function


# Global variables
model = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Striga severity labels
SEVERITY_LABELS = {
    0: 'clean',
    1: 'light_infestation',
    2: 'moderate_infestation',
    3: 'severe_infestation'
}

# API Endpoints


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector is not None,
        "device": "cuda" if detector else "unknown"
    })


@app.route('/info', methods=['GET'])
def info():
    """Get API and model information"""
    return jsonify({
        "api_version": "1.0.0",
        "app_name": "Striga Detection API",
        "description": "YOLO-based striga detection for sorghum plants",
        "model_info": {
            "type": "YOLOv8",
            "trained_labels": ["clean", "light_infestation", "moderate_infestation", "severe_infestation"],
            "input_size": "640x640"
        },
        "endpoints": [
            "/health",
            "/info",
            "/predict",
            "/predict-with-image"
        ]
    })


@app.route('/predict', methods=['POST'])
@require_detector
def predict():
    """
    Main prediction endpoint
    
    Accepts:
    - multipart form with 'image' file, OR
    - JSON with 'image' as base64 string
    
    Returns:
    - Detection results with bounding boxes and confidence scores
    """
    try:
        image = None
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(file.filename):
                return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
            
            image = Image.open(file.stream).convert('RGB')
            logger.info(f"Processing uploaded file: {file.filename}")
        
        # Handle base64 image
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({"error": "No image provided"}), 400
            
            try:
                image_data = base64.b64decode(data['image'])
                image = Image.open(BytesIO(image_data)).convert('RGB')
                logger.info("Processing base64 image")
            except Exception as e:
                return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Run inference
        logger.info("Running inference...")
        results = detector.detect(np.array(image), conf_threshold=INFERENCE_CONFIDENCE_THRESHOLD)
        
        # Draw bounding boxes on image
        annotated_image = detector.draw_boxes(np.array(image), results.detections)
        annotated_pil = Image.fromarray(annotated_image)
        img_io = BytesIO()
        annotated_pil.save(img_io, 'JPEG')
        annotated_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Format response
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": image.width,
                "height": image.height
            },
            "detections": [
                {
                    "class": result.class_name,
                    "confidence": float(result.confidence),
                    "bbox": {
                        "x1": int(result.bbox[0]),
                        "y1": int(result.bbox[1]),
                        "x2": int(result.bbox[2]),
                        "y2": int(result.bbox[3])
                    }
                }
                for result in results.detections
            ],
            "overall_health": results.overall_health,
            "infection_level": results.infection_level,
            "primary_health": results.primary_health,
            "primary_infection_level": results.primary_infection_level,
            "confidence_average": float(results.avg_confidence),
            "recommendations": results.recommendations,
            "detected_areas": len(results.detections),
            "annotated_image": annotated_base64
        }
        
        logger.info(f"Inference complete. Detections: {len(results.detections)}, Health: {results.overall_health}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/predict-with-image', methods=['POST'])
@require_detector
def predict_with_image():
    """
    Prediction endpoint that returns annotated image
    Useful for debugging and visualization
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        image = Image.open(file.stream).convert('RGB')
        image_array = np.array(image)
        
        # Run inference
        results = detector.detect(image_array)
        
        # Draw bounding boxes on image
        annotated_image = detector.draw_boxes(image_array, results.detections)
        
        # Convert to PIL Image and save to bytes
        annotated_pil = Image.fromarray(annotated_image)
        img_io = BytesIO()
        annotated_pil.save(img_io, 'PNG')
        img_io.seek(0)
        
        logger.info(f"Returned annotated image with {len(results.detections)} detections")
        return send_file(img_io, mimetype='image/png'), 200
    
    except Exception as e:
        logger.error(f"Error in predict_with_image: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': 'YOLOv8 - Striga Detection',
        'classes': {
            0: 'Anthracnose_Leaf_Spot',
            1: 'Sorghum Rust',
            2: 'Sorghum_Burned_Leaf',
            3: 'Sorghum_Healthy',
            4: 'Sorghum_Loose_Smut',
            5: 'Sorghum_Red_Rot',
            6: 'Striga_Flower'
        },
        'version': '1.0'
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


def main():
    """Main entry point"""
    # Load model
    if not initialize_model():
        logger.warning("Model not loaded. API will run but /predict endpoint will fail")
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Striga Detection API on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
