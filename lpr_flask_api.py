from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import traceback
from datetime import datetime
import base64
from PIL import Image
import io
import uuid
from functools import wraps
import time
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lpr_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Enable CORS for all routes and origins
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001'])

# Or for more specific configuration:
# CORS(app, 
#      origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
#      methods=['GET', 'POST', 'OPTIONS'],
#      allow_headers=['Content-Type'],
#      supports_credentials=True)


# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', 'model_using.pt')
app.config['CONFIDENCE_THRESHOLD'] = float(os.getenv('CONFIDENCE_THRESHOLD', '0.15'))
app.config['IOU_THRESHOLD'] = float(os.getenv('IOU_THRESHOLD', '0.5'))

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class LicensePlateRecognizer:
    def __init__(self, model_path, conf=0.25, classes=None, iou=0.7, resize_size=(320, 320)):
        """
        Initialize the License Plate Recognizer.

        Args:
            model_path (str): Path to the YOLO model file.
            conf (float): Confidence threshold for predictions.
            classes (list): List of class IDs to detect (None for all classes).
            iou (float): IoU threshold for non-max suppression.
            resize_size (tuple): (width, height) for resizing input images.
        """
        self.model = self._load_model(model_path)
        self.conf = conf
        self.classes = classes
        self.iou = iou
        self.resize_size = resize_size
        self.device = 'cpu'
        
        # Define city and vehicle class IDs based on provided class names
        self.city_class_ids = [
            12, 13, 14, 15, 16, 18, 19, 21, 22, 24, 26, 27, 28, 32, 33, 35, 36,
            38, 39, 41, 43, 45, 46, 48, 49, 50, 54, 55, 56, 57, 59, 60, 62, 63,
            64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 84,
            85, 86, 87, 89, 91, 92, 93, 94, 95, 98, 100
        ]
        self.vehicle_class_ids = [
            10, 11, 17, 20, 25, 29, 30, 31, 34, 37, 40, 42, 44, 47, 51, 52, 58,
            61, 70, 79, 83, 88, 90, 96, 97, 99, 101
        ]

    def _load_model(self, model_path):
        """Load the YOLO model."""
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} does not exist.")
            return None
        try:
            model = YOLO(model_path)
            logger.info(f"Model {model_path} loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None

    def _load_image(self, image_input):
        """Load a single image from various input types."""
        try:
            if isinstance(image_input, np.ndarray):
                return image_input
            elif isinstance(image_input, str):
                if os.path.exists(image_input):
                    img = cv2.imread(image_input)
                    if img is None:
                        logger.error(f"Failed to load image {image_input}.")
                        return None
                    return img
            return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def _process_detections(self, img):
        """Process detections and construct license plate string."""
        try:
            results = self.model.predict(
                img, 
                conf=self.conf, 
                classes=self.classes, 
                iou=self.iou, 
                verbose=False, 
                device=self.device, 
                imgsz=(640, 480)
            )

            if not results:
                return "", []

            # Initialize detection components
            digits = []
            city = None
            has_metro = False
            vehicle_class = None
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    xmin, ymin = box.xyxy[0][:2].cpu().numpy().astype(int)
                    xmax, ymax = box.xyxy[0][2:4].cpu().numpy().astype(int)
                    class_id = int(box.cls.item()) if box.cls is not None else -1
                    confidence = float(box.conf.item()) if box.conf is not None else 0.0
                    label = self.model.names[class_id] if class_id in self.model.names else "Unknown"

                    # Store detection info
                    detection = {
                        'class_id': class_id,
                        'label': label,
                        'confidence': confidence,
                        'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]
                    }
                    detections.append(detection)

                    # Collect components
                    if class_id in range(0, 10):  # Digits 0-9
                        digits.append((label, xmin, confidence))
                    elif class_id in self.city_class_ids:  # City names
                        if city is None or confidence > city[1]:  # Take highest confidence city
                            city = (label, confidence)
                    elif class_id == 66:  # Metro
                        has_metro = True
                    elif class_id in self.vehicle_class_ids:  # Vehicle classes
                        if vehicle_class is None or confidence > vehicle_class[1]:
                            vehicle_class = (label, confidence)

            # Construct license plate
            license_plate = ""
            if len(digits) == 6:
                sorted_digits = sorted(digits, key=lambda x: x[1])  # Sort by x position
                number = ''.join(digit[0] for digit in sorted_digits)
                
                city_name = city[0] if city else None
                vehicle_name = vehicle_class[0] if vehicle_class else None
                
                if city_name and has_metro and vehicle_name:
                    license_plate = f"{city_name} Metro {vehicle_name} {number}"
                elif city_name and has_metro:
                    license_plate = f"{city_name} Metro {number}"
                else:
                    license_plate = number

            return license_plate, detections

        except Exception as e:
            logger.error(f"Error processing detections: {e}")
            return "", []

    def process_image(self, image_input):
        """
        Process a single image and return the license plate string and detections.

        Args:
            image_input: Image file path or numpy array

        Returns:
            tuple: (license_plate_string, detections_list)
        """
        if self.model is None:
            logger.error("No model loaded.")
            return "", []

        img = self._load_image(image_input)
        if img is None:
            return "", []

        return self._process_detections(img)

# Initialize the recognizer globally
recognizer = None

def init_recognizer():
    """Initialize the recognizer with model."""
    global recognizer
    try:
        model_path = app.config['MODEL_PATH']
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        recognizer = LicensePlateRecognizer(
            model_path=model_path,
            conf=app.config['CONFIDENCE_THRESHOLD'],
            iou=app.config['IOU_THRESHOLD'],
            resize_size=(320, 320)
        )
        
        if recognizer.model is None:
            logger.error("Failed to load model")
            return False
            
        logger.info("License Plate Recognizer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing recognizer: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def require_model(f):
    """Decorator to ensure model is loaded before processing."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if recognizer is None or recognizer.model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server configuration.',
                'error_code': 'MODEL_NOT_LOADED'
            }), 503
        return f(*args, **kwargs)
    return decorated_function

def validate_image(image_data):
    """Validate image data."""
    try:
        # Try to open with PIL to validate
        img = Image.open(io.BytesIO(image_data))
        img.verify()
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_status = recognizer is not None and recognizer.model is not None
    return jsonify({
        'status': 'healthy' if model_status else 'unhealthy',
        'model_loaded': model_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
@require_model
def predict_license_plate():
    """
    Predict license plate from uploaded image.
    Accepts both file upload and base64 encoded images.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Request {request_id}: Processing license plate prediction")
        
        image_data = None
        filename = None
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid file type. Allowed: PNG, JPG, JPEG',
                        'error_code': 'INVALID_FILE_TYPE'
                    }), 400
                
                filename = secure_filename(file.filename)
                image_data = file.read()
        
        # Handle base64 encoded image
        elif 'image_base64' in request.json if request.is_json else {}:
            try:
                base64_string = request.json['image_base64']
                if 'data:' in base64_string:
                    base64_string = base64_string.split(',')[1]
                image_data = base64.b64decode(base64_string)
                filename = f"base64_image_{request_id}.jpg"
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Invalid base64 image data',
                    'error_code': 'INVALID_BASE64'
                }), 400
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image provided. Use "image" file field or "image_base64" JSON field.',
                'error_code': 'NO_IMAGE_PROVIDED'
            }), 400
        
        # Validate image
        if not validate_image(image_data):
            return jsonify({
                'success': False,
                'error': 'Invalid or corrupted image file',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image',
                'error_code': 'IMAGE_DECODE_ERROR'
            }), 400
        
        # Process image
        license_plate, detections = recognizer.process_image(img)
        print(license_plate)
        
        processing_time = time.time() - start_time
        
        response = {
            'success': True,
            'license_plate': license_plate,
            'detected': bool(license_plate),
            'detections': detections,
            'processing_time_seconds': round(processing_time, 3),
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Request {request_id}: Completed in {processing_time:.3f}s - Result: '{license_plate}'")
        return jsonify(response)
        
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 16MB.',
            'error_code': 'FILE_TOO_LARGE'
        }), 413
        
    except Exception as e:
        logger.error(f"Request {request_id}: Error processing image: {e}")
        logger.error(f"Request {request_id}: Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'Internal server error occurred during processing',
            'error_code': 'INTERNAL_ERROR',
            'request_id': request_id
        }), 500

@app.route('/batch_predict', methods=['POST'])
@require_model
def batch_predict():
    """Process multiple images in a single request."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Batch Request {request_id}: Processing multiple images")
        
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No images provided in "images" field',
                'error_code': 'NO_IMAGES_PROVIDED'
            }), 400
        
        files = request.files.getlist('images')
        if len(files) > 10:  # Limit batch size
            return jsonify({
                'success': False,
                'error': 'Maximum 10 images allowed per batch',
                'error_code': 'BATCH_SIZE_EXCEEDED'
            }), 400
        
        results = []
        for i, file in enumerate(files):
            try:
                if not file or file.filename == '':
                    results.append({
                        'index': i,
                        'filename': '',
                        'success': False,
                        'error': 'Empty file'
                    })
                    continue
                
                if not allowed_file(file.filename):
                    results.append({
                        'index': i,
                        'filename': file.filename,
                        'success': False,
                        'error': 'Invalid file type'
                    })
                    continue
                
                image_data = file.read()
                if not validate_image(image_data):
                    results.append({
                        'index': i,
                        'filename': file.filename,
                        'success': False,
                        'error': 'Invalid image'
                    })
                    continue
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    results.append({
                        'index': i,
                        'filename': file.filename,
                        'success': False,
                        'error': 'Failed to decode image'
                    })
                    continue
                
                # Process image
                license_plate, detections = recognizer.process_image(img)
                
                results.append({
                    'index': i,
                    'filename': file.filename,
                    'success': True,
                    'license_plate': license_plate,
                    'detected': bool(license_plate),
                    'detections': detections
                })
                
            except Exception as e:
                logger.error(f"Batch Request {request_id}: Error processing image {i}: {e}")
                results.append({
                    'index': i,
                    'filename': file.filename if file else '',
                    'success': False,
                    'error': 'Processing error'
                })
        
        processing_time = time.time() - start_time
        successful_results = sum(1 for r in results if r['success'])
        
        response = {
            'success': True,
            'results': results,
            'total_images': len(files),
            'successful_predictions': successful_results,
            'processing_time_seconds': round(processing_time, 3),
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch Request {request_id}: Completed {successful_results}/{len(files)} in {processing_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch Request {request_id}: Error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error occurred during batch processing',
            'error_code': 'BATCH_INTERNAL_ERROR',
            'request_id': request_id
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    if recognizer is None or recognizer.model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        model_info = {
            'success': True,
            'model_path': app.config['MODEL_PATH'],
            'confidence_threshold': app.config['CONFIDENCE_THRESHOLD'],
            'iou_threshold': app.config['IOU_THRESHOLD'],
            'device': recognizer.device,
            'class_names': recognizer.model.names if hasattr(recognizer.model, 'names') else {},
            'total_classes': len(recognizer.model.names) if hasattr(recognizer.model, 'names') else 0
        }
        return jsonify(model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve model information'
        }), 500


@app.route('/cors-test', methods=['GET', 'OPTIONS'])
def cors_test():
    """Simple endpoint to test CORS is working"""
    return jsonify({
        'success': True,
        'message': 'CORS is working!',
        'timestamp': datetime.now().isoformat()
    })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.',
        'error_code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'error_code': 'NOT_FOUND'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'error_code': 'INTERNAL_ERROR'
    }), 500

# Initialize the app
if __name__ == '__main__':
    # Initialize recognizer
    if not init_recognizer():
        logger.error("Failed to initialize recognizer. Exiting.")
        exit(1)
    
    # Run the app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting License Plate Recognition API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)