"""
Striga Detection Inference Module
Handles YOLO model loading and inference operations
"""

import torch
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detection result"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class InferenceResults:
    """Complete inference results"""
    detections: List[DetectionResult]
    overall_health: str
    infection_level: str
    avg_confidence: float
    recommendations: List[str]
    primary_health: str = "no_plant_detected"
    primary_infection_level: str = "invalid"


class StrigaDetector:
    """YOLO-based Striga Detection Model"""
    
    # Class labels
    CLASS_LABELS = {
        0: 'Anthracnose_Leaf_Spot',
        1: 'Sorghum Rust',
        2: 'Sorghum_Burned_Leaf',
        3: 'Sorghum_Healthy',
        4: 'Sorghum_Loose_Smut',
        5: 'Sorghum_Red_Rot',
        6: 'Striga_Flower'
    }
    
    # Recommendations by infection level
    RECOMMENDATIONS = {
        'none': [
            'Continue regular monitoring',
            'Maintain field hygiene',
            'Implement crop rotation'
        ],
        'other_disease': [
            'A non-striga disease detected (e.g. Rust, Anthracnose)',
            'Consider applying appropriate fungicide',
            'Monitor field for spread'
        ],
        'possible_striga': [
            'Plant shows symptoms associated with Striga (e.g. Burned Leaf, Red Rot)',
            'Begin early intervention and inspect soil for emerging weeds',
            'Consider field isolation'
        ],
        'striga_confirmed': [
            'Striga flower visibly detected!',
            'URGENT: Contact agricultural extension',
            'Hand-pull visible weeds before they drop seeds',
            'Treat soil before replanting'
        ]
    }
    
    def __init__(self, model_path: str):
        """
        Initialize detector with YOLO model
        
        Args:
            model_path: Path to .pt model file
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model from file"""
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            
            # Try loading with ultralytics (YOLOv8)
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info("Loaded as YOLOv8 model")
                return
            except ImportError:
                pass
            
            # Fallback to YOLOv5
            logger.info("Attempting YOLOv5 loading...")
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=self.model_path,
                force_reload=False
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on device: {self.device}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> InferenceResults:
        """
        Run inference on image
        
        Args:
            image: Input image as numpy array (RGB format)
            conf_threshold: Confidence threshold for detections
        
        Returns:
            InferenceResults with detections and analysis
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run inference
            if hasattr(self.model, 'predict'):  # YOLOv8 interface
                results = self.model.predict(
                    image,
                    conf=conf_threshold,
                    verbose=False
                )
                detections = self._parse_yolov8_results(results, image.shape)
            else:  # YOLOv5 interface
                results = self.model(image)
                detections = self._parse_yolov5_results(results, image.shape, conf_threshold)
            
            # Generate overall health assessment
            health_status, infection_level = self._assess_health(detections)
            
            # Generate primary plant assessment (closest to center)
            primary_health, primary_infection = self._get_primary_health(detections, image.shape)
            
            # Get recommendations based on primary infection level
            recommendations = self.RECOMMENDATIONS.get(primary_infection, self.RECOMMENDATIONS.get(infection_level, []))
            
            # Calculate average confidence
            avg_conf = np.mean([d.confidence for d in detections]) if detections else 0.0
            
            return InferenceResults(
                detections=detections,
                overall_health=health_status,
                infection_level=infection_level,
                avg_confidence=avg_conf,
                recommendations=recommendations,
                primary_health=primary_health,
                primary_infection_level=primary_infection
            )
        
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _parse_yolov8_results(self, results, image_shape) -> List[DetectionResult]:
        """Parse YOLOv8 results"""
        detections = []
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                class_name = self.CLASS_LABELS.get(cls, 'unknown')
                
                detection = DetectionResult(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2))
                )
                detections.append(detection)
        
        return detections
    
    def _parse_yolov5_results(self, results, image_shape, conf_threshold) -> List[DetectionResult]:
        """Parse YOLOv5 results"""
        detections = []
        
        if results is None or len(results.xyxy[0]) == 0:
            return detections
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            conf = float(conf)
            
            # Skip low confidence detections
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = self.CLASS_LABELS.get(class_id, 'unknown')
            
            detection = DetectionResult(
                class_name=class_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2)
            )
            detections.append(detection)
        
        return detections
    
    def _assess_health(self, detections: List[DetectionResult]) -> Tuple[str, str]:
        """
        Assess overall plant health from detections
        
        Returns:
            (overall_health, infection_level)
        """
        if not detections:
            return 'no_plant_detected', 'invalid'
        
        # Map class names to severity levels
        severity_map = {
            'Sorghum_Healthy': 0,
            'Anthracnose_Leaf_Spot': 1,
            'Sorghum Rust': 1,
            'Sorghum_Loose_Smut': 1,
            'Sorghum_Burned_Leaf': 2,
            'Sorghum_Red_Rot': 2,
            'Striga_Flower': 3
        }
        
        # Get maximum severity
        severities = [severity_map.get(d.class_name, 0) for d in detections]
        max_severity = max(severities)
        
        # Determine infection level
        severity_labels = {
            0: 'none',
            1: 'other_disease',
            2: 'possible_striga',
            3: 'striga_confirmed'
        }
        
        infection_level = severity_labels.get(max_severity, 'unknown')
        overall_health = 'clean' if max_severity == 0 else 'infected'
        
        return overall_health, infection_level
    
    def _get_primary_health(self, detections: List[DetectionResult], image_shape: Tuple[int, ...]) -> Tuple[str, str]:
        """
        Identify the primary plant (closest to image center) and assess its health.
        
        Returns:
            (primary_health, primary_infection_level)
        """
        if not detections:
            return 'no_plant_detected', 'invalid'
            
        height, width = image_shape[:2]
        center_x, center_y = width / 2, height / 2
        
        closest_detection = None
        min_dist = float('inf')
        
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            
            dist = (box_center_x - center_x)**2 + (box_center_y - center_y)**2
            if dist < min_dist:
                min_dist = dist
                closest_detection = d
                
        if closest_detection:
            return self._assess_health([closest_detection])
            
        return 'no_plant_detected', 'invalid'
    
    def draw_boxes(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        Draw bounding boxes on image for visualization
        
        Args:
            image: Input image as numpy array
            detections: List of detected objects
        
        Returns:
            Image with drawn boxes
        """
        image_copy = image.copy()
        
        # Color map for classes
        colors = {
            'Sorghum_Healthy': (0, 255, 0),  # Green
            'Anthracnose_Leaf_Spot': (255, 255, 0),  # Yellow
            'Sorghum Rust': (255, 255, 0),  # Yellow
            'Sorghum_Loose_Smut': (255, 255, 0),  # Yellow
            'Sorghum_Burned_Leaf': (255, 165, 0),  # Orange
            'Sorghum_Red_Rot': (255, 165, 0),  # Orange
            'Striga_Flower': (0, 0, 255)  # Red
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors.get(detection.class_name, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(
                image_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return image_copy
