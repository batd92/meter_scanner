"""
YOLO Detector - Main detection interface
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .yolo_config import YOLOConfig
from .yolo_model import YOLOModelManager
from .yolo_postprocessor import YOLOPostProcessor
from .yolo_utils import draw_bbox_on_image, crop_image_with_bbox

class YOLODetector:
    """
    Main YOLO detection interface that orchestrates all components
    """
    
    def __init__(self, config: YOLOConfig = None):
        """
        Initialize YOLO detector
        
        Args:
            config: YOLO configuration (uses default if None)
        """
        self.config = config or YOLOConfig()
        self.model_manager = YOLOModelManager(self.config)
        self.postprocessor = YOLOPostProcessor(self.config)
        
        self.model_manager.load_model()
    
    def detect(self, img: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Detect objects in image
        
        Args:
            img: Input image (BGR format)
            **kwargs: Additional detection parameters
            
        Returns:
            List of detection dictionaries
        """
        if not self.model_manager.is_model_ready():
            print("❌ YOLO model not ready")
            return []
        
        try:
            results, info = self.model_manager.infer_image(img, **kwargs)
            if results is None or info is None:
                return []
            
            img_shape = img.shape[:2]
            detections = self.postprocessor.process_detections(results, img_shape, info)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return []
    
    def detect_with_visualization(self, img: np.ndarray, 
                                save_path: str = None,
                                show: bool = False,
                                **kwargs) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect objects and create visualization
        
        Args:
            img: Input image (BGR format)
            save_path: Path to save annotated image
            show: Whether to display image
            **kwargs: Additional detection parameters
            
        Returns:
            Tuple of (detections, annotated_image)
        """
        detections = self.detect(img, **kwargs)
        
        annotated_img = img.copy()
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['score']
            class_id = detection['class_id']
            
            label = f'{class_id}:{conf:.2f}'
            annotated_img = draw_bbox_on_image(annotated_img, bbox, label)
        
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            print(f"💾 Saved annotated image to: {save_path}")
        
        if show:
            cv2.imshow('YOLO Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections, annotated_img
    
    def crop_detections(self, img: np.ndarray, detections: List[Dict[str, Any]], 
                       padding_ratio: float = 0.2) -> List[Tuple[Dict[str, Any], np.ndarray]]:
        """
        Crop detected regions from image
        
        Args:
            img: Input image
            detections: List of detection dictionaries
            padding_ratio: Padding ratio for cropping
            
        Returns:
            List of (detection, cropped_image) tuples
        """
        cropped_regions = []
        
        for detection in detections:
            bbox = detection['bbox']
            cropped_img = crop_image_with_bbox(img, bbox, padding_ratio)
            cropped_regions.append((detection, cropped_img))
        
        return cropped_regions
    
    def filter_detections(self, detections: List[Dict[str, Any]], 
                         min_confidence: float = None,
                         allowed_classes: List[int] = None,
                         remove_overlapping: bool = True) -> List[Dict[str, Any]]:
        """
        Filter detections by various criteria
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold
            allowed_classes: List of allowed class IDs
            remove_overlapping: Whether to remove overlapping detections
            
        Returns:
            Filtered detections
        """
        filtered = detections.copy()
        
        # Filter by confidence
        if min_confidence is not None:
            filtered = self.postprocessor.filter_by_confidence(filtered, min_confidence)
        
        # Filter by class
        if allowed_classes is not None:
            filtered = self.postprocessor.filter_by_class(filtered, allowed_classes)
        
        # Remove overlapping
        if remove_overlapping:
            filtered = self.postprocessor.remove_overlapping_detections(filtered)
        
        return filtered
    
    def get_detection_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary dictionary
        """
        return self.postprocessor.get_detection_summary(detections)
    
    def update_config(self, **kwargs) -> bool:
        """
        Update configuration parameters
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            True if update successful
        """
        success = self.model_manager.update_config(**kwargs)
        if success:
            # Update postprocessor config reference
            self.postprocessor.config = self.model_manager.config
        return success
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_manager.get_model_info()
    
    def reload_model(self) -> bool:
        """Reload the model"""
        return self.model_manager.reload_model() 