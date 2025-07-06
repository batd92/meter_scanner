"""
YOLO Post-processing - NMS, filtering, and result processing
"""

import numpy as np
from typing import List, Dict, Any
from .yolo_utils import validate_bbox, clip_bbox_to_image, map_bbox_to_original

class YOLOPostProcessor:
    """Post-process YOLO detection results"""
    
    def __init__(self, config):
        self.config = config
    
    def process_detections(self, results, img_shape: tuple, info=None) -> List[Dict[str, Any]]:
        """
        Process raw YOLO detection results
        
        Args:
            results: Raw YOLO results
            img_shape: (height, width) of original image
            info: Additional information for mapping bboxes
            
        Returns:
            List of processed detection dictionaries
        """
        processed_detections = []
        
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.cpu().numpy()
                
                for i in range(len(boxes.xyxy)):
                    x1, y1, x2, y2 = boxes.xyxy[i]
                    conf = boxes.conf[i]
                    cls = boxes.cls[i]
                    
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    if info is not None:
                        bbox = map_bbox_to_original(bbox, info)
                    bbox = clip_bbox_to_image(bbox, img_shape)
                    
                    if not validate_bbox(bbox, img_shape):
                        continue
                    
                    min_w, min_h = self.config.min_bbox_size
                    if (bbox[2] - bbox[0]) < min_w or (bbox[3] - bbox[1]) < min_h:
                        continue
                    
                    detection = {
                        'bbox': bbox,
                        'score': float(conf),
                        'class_id': int(cls)
                    }
                    
                    processed_detections.append(detection)
        
        if self.config.sort_by_confidence:
            processed_detections.sort(key=lambda x: x['score'], reverse=True)
        
        if self.config.max_det > 0:
            processed_detections = processed_detections[:self.config.max_det]
        
        return processed_detections
    
    def filter_by_confidence(self, detections: List[Dict], 
                           min_confidence: float = None) -> List[Dict]:
        """
        Filter detections by confidence threshold
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold (uses config if None)
            
        Returns:
            Filtered detections
        """
        if min_confidence is None:
            min_confidence = self.config.conf_threshold
        
        return [det for det in detections if det['score'] >= min_confidence]
    
    def filter_by_class(self, detections: List[Dict], 
                       allowed_classes: List[int] = None) -> List[Dict]:
        """
        Filter detections by class ID
        
        Args:
            detections: List of detection dictionaries
            allowed_classes: List of allowed class IDs (None = all classes)
            
        Returns:
            Filtered detections
        """
        if allowed_classes is None:
            return detections
        
        return [det for det in detections if det['class_id'] in allowed_classes]
    
    def remove_overlapping_detections(self, detections: List[Dict], 
                                    iou_threshold: float = None) -> List[Dict]:
        """
        Remove overlapping detections using Non-Maximum Suppression
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for NMS (uses config if None)
            
        Returns:
            Non-overlapping detections
        """
        if iou_threshold is None:
            iou_threshold = self.config.iou_threshold
        
        if len(detections) <= 1:
            return detections
        
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        kept_detections = []
        
        for detection in sorted_detections:
            should_keep = True
            
            for kept_detection in kept_detections:
                if detection['class_id'] != kept_detection['class_id']:
                    continue
                
                iou = self._calculate_iou(detection['bbox'], kept_detection['bbox'])
                
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept_detections.append(detection)
        
        return kept_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Get summary statistics of detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary dictionary
        """
        if not detections:
            return {
                'total_detections': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'confidence_range': (0.0, 0.0)
            }
        
        class_counts = {}
        confidences = []
        
        for det in detections:
            class_id = det['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            confidences.append(det['score'])
        
        return {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'avg_confidence': np.mean(confidences),
            'confidence_range': (min(confidences), max(confidences))
        } 