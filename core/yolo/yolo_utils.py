"""
YOLO Utilities - Image preprocessing and bounding box operations with performance optimizations
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

def preprocess_image(img: np.ndarray, target_size: Tuple[int, int] = None, pad_to_square: bool = True, return_info: bool = False):
    """
    Preprocess image for YOLO inference with performance optimizations
    If return_info=True, also return scale/pad info for bbox mapping.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    info = {
        'orig_w': w, 'orig_h': h,
        'scale_x': 1.0, 'scale_y': 1.0,
        'pad_x': 0, 'pad_y': 0,
        'new_w': w, 'new_h': h
    }
    if target_size is not None:
        target_w, target_h = target_size
        if pad_to_square:
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
            img_rgb = canvas
            info.update({
                'scale_x': scale, 'scale_y': scale,
                'pad_x': pad_x, 'pad_y': pad_y,
                'new_w': new_w, 'new_h': new_h
            })
        else:
            img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            info.update({
                'scale_x': target_w / w,
                'scale_y': target_h / h,
                'new_w': target_w, 'new_h': target_h
            })
    if return_info:
        return img_rgb, info
    return img_rgb

def map_bbox_to_original(bbox, info):
    """Map bbox from resized/padded image to original image coordinates."""
    x1, y1, x2, y2 = bbox
    x1 = (x1 - info['pad_x']) / info['scale_x']
    x2 = (x2 - info['pad_x']) / info['scale_x']
    y1 = (y1 - info['pad_y']) / info['scale_y']
    y2 = (y2 - info['pad_y']) / info['scale_y']
    
    x1 = int(np.clip(x1, 0, info['orig_w']-1))
    x2 = int(np.clip(x2, 0, info['orig_w']-1))
    y1 = int(np.clip(y1, 0, info['orig_h']-1))
    y2 = int(np.clip(y2, 0, info['orig_h']-1))
    return [x1, y1, x2, y2]

def preprocess_batch(images: List[np.ndarray], target_size: Tuple[int, int] = None,
                    pad_to_square: bool = True) -> List[np.ndarray]:
    """
    Preprocess a batch of images for YOLO inference
    
    Args:
        images: List of input images (BGR format)
        target_size: Target size (width, height) for resizing
        pad_to_square: Whether to pad to square
        
    Returns:
        List of preprocessed images (RGB format)
    """
    processed_images = []
    for img in images:
        processed = preprocess_image(img, target_size, pad_to_square)
        processed_images.append(processed)
    return processed_images

def validate_bbox(bbox: List[int], img_shape: Tuple[int, int]) -> bool:
    """
    Validate bounding box coordinates with enhanced checks
    
    Args:
        bbox: [x1, y1, x2, y2]
        img_shape: (height, width)
        
    Returns:
        True if bbox is valid
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    h, w = img_shape
    
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    min_size = 20
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return False
    
    max_size = min(w, h) * 0.8
    if (x2 - x1) > max_size or (y2 - y1) > max_size:
        return False
    
    return True

def clip_bbox_to_image(bbox: List[int], img_shape: Tuple[int, int]) -> List[int]:
    """
    Clip bounding box coordinates to image bounds
    
    Args:
        bbox: [x1, y1, x2, y2]
        img_shape: (height, width)
        
    Returns:
        Clipped bbox coordinates
    """
    x1, y1, x2, y2 = bbox
    h, w = img_shape
    
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return [int(x1), int(y1), int(x2), int(y2)]

def add_padding_to_bbox(bbox: List[int], padding_ratio: float = 0.2,
                       img_shape: Tuple[int, int] = None) -> List[int]:
    """
    Add padding around bounding box with enhanced bounds checking
    
    Args:
        bbox: [x1, y1, x2, y2]
        padding_ratio: Padding as ratio of bbox size
        img_shape: (height, width) for bounds checking
        
    Returns:
        Padded bbox coordinates
    """
    x1, y1, x2, y2 = bbox
    
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    
    new_x1 = x1 - pad_x
    new_y1 = y1 - pad_y
    new_x2 = x2 + pad_x
    new_y2 = y2 + pad_y
    
    if img_shape is not None:
        h, w = img_shape
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)
    
    return [new_x1, new_y1, new_x2, new_y2]

def crop_image_with_bbox(img: np.ndarray, bbox: List[int], 
                        padding_ratio: float = 0.2) -> np.ndarray:
    """
    Crop image using bounding box with enhanced padding
    
    Args:
        img: Input image
        bbox: [x1, y1, x2, y2]
        padding_ratio: Padding ratio
        
    Returns:
        Cropped image
    """
    h, w = img.shape[:2]
    
    # Add padding to bbox
    padded_bbox = add_padding_to_bbox(bbox, padding_ratio, (h, w))
    x1, y1, x2, y2 = [int(coord) for coord in padded_bbox]
    
    # Ensure valid coordinates
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w))
    y2 = max(y1+1, min(y2, h))
    
    cropped = img[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    
    return cropped

def draw_bbox_on_image(img: np.ndarray, bbox: List[int], 
                      label: str = None, color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on image with enhanced visualization
    
    Args:
        img: Input image
        bbox: [x1, y1, x2, y2]
        label: Optional label text
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with drawn bbox
    """
    img_copy = img.copy()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        text_bg_x1 = x1
        text_bg_y1 = max(0, y1 - text_height - 10)
        text_bg_x2 = x1 + text_width + 10
        text_bg_y2 = y1
        
        cv2.rectangle(img_copy, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        
        cv2.putText(img_copy, label, (x1 + 5, y1 - 5), 
                   font, font_scale, (0, 0, 0), font_thickness)
    
    return img_copy

def calculate_bbox_area(bbox: List[int]) -> float:
    """Calculate area of bounding box"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def calculate_bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """Calculate center point of bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def calculate_bbox_aspect_ratio(bbox: List[int]) -> float:
    """Calculate aspect ratio of bounding box (width/height)"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width / height if height > 0 else 0

def filter_bboxes_by_size(bboxes: List[List[int]], img_shape: Tuple[int, int],
                         min_size: Tuple[int, int] = (20, 20),
                         max_size: Tuple[int, int] = None) -> List[List[int]]:
    """
    Filter bounding boxes by size constraints
    
    Args:
        bboxes: List of bbox coordinates [x1, y1, x2, y2]
        img_shape: (height, width) of image
        min_size: (min_width, min_height)
        max_size: (max_width, max_height) - if None, uses 80% of image size
        
    Returns:
        Filtered list of bboxes
    """
    if max_size is None:
        h, w = img_shape
        max_size = (int(w * 0.8), int(h * 0.8))
    
    filtered_bboxes = []
    min_w, min_h = min_size
    max_w, max_h = max_size
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if (width >= min_w and height >= min_h and 
            width <= max_w and height <= max_h):
            filtered_bboxes.append(bbox)
    
    return filtered_bboxes

def sort_bboxes_by_area(bboxes: List[List[int]], reverse: bool = True) -> List[List[int]]:
    """
    Sort bounding boxes by area
    
    Args:
        bboxes: List of bbox coordinates
        reverse: True for descending order (largest first)
        
    Returns:
        Sorted list of bboxes
    """
    return sorted(bboxes, key=lambda bbox: calculate_bbox_area(bbox), reverse=reverse)

def merge_overlapping_bboxes(bboxes: List[List[int]], iou_threshold: float = 0.5) -> List[List[int]]:
    """
    Merge overlapping bounding boxes
    
    Args:
        bboxes: List of bbox coordinates
        iou_threshold: IoU threshold for merging
        
    Returns:
        List of merged bboxes
    """
    if len(bboxes) <= 1:
        return bboxes
    
    detections = [{'bbox': bbox, 'score': 1.0} for bbox in bboxes]
    
    from .yolo_postprocessor import YOLOPostProcessor
    postprocessor = YOLOPostProcessor(None)
    
    merged_detections = postprocessor.remove_overlapping_detections(detections, iou_threshold)
    
    return [det['bbox'] for det in merged_detections] 