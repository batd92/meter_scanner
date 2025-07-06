"""
YOLO Detection Module

This module provides YOLO-based object detection functionality for the OCR system.
It includes the YOLODetector class for detecting meter regions in images.
"""

from .yolo_detector import YOLODetector
from .yolo_config import YOLOConfig, DEFAULT_CONFIG
from .yolo_model import YOLOModelManager
from .yolo_postprocessor import YOLOPostProcessor
from .yolo_utils import (
    preprocess_image,
    validate_bbox,
    clip_bbox_to_image,
    add_padding_to_bbox,
    crop_image_with_bbox,
    draw_bbox_on_image,
    calculate_bbox_area,
    calculate_bbox_center
)

__all__ = [
    'YOLODetector',
    'YOLOConfig',
    'DEFAULT_CONFIG',
    'YOLOModelManager',
    'YOLOPostProcessor',
    'preprocess_image',
    'validate_bbox',
    'clip_bbox_to_image',
    'add_padding_to_bbox',
    'crop_image_with_bbox',
    'draw_bbox_on_image',
    'calculate_bbox_area',
    'calculate_bbox_center'
] 