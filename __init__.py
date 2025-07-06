"""
OCR Project - Water Meter Reading with YOLO + PaddleOCR
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.yolo.yolo_infer import YOLODetector
from core.onnxocr.onnx_paddleocr import ONNXPaddleOcr

from services.yolo_ocr_service import app as yolo_ocr_app
from services.app_service import app as basic_ocr_app

__version__ = "1.0.0"
__author__ = "OCR Team"

__all__ = [
    'YOLODetector',
    'ONNXPaddleOcr',
    'yolo_ocr_app',
    'basic_ocr_app'
] 