"""
Core modules for OCR and meter reading
"""

from .yolo import YOLODetector, YOLOConfig
from .onnxocr.onnx_paddleocr import ONNXPaddleOcr

__all__ = [
    'YOLODetector',
    'YOLOConfig',
    'ONNXPaddleOcr'
] 