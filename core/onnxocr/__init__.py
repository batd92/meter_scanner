"""
ONNX OCR module for PaddleOCR inference
"""

from .onnx_paddleocr import ONNXPaddleOcr
from .predict_system import TextSystem
from .predict_det import TextDetector
from .predict_rec import TextRecognizer
from .predict_cls import TextClassifier

__all__ = [
    'ONNXPaddleOcr',
    'TextSystem', 
    'TextDetector',
    'TextRecognizer',
    'TextClassifier'
]
