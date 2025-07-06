#!/usr/bin/env python3
"""
Basic OCR Processor - Core processing logic for basic OCR
"""

import os
import sys
import cv2
import time
import base64
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.onnxocr.onnx_paddleocr import ONNXPaddleOcr

class BasicOCRProcessor:
    """Basic OCR processing logic"""
    
    def __init__(self):
        """Initialize OCR model"""
        self.model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
    
    def process_image(self, image):
        """Process image with OCR"""
        try:
            start_time = time.time()
            
            print(f"📸 Processing image with OCR: {image.shape}")
            
            result = self.model.ocr(image)
            processing_time = time.time() - start_time
            
            ocr_results = []
            if result and len(result) > 0:
                for line in result[0]:
                    if isinstance(line[0], (list, np.ndarray)):
                        bounding_box = np.array(line[0]).reshape(4, 2).tolist()
                    else:
                        bounding_box = []
                    
                    ocr_results.append({
                        "text": line[1][0],
                        "confidence": float(line[1][1]),
                        "bounding_box": bounding_box
                    })
            
            return {
                'success': True,
                'processing_time': processing_time,
                'results': ocr_results,
                'total_texts': len(ocr_results)
            }
            
        except Exception as e:
            print(f"❌ Error in OCR processing: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def decode_base64_image(self, base64_image):
        """Decode base64 image to OpenCV format"""
        try:
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            
            image_bytes = base64.b64decode(base64_image)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, "Failed to decode image from base64"
            
            return image, None
            
        except Exception as e:
            return None, f"Image decoding failed: {str(e)}"
    
    def encode_image_to_base64(self, image):
        """Encode OpenCV image to base64"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image: {str(e)}")
            return None
    
    def ocr_api(self, data):
        """API logic: nhận dict, decode ảnh, gọi process_image, trả dict kết quả"""
        if not data or 'image' not in data:
            return {"success": False, "error": "No image data provided"}
        image, error = self.decode_base64_image(data["image"])
        if error:
            return {"success": False, "error": error}
        result = self.process_image(image)
        return result 