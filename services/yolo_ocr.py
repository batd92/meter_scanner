#!/usr/bin/env python3
"""
YOLO + OCR Processor - Core processing logic for YOLO detection and OCR with performance monitoring
"""

import os
import sys
import base64
import cv2
import numpy as np
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.yolo.yolo_detector import YOLODetector
from core.onnxocr.onnx_paddleocr import ONNXPaddleOcr
from .performance_monitor import performance_monitor, PerformanceMonitor

# Setup logger
logger = logging.getLogger(__name__)

class YOLOOCRProcessor:
    """YOLO + OCR processing logic with performance monitoring"""
    
    def __init__(self):
        """Initialize YOLO and OCR models with performance monitoring"""
        self.yolo_detector = YOLODetector()
        self.ocr_model = ONNXPaddleOcr()
        
        performance_monitor.start_monitoring()
        
        print("✅ YOLO+OCR processor initialized with performance monitoring")
    
    def process_image_ocr_only(self, image):
        """Process image with OCR only (no YOLO detection)"""
        start_time = time.time()
        
        try:
            ocr_result = self.ocr_model.ocr(image)
            
            texts = []
            for line in ocr_result:
                for word_info in line:
                    bbox, (text, confidence) = word_info
                    texts.append({
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': bbox
                    })
            
            processing_time = time.time() - start_time
            
            performance_monitor.record_inference_time('ocr_only', processing_time)
            
            return {
                'success': True,
                'texts': texts,
                'processing_time': processing_time,
                'total_texts': len(texts)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            performance_monitor.record_inference_time('ocr_only_error', processing_time)
            
            print(f"❌ Error in OCR only processing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def process_image_yolo_ocr(self, image_path):
        """Process image with YOLO detection + OCR recognition"""
        start_time = time.time()
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image from path',
                    'processing_time': time.time() - start_time
                }
            
            print(f"📸 Processing image: {image.shape}")
            
            yolo_start = time.time()
            detections = self.yolo_detector.detect(image, conf=0.1, iou=0.3)  # Lower threshold
            yolo_time = time.time() - yolo_start
            
            
            if len(detections) == 0:
                print("🔍 No YOLO detections, trying OCR on full image...")
                try:
                    full_ocr_result = self.ocr_model.ocr(image)
                    full_texts = []
                    for line in full_ocr_result:
                        for word_info in line:
                            bbox, (text, confidence) = word_info
                            full_texts.append({
                                'text': text,
                                'confidence': float(confidence),
                                'bbox': bbox
                            })
                    
                    if full_texts:
                        detections = [{
                            'bbox': [0, 0, image.shape[1], image.shape[0]],
                            'score': 1.0,
                            'class_id': 'full_image'
                        }]
                    else:
                        print("❌ No text found in full image either")
                except Exception as e:
                    print(f"❌ Error in full image OCR: {e}")
            
            result_image = image.copy()
            for detection in detections:
                bbox = detection['bbox']
                cv2.rectangle(result_image, (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            
            cv2.imwrite(f"{debug_dir}/00_original_with_bboxes.jpg", result_image)
            print(f"💾 Saved debug image: {debug_dir}/00_original_with_bboxes.jpg")
            
            image_with_bboxes = self.encode_image_to_base64(result_image)
            
            ocr_start = time.time()
            region_results = []
            for i, detection in enumerate(detections):
                try:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    region = image[y1:y2, x1:x2]
                    if region.size == 0:
                        continue
                    region_filename = f"{debug_dir}/region_{i}_{detection.get('class_id', 'unknown')}_original_crop.jpg"
                    cv2.imwrite(region_filename, region)
                    if detection.get('class_id') == 'full_image':
                        ocr_result = self.ocr_model.ocr(image)
                    else:
                        ocr_result = self.ocr_model.ocr(region)
                    region_texts = []
                    for line in ocr_result:
                        for word_info in line:
                            bbox, (text, confidence) = word_info
                            region_texts.append({
                                'text': text,
                                'confidence': float(confidence),
                                'bbox': bbox
                            })
                    region_results.append({
                        'region_id': i,
                        'bbox': bbox,
                        'class_id': detection['class_id'],
                        'confidence': detection['score'],
                        'texts': region_texts,
                        'total_texts': len(region_texts)
                    })
                except Exception as e:
                    region_results.append({
                        'region_id': i,
                        'bbox': detection['bbox'],
                        'class_id': detection['class_id'],
                        'confidence': detection['score'],
                        'texts': [],
                        'total_texts': 0,
                        'error': str(e)
                    })
            
            ocr_time = time.time() - ocr_start
            total_time = time.time() - start_time
            
            performance_monitor.record_inference_time('yolo_detection', yolo_time)
            performance_monitor.record_inference_time('ocr_processing', ocr_time)
            performance_monitor.record_inference_time('yolo_ocr_total', total_time)
            
            return {
                'success': True,
                'detections': len(detections),
                'regions': region_results,
                'processing_time': total_time,
                'yolo_time': yolo_time,
                'ocr_time': ocr_time,
                'total_texts': sum(region.get('total_texts', 0) for region in region_results),
                'image_with_bboxes': image_with_bboxes
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            performance_monitor.record_inference_time('yolo_ocr_error', total_time)
            
            print(f"❌ Error in YOLO+OCR processing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': total_time
            }
    
    def decode_base64_image(self, base64_image):
        """Decode base64 image to OpenCV format"""
        try:
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            
            image_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, "Invalid image data"
            
            return image, None
            
        except Exception as e:
            return None, f"Error decoding image: {str(e)}"
    
    def encode_image_to_base64(self, image):
        """Encode OpenCV image to base64"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image: {str(e)}")
            return None

    def yolo_ocr_api(self, image_path, confidence_threshold=0.5):
        """
        API function for YOLO + OCR processing with performance monitoring
        Returns new efficient format with performance data
        """
        start_time = time.time()
        
        try:
            # Start performance monitoring
            perf_monitor = PerformanceMonitor()
            perf_monitor.start_monitoring()
            
            # Process image
            result = self.process_image_yolo_ocr(image_path)
            
            # Stop monitoring and get metrics
            perf_monitor.stop_monitoring()
            performance_data = perf_monitor.get_performance_summary()
            
            # Calculate timing
            total_time = time.time() - start_time
            yolo_time = result.get('yolo_time', 0)
            ocr_time = result.get('ocr_time', 0)
            
            # Count detections and texts
            detections = len(result.get('regions', []))
            total_texts = sum(len(region.get('texts', [])) for region in result.get('regions', []))
            
            # Return new efficient format
            response = {
                'success': True,
                'processing_time': total_time,
                'yolo_time': yolo_time,
                'ocr_time': ocr_time,
                'detections': detections,
                'total_texts': total_texts,
                'regions': result.get('regions', []),
                'image_with_bboxes': result.get('image_with_bboxes'),
                'performance': {
                    'avg_cpu': performance_data.get('averages', {}).get('cpu_percent', 0),
                    'avg_memory': performance_data.get('averages', {}).get('memory_percent', 0),
                    'recommendations': performance_monitor.get_optimization_recommendations()
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in YOLO OCR API: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def ocr_only_api(self, data):
        """API endpoint for OCR-only processing"""
        try:
            if 'image' not in data:
                return {'success': False, 'error': 'No image data provided'}
            
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'success': False, 'error': 'Invalid image data'}
            
            result = self.process_image_ocr_only(image)
            
            if result['success']:
                perf_summary = performance_monitor.get_performance_summary()
                result['performance'] = {
                    'avg_cpu': perf_summary.get('averages', {}).get('cpu_percent', 0),
                    'avg_memory': perf_summary.get('averages', {}).get('memory_percent', 0),
                    'recommendations': performance_monitor.get_optimization_recommendations()
                }
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return performance_monitor.get_performance_summary()
    
    def optimize_for_production(self):
        """Apply production optimizations"""
        self.yolo_detector.model_manager.optimize_for_production()
        
        performance_monitor.inference_time_threshold = 1.5  # Stricter threshold for production
        
        print("✅ Production optimizations applied to YOLO+OCR processor") 