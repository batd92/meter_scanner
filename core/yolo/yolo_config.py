"""
YOLO Configuration Management
"""

import torch

class YOLOConfig:
    """Configuration class for YOLO detection parameters"""
    
    def __init__(self):
        # Model settings
        self.model_path = './core/yolo/model/best.pt'
        self.use_onnx = False  # Enable ONNX for better performance
        self.onnx_path = './core/yolo/model/best.onnx'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Detection thresholds - Optimized for meter detection
        self.conf_threshold = 0.25  # Lower for better recall
        self.iou_threshold = 0.45
        self.max_det = 20  # Increased for multiple meters
        
        # Image preprocessing - Optimized for performance
        self.input_size = (640, 640)  # Standard YOLO input size
        self.auto_resize = True  # Enable for consistent performance
        self.pad_to_square = True  # Better for YOLO
        
        # Performance optimizations
        self.enable_half_precision = True  # FP16 for speed
        self.enable_tensorrt = False  # Enable if available
        self.batch_size = 1  # For real-time processing
        self.num_workers = 0  # For single-threaded inference
        
        # Post-processing optimizations
        self.sort_by_confidence = True
        self.filter_invalid_bbox = True
        self.min_bbox_size = (20, 20)  # Increased minimum size
        self.max_bbox_size = (1000, 1000)  # Maximum size limit
        
        # Caching and memory management
        self.enable_model_caching = True
        self.clear_cache_after_inference = False
        self.max_memory_usage = 0.8  # 80% of available memory
        
        # Debug and monitoring
        self.verbose = False
        self.save_debug_images = False  # Disable in production
        self.enable_profiling = False
        self.log_inference_time = True
        
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    def get_model_params(self):
        """Get parameters for model inference"""
        params = {
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'max_det': self.max_det,
            'verbose': self.verbose,
            'device': self.device
        }
        
        # Add performance optimizations
        if self.enable_half_precision and self.device == 'cuda':
            params['half'] = True
            
        return params
    
    def get_optimization_info(self):
        """Get optimization configuration summary"""
        return {
            'device': self.device,
            'use_onnx': self.use_onnx,
            'half_precision': self.enable_half_precision,
            'tensorrt': self.enable_tensorrt,
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'model_caching': self.enable_model_caching
        }
    
    def __str__(self):
        return f"YOLOConfig(conf={self.conf_threshold}, iou={self.iou_threshold}, max_det={self.max_det}, device={self.device})"

# Default configuration instance
DEFAULT_CONFIG = YOLOConfig() 