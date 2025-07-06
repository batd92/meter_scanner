"""
YOLO Model Manager - Model loading and inference with performance optimizations
"""

import os
import time
import gc
import torch
from ultralytics import YOLO
from typing import Optional, Dict, Any
from .yolo_config import YOLOConfig
from .yolo_utils import preprocess_image

class YOLOModelManager:
    """Manages YOLO model loading and inference with performance optimizations"""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model = None
        self.is_loaded = False
        self.inference_times = []
        self.memory_usage = []
        
    def load_model(self) -> bool:
        """
        Load YOLO model with performance optimizations
        
        Returns:
            True if model loaded successfully
        """
        try:
            if not os.path.exists(self.config.model_path):
                print(f"❌ YOLO model not found: {self.config.model_path}")
                return False
            
            # Clear GPU memory if needed
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Load model with optimizations
            self.model = YOLO(self.config.model_path)
            
            if self.config.device == 'cuda':
                self.model.to(self.config.device)
                
                if self.config.enable_half_precision:
                    self.model.half()
                    print("✅ Half precision (FP16) enabled")
                
                if self.config.enable_tensorrt:
                    try:
                        self.model.fuse()
                        print("✅ TensorRT optimization enabled")
                    except Exception as e:
                        print(f"⚠️  TensorRT not available: {e}")
            
            self.is_loaded = True
            
            opt_info = self.config.get_optimization_info()
            print(f"✅ YOLO model loaded from: {self.config.model_path}")
            print(f"⚙️  Config: {self.config}")
            print(f"🚀 Optimizations: {opt_info}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.is_loaded = False
            return False
    
    def is_model_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self.is_loaded and self.model is not None
    
    def infer_image(self, img, **kwargs):
        if not self.is_model_ready():
            print("❌ Model not loaded. Call load_model() first.")
            return None, None
        start_time = time.time()
        try:
            img_rgb, info = preprocess_image(img, self.config.input_size, return_info=True)
            model_params = self.config.get_model_params()
            model_params.update(kwargs)
            if 'device' in kwargs:
                del kwargs['device']
            results = self.model(img_rgb, **model_params)
            inference_time = time.time() - start_time
            if self.config.log_inference_time:
                self.inference_times.append(inference_time)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)
                print(f"⚡ Inference time: {inference_time:.3f}s")
            if self.config.clear_cache_after_inference and self.config.device == 'cuda':
                torch.cuda.empty_cache()
            return results, info
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return None, None
    
    def batch_infer(self, images: list, **kwargs) -> list:
        """
        Run batch inference for multiple images
        
        Args:
            images: List of input images
            **kwargs: Additional inference parameters
            
        Returns:
            List of YOLO results
        """
        if not self.is_model_ready():
            print("❌ Model not loaded. Call load_model() first.")
            return []
        
        if len(images) == 0:
            return []
        
        start_time = time.time()
        
        try:
            processed_images = []
            for img in images:
                img_rgb = preprocess_image(img, self.config.input_size)
                processed_images.append(img_rgb)
            
            model_params = self.config.get_model_params()
            model_params.update(kwargs)
            
            results = self.model(processed_images, device=self.config.device, **model_params)
            
            batch_time = time.time() - start_time
            avg_time = batch_time / len(images)
            print(f"⚡ Batch inference: {len(images)} images in {batch_time:.3f}s (avg: {avg_time:.3f}s/image)")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during batch inference: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Performance statistics dictionary
        """
        stats = {
            'model_loaded': self.is_loaded,
            'device': self.config.device,
            'total_inferences': len(self.inference_times)
        }
        
        if self.inference_times:
            stats.update({
                'avg_inference_time': sum(self.inference_times) / len(self.inference_times),
                'min_inference_time': min(self.inference_times),
                'max_inference_time': max(self.inference_times),
                'recent_inference_times': self.inference_times[-10:]
            })
        
        if self.config.device == 'cuda':
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                stats.update({
                    'gpu_memory_allocated_gb': round(memory_allocated, 2),
                    'gpu_memory_reserved_gb': round(memory_reserved, 2)
                })
            except Exception as e:
                stats['gpu_memory_error'] = str(e)
        
        return stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Model information dictionary
        """
        if not self.is_model_ready():
            return {"error": "Model not loaded"}
        
        try:
            info = {
                'model_path': self.config.model_path,
                'device': self.config.device,
                'model_type': type(self.model).__name__,
                'is_loaded': self.is_loaded,
                'config': str(self.config),
                'optimizations': self.config.get_optimization_info(),
                'performance': self.get_performance_stats()
            }
            
            if hasattr(self.model, 'ckpt_path'):
                info['checkpoint_path'] = self.model.ckpt_path
            
            return info
            
        except Exception as e:
            return {"error": f"Error getting model info: {e}"}
    
    def reload_model(self) -> bool:
        """
        Reload the model with memory cleanup
        
        Returns:
            True if reload successful
        """
        # Cleanup
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
        
        gc.collect()
        
        self.is_loaded = False
        return self.load_model()
    
    def update_config(self, **kwargs) -> bool:
        """
        Update configuration and optionally reload model
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            True if update successful
        """
        try:
            self.config.update(**kwargs)
            print(f"⚙️  Updated config: {self.config}")
            return True
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            return False
    
    def optimize_for_production(self):
        """Apply production optimizations"""
        if not self.is_model_ready():
            return False
        
        try:
            self.config.verbose = False
            self.config.save_debug_images = False
            self.config.enable_profiling = False
            
            self.config.enable_half_precision = True
            self.config.clear_cache_after_inference = True
            
            print("✅ Production optimizations applied")
            return True
            
        except Exception as e:
            print(f"❌ Error applying production optimizations: {e}")
            return False 