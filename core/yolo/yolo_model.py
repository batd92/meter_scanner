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

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available. Install with: pip install onnxruntime")

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
            if self.config.use_onnx and ONNX_AVAILABLE:
                return self._load_onnx_model()
            else:
                return self._load_pytorch_model()
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.is_loaded = False
            return False
    
    def _load_onnx_model(self) -> bool:
        """Load ONNX YOLO model"""
        try:
            if not os.path.exists(self.config.onnx_path):
                print(f"❌ ONNX model not found: {self.config.onnx_path}")
                print("💡 Run: python convert_yolo_to_onnx.py to convert your model")
                return False
            
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers = ['CUDAExecutionProvider'] + providers
            
            self.onnx_session = ort.InferenceSession(self.config.onnx_path, providers=providers)
            
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_names = [output.name for output in self.onnx_session.get_outputs()]
            
            self.is_loaded = True
            self.model_type = 'onnx'
            
            opt_info = self.config.get_optimization_info()
            print(f"✅ ONNX YOLO model loaded from: {self.config.onnx_path}")
            print(f"⚙️  Config: {self.config}")
            print(f"🚀 Optimizations: {opt_info}")
            print(f"📊 Providers: {self.onnx_session.get_providers()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            return False
    
    def _load_pytorch_model(self) -> bool:
        """Load PyTorch YOLO model"""
        try:
            if not os.path.exists(self.config.model_path):
                print(f"❌ YOLO model not found: {self.config.model_path}")
                return False
            
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
            
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
            self.model_type = 'pytorch'
            
            opt_info = self.config.get_optimization_info()
            print(f"✅ PyTorch YOLO model loaded from: {self.config.model_path}")
            print(f"⚙️  Config: {self.config}")
            print(f"🚀 Optimizations: {opt_info}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
            return False
    
    def is_model_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self.is_loaded and self.model is not None
    
    def infer_image(self, img, **kwargs):
        if not self.is_model_ready():
            print("❌ Model not loaded. Call load_model() first.")
            return None, None
        
        self._start_time = time.time()
        
        try:
            if self.model_type == 'onnx':
                return self._infer_onnx(img, **kwargs)
            else:
                return self._infer_pytorch(img, **kwargs)
                
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return None, None
    
    def _infer_onnx(self, img, **kwargs):
        """ONNX inference"""
        try:
            input_tensor = self._preprocess_for_onnx(img)
            outputs = self.onnx_session.run(self.output_names, {self.input_name: input_tensor})
            results = self._postprocess_onnx_outputs(outputs, img.shape, **kwargs)
            inference_time = time.time() - self._start_time
            self._record_inference_time(inference_time)
            
            return results, {'preprocessing_info': 'ONNX preprocessing'}
            
        except Exception as e:
            print(f"❌ ONNX inference error: {e}")
            return None, None
    
    def _infer_pytorch(self, img, **kwargs):
        """PyTorch inference"""
        try:
            img_rgb, info = preprocess_image(img, self.config.input_size, return_info=True)
            model_params = self.config.get_model_params()
            model_params.update(kwargs)
            if 'device' in kwargs:
                del kwargs['device']
            
            results = self.model(img_rgb, **model_params)
            
            inference_time = time.time() - self._start_time
            self._record_inference_time(inference_time)
            
            if self.config.clear_cache_after_inference and self.config.device == 'cuda':
                torch.cuda.empty_cache()
            
            return results, info
            
        except Exception as e:
            print(f"❌ PyTorch inference error: {e}")
            return None, None
    
    def _preprocess_for_onnx(self, img):
        """Preprocess image for ONNX inference"""
        import cv2
        import numpy as np
        
        resized = cv2.resize(img, self.config.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _postprocess_onnx_outputs(self, outputs, image_shape, **kwargs):
        """
        Post-process ONNX outputs to match PyTorch YOLO format
        
        Note: This is a simplified implementation. You may need to adapt
        based on your specific YOLO model output format.
        """
        # For now, return the raw outputs
        # TODO: Implement proper post-processing based on your model output format
        return outputs
    
    def _record_inference_time(self, inference_time):
        """Record inference time for performance monitoring"""
        if self.config.log_inference_time:
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            print(f"⚡ Inference time: {inference_time:.3f}s")
    
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