# 📁 OCR System - Unified Flask Application

## ✅ Cấu trúc hiện tại (Unified Architecture)

```
OCR/
├── README.md                    # 📖 Hướng dẫn chính
├── requirements.txt             # 📦 Dependencies
├── Dockerfile                   # 🐳 Docker config
├── LICENSE                      # 📄 License
├── .gitignore                   # 🚫 Git ignore
├── main.py                      # 🚀 Unified Flask application
│
├── core/                        # 🔧 Core AI modules
│   ├── __init__.py
│   ├── yolo/                    # 🎯 YOLO detection modules
│   │   ├── yolo_detector.py     # Main YOLO detector
│   │   ├── yolo_config.py       # YOLO configuration
│   │   ├── yolo_model.py        # YOLO model manager
│   │   ├── yolo_postprocessor.py # Post-processing
│   │   ├── yolo_utils.py        # YOLO utilities
│   │   └── model/               # YOLO model files
│   │       └── best.pt          # Trained YOLO model
│   └── onnxocr/                 # 📚 PaddleOCR models
│       └── onnx_paddleocr.py    # ONNX PaddleOCR implementation
│
├── services/                    # 🌐 Business logic services
│   ├── __init__.py
│   ├── basic_ocr.py             # 📝 Basic OCR processor
│   ├── yolo_ocr.py              # 🎯 YOLO + OCR processor
│   └── ocr_utils.py             # 🛠️ OCR utilities & WebUI processor
│
├── templates/                   # 🎨 Web UI templates
│   ├── router.html              # 🏠 Main router page
│   ├── index.html               # 📝 Basic OCR UI
│   ├── yolo_ocr_ui.html         # 🎯 YOLO + OCR UI
│   └── webui.html               # 🌐 Web UI for file upload
│
├── static/                      # 📁 Static files (CSS, JS, images)
│   └── webui.css                # WebUI stylesheet
│
├── uploads/                     # 📤 Upload directory
├── results/                     # 📥 Results directory
├── debug_images/                # 🐛 Debug images
└── venv/                        # 🐍 Virtual environment
```

## 🏗️ Architecture Overview

### **Clean Architecture Pattern:**
- **`main.py`** - Flask application với routes only
- **`/services`** - Business logic và processing services
- **`/core`** - Core AI models và functionality
- **`/templates`** - Web UI templates

### **Service Layer:**
- **`BasicOCRProcessor`** - Xử lý OCR cơ bản
- **`YOLOOCRProcessor`** - Xử lý YOLO detection + OCR
- **`WebUIProcessor`** - Xử lý file upload và batch processing

## 🚀 Cách sử dụng

### **1. Khởi động ứng dụng**
```bash
# Kích hoạt virtual environment
source venv/bin/activate

# Chạy unified Flask app
python main.py
```

### **2. Truy cập các services**
- **Trang chủ**: `http://localhost:5001/`
- **Basic OCR**: `http://localhost:5001/basic-ocr`
- **YOLO + OCR**: `http://localhost:5001/yolo-ocr`
- **Web UI**: `http://localhost:5001/webui`

### **3. API Endpoints**
- **Basic OCR API**: `POST /api/ocr`
- **YOLO + OCR API**: `POST /api/yolo_ocr`
- **OCR Only API**: `POST /api/ocr_only`
- **Health Check**: `GET /health`

## 🎯 Tính năng chính

### **1. Basic OCR Service**
- Nhận dạng text từ ảnh
- Hỗ trợ nhiều định dạng ảnh
- Trả về text với confidence scores

### **2. YOLO + OCR Service**
- YOLO object detection trước
- OCR trên từng region được detect
- Hiển thị bounding boxes và text results

### **3. Web UI Service**
- Upload nhiều file cùng lúc
- Batch processing
- Download kết quả dạng ZIP
- Preview ảnh và text

## 🔧 Cài đặt và Setup

### **1. Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

### **2. Setup YOLO model**
```bash
# Đảm bảo YOLO model đã có trong core/yolo/model/best.pt
```

### **3. Chạy ứng dụng**
```bash
python main.py
```

## 📊 Response Formats

### **Basic OCR Response:**
```json
{
  "success": true,
  "processing_time": 0.123,
  "results": [
    {
      "text": "recognized text",
      "confidence": 0.95,
      "bounding_box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ],
  "total_texts": 1
}
```

### **YOLO + OCR Response:**
```json
{
  "success": true,
  "processing_time": {
    "total": 0.456,
    "yolo_detection": 0.123,
    "ocr_recognition": 0.333
  },
  "yolo_detections": 2,
  "image_with_bboxes": "base64_encoded_image",
  "results": [
    {
      "region_id": 0,
      "yolo_confidence": 0.95,
      "yolo_bbox": [x1, y1, x2, y2],
      "texts": [
        {
          "text": "detected text",
          "confidence": 0.88,
          "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        }
      ]
    }
  ]
}
```

## 🎨 UI Features

### **Router Page (`/`)**
- Trang chủ đơn giản với links đến các services
- Clean design, responsive

### **Basic OCR UI (`/basic-ocr`)**
- Drag & drop ảnh
- Real-time OCR processing
- Hiển thị kết quả với confidence

### **YOLO + OCR UI (`/yolo-ocr`)**
- Upload ảnh
- YOLO detection + OCR processing
- Hiển thị bounding boxes và text regions

### **Web UI (`/webui`)**
- Multi-file upload
- Progress bar
- Batch processing
- Download results as ZIP

## 🔍 Troubleshooting

### **Port 5000/5001 đã được sử dụng:**
```bash
# Kiểm tra process đang chạy
lsof -i :5001

# Dừng process
kill -9 <PID>
```

### **YOLO model không load:**
- Đảm bảo file `core/yolo/model/best.pt` tồn tại
- Kiểm tra model path trong code

### **OCR model warning:**
- Warning về CUDA provider là bình thường trên macOS
- Model sẽ tự động fallback về CPU

## 📝 Notes

- **Unified Architecture**: Tất cả services chạy trên cùng một Flask app
- **Clean Separation**: Business logic tách biệt khỏi routes
- **Port 5001**: Tránh conflict với AirPlay Receiver trên macOS
- **File Upload**: Hỗ trợ multiple files và batch processing
- **Error Handling**: Comprehensive error handling và user feedback

## 🚀 **Performance Optimizations**

### **Model Optimizations**
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Half Precision (FP16)**: 2x faster inference with minimal accuracy loss
- **TensorRT Support**: Optional TensorRT optimization for maximum speed
- **Model Caching**: Intelligent model caching to reduce loading times
- **Memory Management**: Automatic GPU memory cleanup and optimization

### **Inference Optimizations**
- **Batch Processing**: Support for batch inference on multiple images
- **Optimized Preprocessing**: Smart image resizing and padding
- **Post-processing Filters**: Enhanced NMS and bbox filtering
- **Performance Monitoring**: Real-time inference time tracking

### **System Optimizations**
- **Performance Monitoring**: Real-time CPU, memory, and disk usage tracking
- **Automatic Alerts**: Performance threshold monitoring and alerts
- **Optimization Recommendations**: AI-powered performance suggestions
- **Production Mode**: One-click production optimizations

## 📁 **Project Structure**

```
OCR/
├── main.py                 # Unified Flask application
├── services/               # Business logic services
│   ├── basic_ocr_processor.py
│   ├── yolo_ocr.py
│   ├── ocr_utils.py
│   ├── performance_monitor.py  # NEW: Performance monitoring
│   └── __init__.py
├── core/                   # Core AI models
│   ├── yolo/              # YOLO detection with optimizations
│   │   ├── yolo_config.py     # Enhanced configuration
│   │   ├── yolo_model.py      # Optimized model manager
│   │   ├── yolo_detector.py   # Main detection interface
│   │   ├── yolo_postprocessor.py
│   │   ├── yolo_utils.py      # Enhanced preprocessing
│   │   └── __init__.py
│   └── onnxocr/           # ONNX OCR models
├── templates/             # HTML templates
├── static/               # CSS, JS, images
├── uploads/              # Upload directory
├── results/              # Results directory
├── requirements.txt      # Updated dependencies
└── Dockerfile           # Updated for unified app
```

## 🏗️ **Architecture Overview**

### **Unified Flask Application**
- **Single Entry Point**: `main.py` handles all routes and APIs
- **Service Layer**: Business logic separated into service classes
- **Core Layer**: AI models and utilities in `/core`
- **Performance Monitoring**: Real-time system and inference monitoring

### **Service Architecture**
```
main.py (Routes) → services/ (Business Logic) → core/ (AI Models)
```

### **Performance Monitoring**
- **Real-time Metrics**: CPU, memory, disk, network usage
- **Inference Tracking**: Per-service inference time monitoring
- **Alert System**: Automatic performance issue detection
- **Optimization Engine**: AI-powered performance recommendations

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone repository
git clone <repository-url>
cd OCR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Running the Application**
```bash
# Start the unified service
python main.py

# Access the application
# Main Router: http://localhost:5001
# Basic OCR: http://localhost:5001/basic-ocr
# YOLO+OCR: http://localhost:5001/yolo-ocr
# Web UI: http://localhost:5001/webui
```

### **Performance Monitoring**
```bash
# Get performance statistics
curl http://localhost:5001/api/performance/stats

# Get optimization recommendations
curl http://localhost:5001/api/performance/recommendations

# Apply production optimizations
curl -X POST http://localhost:5001/api/performance/optimize
```

## 📊 **API Endpoints**

### **Core Services**
- `POST /api/ocr` - Basic OCR processing
- `POST /api/yolo_ocr` - YOLO+OCR processing
- `POST /api/ocr_only` - OCR-only processing

### **Performance Monitoring**
- `GET /api/performance/stats` - Performance statistics
- `GET /api/performance/recent` - Recent metrics
- `GET /api/performance/recommendations` - Optimization suggestions
- `POST /api/performance/export` - Export metrics
- `POST /api/performance/optimize` - Apply optimizations

### **Web UI**
- `POST /ocr` - File upload processing
- `POST /set_model` - Model selection
- `GET /download/<timestamp>` - Download results

### **Health & Status**
- `GET /health` - Health check with performance metrics
- `GET /` - Main router page

## ⚡ **Performance Features**

### **Model Optimizations**
```python
# Automatic GPU detection and optimization
config = YOLOConfig()
# - GPU acceleration (if available)
# - Half precision (FP16)
# - TensorRT optimization
# - Memory management
```

### **Inference Optimizations**
```python
# Batch processing support
results = model_manager.batch_infer(images)

# Performance tracking
performance_monitor.record_inference_time('service_name', inference_time)

# Automatic optimization recommendations
recommendations = performance_monitor.get_optimization_recommendations()
```

### **Production Optimizations**
```python
# One-click production mode
yolo_ocr_service.optimize_for_production()
# - Disables debug features
# - Enables performance optimizations
# - Adjusts thresholds for production
```

## 🔧 **Configuration**

### **YOLO Configuration**
```python
# Enhanced configuration with performance options
config = YOLOConfig()
config.device = 'cuda'  # Auto-detect GPU
config.enable_half_precision = True
config.enable_tensorrt = False  # Enable if available
config.input_size = (640, 640)  # Optimized input size
config.batch_size = 1  # Real-time processing
```

### **Performance Thresholds**
```python
# Configurable performance thresholds
performance_monitor.cpu_threshold = 80.0
performance_monitor.memory_threshold = 85.0
performance_monitor.inference_time_threshold = 2.0
```

## 📈 **Performance Monitoring**

### **Real-time Metrics**
- **CPU Usage**: Process and system CPU utilization
- **Memory Usage**: RAM and GPU memory tracking
- **Disk Usage**: Storage space monitoring
- **Network I/O**: Network traffic tracking
- **Inference Times**: Per-service performance tracking

### **Automatic Alerts**
- High CPU usage (>80%)
- High memory usage (>85%)
- Slow inference times (>2s)
- Process memory leaks (>1GB)

### **Optimization Recommendations**
- GPU acceleration suggestions
- Batch size optimization
- Memory cleanup recommendations
- Model optimization tips

## 🐳 **Docker Deployment**

### **Build and Run**
```bash
# Build Docker image
docker build -t ocr-app .

# Run container
docker run -p 5001:5001 --gpus all ocr-app
```

### **Production Deployment**
```bash
# Apply production optimizations
curl -X POST http://localhost:5001/api/performance/optimize

# Monitor performance
curl http://localhost:5001/api/performance/stats
```

## 🔍 **Troubleshooting**

### **Performance Issues**
1. **Check GPU availability**: `nvidia-smi`
2. **Monitor system resources**: `/api/performance/stats`
3. **Apply optimizations**: `/api/performance/optimize`
4. **Check recommendations**: `/api/performance/recommendations`

### **Common Issues**
- **Port conflicts**: Change port in `main.py`
- **Memory issues**: Enable memory cleanup in config
- **Slow inference**: Enable GPU acceleration or half precision
- **Model loading errors**: Check model file paths

### **Performance Tuning**
```python
# For high-throughput scenarios
config.batch_size = 4
config.enable_half_precision = True
config.clear_cache_after_inference = True

# For real-time scenarios
config.batch_size = 1
config.enable_half_precision = True
config.input_size = (416, 416)  # Smaller input
```

## 📝 **Notes**

### **Performance Optimizations**
- **GPU Acceleration**: Automatically detected and utilized
- **Memory Management**: Intelligent cleanup and optimization
- **Batch Processing**: Support for multiple image processing
- **Real-time Monitoring**: Continuous performance tracking

### **Production Ready**
- **Performance Monitoring**: Real-time system health tracking
- **Automatic Optimization**: AI-powered performance suggestions
- **Scalable Architecture**: Easy to extend and maintain
- **Docker Support**: Containerized deployment

### **Future Enhancements**
- **Model Quantization**: INT8 quantization for faster inference
- **Distributed Processing**: Multi-GPU support
- **Advanced Caching**: Redis-based model caching
- **Load Balancing**: Multiple instance support

---

**Performance Optimized OCR & YOLO+OCR System** - Built for speed, efficiency, and scalability. 