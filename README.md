# 📁 Meter Scanner - Unified OCR & YOLO+OCR System

## 1. Giới thiệu
- Hệ thống nhận diện số công tơ nước/điện bằng YOLOv8 + PaddleOCR.
- Web UI, API, batch processing, monitoring hiệu suất.

## 2. Cấu trúc thư mục
```
├── main.py                # Flask app
├── requirements.txt       # Thư viện
├── Dockerfile             # Docker config
├── core/
│   └── yolo/              # YOLO modules & model
│       ├── yolo_detector.py
│       ├── yolo_model.py
│       ├── yolo_postprocessor.py
│       ├── yolo_utils.py
│       └── model/best.pt
│   └── onnxocr/           # PaddleOCR models
├── services/              # Business logic
├── templates/             # HTML UI
├── static/                # CSS, JS
├── uploads/, results/     # Lưu file
```

## 3. Hướng dẫn sử dụng nhanh
### Cài đặt local
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
- Truy cập: http://localhost:5001/

### Chạy bằng Docker
```bash
docker build -t meterscanner .
docker run -p 5001:5001 --gpus all meterscanner
```

## 4. Các tính năng chính
- Nhận diện số công tơ bằng YOLOv8 + OCR
- Web UI: upload ảnh, xem kết quả, batch processing
- API: /api/ocr, /api/yolo_ocr, /api/ocr_only
- Monitoring: theo dõi CPU, RAM, thời gian xử lý
- Tối ưu hiệu suất: GPU/CPU, batch, FP16, TensorRT

## 5. Troubleshooting
- **YOLO model không load:** kiểm tra file `core/yolo/model/best.pt` và phiên bản ultralytics.
- **Chạy CPU chậm:** nên dùng GPU hoặc giảm kích thước ảnh.
- **Port 5001 bị chiếm:** đổi port trong `main.py` hoặc lệnh docker.

## 6. Tham khảo API
- `POST /api/ocr` - Basic OCR
- `POST /api/yolo_ocr` - YOLO+OCR
- `POST /api/ocr_only` - OCR only

## 7. Ghi chú
- Để tối ưu RAM/CPU: bật batch nhỏ, clear cache, dùng GPU nếu có.
- Có thể export YOLO sang ONNX để inference nhanh hơn trên CPU.

---
**Meter Scanner** - Nhanh, gọn, dễ mở rộng.