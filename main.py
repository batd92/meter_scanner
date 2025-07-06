#!/usr/bin/env python3
"""
Main Flask Application - Unified OCR and YOLO+OCR Service
Routes only - Business logic handled by services
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services import BasicOCRProcessor, YOLOOCRProcessor, OCRUtils, WebUIProcessor, performance_monitor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
RESULT_ROOT = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

basic_ocr_service = BasicOCRProcessor()
yolo_ocr_service = YOLOOCRProcessor()
webui_service = WebUIProcessor(RESULT_ROOT)

MODEL_OPTIONS = OCRUtils.get_model_options()

# ==================== PAGE ROUTES ====================

@app.route('/')
def router():
    return render_template('router.html')

@app.route('/basic-ocr')
def basic_ocr_page():
    return render_template('index.html')

@app.route('/yolo-ocr')
def yolo_ocr_page():
    return render_template('yolo_ocr_ui.html')

@app.route('/webui')
def webui_page():
    return render_template('webui.html', model_options=MODEL_OPTIONS)

# ==================== API ROUTES ====================

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    data = request.get_json()
    result = basic_ocr_service.ocr_api(data)
    return jsonify(result)

@app.route('/api/yolo_ocr', methods=['POST'])
def api_yolo_ocr():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            filename = f"yolo_ocr_{int(time.time())}_{file.filename}"
            filepath = os.path.join(UPLOAD_ROOT, filename)
            file.save(filepath)
            
            confidence = float(request.form.get('confidence', 0.5))
            
            result = yolo_ocr_service.yolo_ocr_api(filepath, confidence)
            
            try:
                os.remove(filepath)
            except:
                pass
                
            return jsonify(result)
        
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'success': False, 'error': 'No image data provided'})
            
            import base64
            import numpy as np
            import cv2
            
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'success': False, 'error': 'Invalid image data'})
            
            filename = f"yolo_ocr_{int(time.time())}.jpg"
            filepath = os.path.join(UPLOAD_ROOT, filename)
            cv2.imwrite(filepath, image)
            
            confidence = float(data.get('confidence', 0.5))
            
            result = yolo_ocr_service.yolo_ocr_api(filepath, confidence)
            
            try:
                os.remove(filepath)
            except:
                pass
                
            return jsonify(result)
        
        else:
            return jsonify({'success': False, 'error': 'No file or image data provided'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ocr_only', methods=['POST'])
def api_ocr_only():
    data = request.get_json()
    result = yolo_ocr_service.ocr_only_api(data)
    return jsonify(result)

# ==================== PERFORMANCE MONITORING ROUTES ====================

@app.route('/api/performance/stats', methods=['GET'])
def api_performance_stats():
    """Get performance statistics"""
    stats = performance_monitor.get_performance_summary()
    return jsonify(stats)

@app.route('/api/performance/recent', methods=['GET'])
def api_performance_recent():
    """Get recent performance metrics"""
    count = request.args.get('count', 10, type=int)
    metrics = performance_monitor.get_recent_metrics(count)
    return jsonify(metrics)

@app.route('/api/performance/recommendations', methods=['GET'])
def api_performance_recommendations():
    """Get performance optimization recommendations"""
    recommendations = performance_monitor.get_optimization_recommendations()
    return jsonify({'recommendations': recommendations})

@app.route('/api/performance/export', methods=['POST'])
def api_performance_export():
    """Export performance metrics to file"""
    try:
        data = request.get_json() or {}
        filepath = data.get('filepath', f'performance_metrics_{int(time.time())}.json')
        
        if not os.path.isabs(filepath):
            filepath = os.path.join(RESULT_ROOT, filepath)
        
        performance_monitor.export_metrics(filepath)
        return jsonify({'success': True, 'filepath': filepath})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/performance/optimize', methods=['POST'])
def api_performance_optimize():
    """Apply production optimizations"""
    try:
        yolo_ocr_service.optimize_for_production()
        
        return jsonify({
            'success': True,
            'message': 'Production optimizations applied successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ==================== WEBUI ROUTES ====================

@app.route("/set_model", methods=["POST"])
def set_model():
    model_name = request.form.get("model_name")
    result = webui_service.set_model(model_name)
    return jsonify(result)

@app.route("/ocr", methods=["POST"])
def ocr_files():
    files = request.files.getlist("files")
    model_name = request.form.get("model_name")
    result = webui_service.process_files(files, model_name)
    return jsonify(result)

@app.route("/download/<timestamp>")
def download_zip(timestamp):
    zip_path = webui_service.get_zip_file(timestamp)
    if zip_path:
        return send_file(zip_path, as_attachment=True, download_name=f"ocr_txt_{timestamp}.zip")
    return jsonify({"success": False, "msg": "文件不存在"}), 404

@app.route("/ocr_api", methods=["POST"])
def ocr_api():
    data = request.get_json()
    result = basic_ocr_service.ocr_api(data)
    return jsonify(result)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    path = request.path
    if not path.startswith("/static") and not path.startswith("/download"):
        return redirect(url_for("router"))
    return jsonify({"detail": "NotFound"}), 404

@app.route('/health')
def health_check():
    perf_stats = performance_monitor.get_performance_summary()
    
    return jsonify({
        "status": "healthy",
        "services": {
            "basic_ocr": "loaded",
            "yolo_ocr": "loaded",
            "webui": "loaded",
            "performance_monitor": "active"
        },
        "performance": {
            "uptime_seconds": perf_stats.get('uptime_seconds', 0),
            "avg_cpu_percent": perf_stats.get('averages', {}).get('cpu_percent', 0),
            "avg_memory_percent": perf_stats.get('averages', {}).get('memory_percent', 0)
        }
    })

if __name__ == '__main__':
    print("🚀 Starting unified OCR and YOLO+OCR service on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=False) 