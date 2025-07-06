#!/usr/bin/env python3
"""
OCR Utilities - Common utilities and WebUI processing
"""

import os
import sys
import cv2
import time
import zipfile
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .basic_ocr import BasicOCRProcessor

class OCRUtils:
    """Common OCR utilities"""
    
    @staticmethod
    def get_model_options():
        """Get available model options"""
        return ["PP-OCRv5", "PP-OCRv4", "ch_ppocr_server_v2.0"]
    
    @staticmethod
    def validate_image_file(filename):
        """Validate if file is a supported image"""
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions

class WebUIProcessor:
    """WebUI file processing logic"""
    
    def __init__(self, result_root):
        """Initialize with result directory"""
        self.result_root = result_root
        self.ocr_processor = BasicOCRProcessor()
    
    def process_files(self, files, model_name):
        """Process multiple files with OCR"""
        if not files or not model_name:
            return {
                "success": False, 
                "msg": "缺少文件或模型参数"
            }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.result_root, timestamp)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save uploaded files
        file_paths = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_dir, filename)
                file.save(file_path)
                file_paths.append(file_path)
        
        results = []
        txt_files = []
        
        # Process each file with OCR
        for file_path in file_paths:
            try:
                # Read image
                image = cv2.imread(file_path)
                if image is None:
                    continue
                    
                result = self.ocr_processor.process_image(image)
                
                if not result['success']:
                    continue
                
                text_content = ""
                for item in result['results']:
                    text_content += item['text'] + " "
                
                txt_filename = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
                txt_path = os.path.join(session_dir, txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_content.strip())
                
                txt_files.append(txt_path)
                results.append({
                    "filename": txt_filename, 
                    "content": text_content.strip()
                })
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        zip_path = os.path.join(session_dir, f"ocr_txt_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for txt_file in txt_files:
                zipf.write(txt_file, os.path.basename(txt_file))
        
        return {
            "success": True,
            "results": results,
            "zip_url": f"/download/{timestamp}",
            "timestamp": timestamp
        }
    
    def get_zip_file(self, timestamp):
        """Get zip file path for download"""
        session_dir = os.path.join(self.result_root, timestamp)
        zip_path = os.path.join(session_dir, f"ocr_txt_{timestamp}.zip")
        return zip_path if os.path.exists(zip_path) else None

    def set_model(self, model_name):
        """Set model for OCR (dummy logic, just for API compatibility)"""
        try:
            return {"success": True, "msg": f"模型已切换为 {model_name}"}
        except Exception as e:
            return {"success": False, "msg": str(e)} 