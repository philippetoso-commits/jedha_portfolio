import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add demo directory to path
sys.path.append(os.path.abspath("demo"))

from demo.utils.pipeline import ALPRPipeline

def diagnose_video(video_path):
    print(f"🔍 Starting diagnosis for: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Error: File not found at {video_path}")
        return

    # 1. Basic Metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Metadata: {width}x{height}, {fps} FPS, {frame_count} total frames")
    
    # 2. Pipeline Test
    print("📦 Loading ALPR Pipeline...")
    start_load = time.time()
    pipeline = ALPRPipeline()
    print(f"✅ Pipeline loaded in {time.time() - start_load:.2f}s")
    
    # 3. Process first few frames with timing
    print("🚀 Processing 5 sample frames (with optimization simulation)...")
    
    MAX_HEIGHT = 1080
    
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Could not read frame {i}")
            break
            
        print(f"  --- Frame {i} ---")
        
        # Optimization: Resize if 4K
        t_resize_start = time.time()
        h, w = frame.shape[:2]
        if h > MAX_HEIGHT:
            scale = MAX_HEIGHT / h
            frame = cv2.resize(frame, (int(w * scale), MAX_HEIGHT))
        t_resize = time.time() - t_resize_start
        
        # Plate Detection
        t0 = time.time()
        results = pipeline.process_image(image_array=frame, conf_threshold=0.5, detect_brand=True)
        t_pipe = time.time() - t0
        
        num_plates = len(results.get('step4_ocr', []))
        brand = results.get('detected_brand', 'None')
        
        print(f"  ⏱️ Resize: {t_resize:.3f}s | Pipe: {t_pipe:.3f}s | Total: {t_resize + t_pipe:.3f}s")
        print(f"  📝 Plates: {num_plates} | Brand: {brand}")
    
    cap.release()
    print("\n✅ Diagnosis complete.")

if __name__ == "__main__":
    video_file = "/home/phili/datascience/projet plaque/testvideo/Traffic Control CCTV.mp4"
    diagnose_video(video_file)
