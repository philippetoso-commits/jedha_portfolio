#!/usr/bin/env python3
"""
Batch process images to extract license plate numbers and save to CSV.
"""

import sys
import os
from pathlib import Path
import csv
from datetime import datetime

# Add demo utils to path
demo_path = Path(__file__).parent / "demo"
sys.path.insert(0, str(demo_path))

from utils.pipeline import ALPRPipeline

def process_batch(input_dir, output_csv, conf_threshold=0.3):
    """
    Process all images in a directory and extract license plates.
    
    Args:
        input_dir: Directory containing images
        output_csv: Path to output CSV file
        conf_threshold: Confidence threshold for detection
    """
    
    print("=" * 80)
    print("🚗 Batch License Plate Extraction")
    print("=" * 80)
    
    # Initialize pipeline
    print("\n📦 Initializing ALPR pipeline...")
    pipeline = ALPRPipeline()
    
    # Get all images
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    images = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    if not images:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"✅ Found {len(images)} images to process")
    print(f"📊 Confidence threshold: {conf_threshold}")
    print(f"💾 Output CSV: {output_csv}\n")
    
    # Prepare CSV
    results = []
    
    # Process each image
    total_plates = 0
    images_with_plates = 0
    
    for i, img_path in enumerate(images, 1):
        print(f"Processing [{i}/{len(images)}]: {img_path.name}...", end=" ")
        
        try:
            # Process image
            pipeline_results = pipeline.process_image(str(img_path), conf_threshold=conf_threshold)
            
            # Extract OCR results
            ocr_results = pipeline_results.get('step4_ocr', [])
            
            if ocr_results:
                images_with_plates += 1
                for j, ocr in enumerate(ocr_results):
                    plate_text = ocr.get('text', '').strip()
                    detection_conf = ocr.get('detection_confidence', 0)
                    ocr_conf = ocr.get('confidence', 0)
                    
                    if plate_text:  # Only save if text was extracted
                        total_plates += 1
                        results.append({
                            'image_filename': img_path.name,
                            'plate_number': plate_text,
                            'detection_confidence': f"{detection_conf:.4f}",
                            'ocr_confidence': f"{ocr_conf:.4f}",
                            'plate_index': j + 1
                        })
                        print(f"✅ Found: '{plate_text}'")
                    else:
                        print(f"⚠️ Detected but no text")
            else:
                print("❌ No plates")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Write CSV
    print(f"\n{'='*80}")
    print("💾 Writing results to CSV...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = ['image_filename', 'plate_number', 'detection_confidence', 
                         'ocr_confidence', 'plate_index']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Summary
    print(f"{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total images processed: {len(images)}")
    print(f"Images with plates detected: {images_with_plates}")
    print(f"Total plates extracted: {total_plates}")
    print(f"Detection rate: {images_with_plates/len(images)*100:.1f}%")
    print(f"Average plates per image: {total_plates/len(images):.2f}")
    print(f"\n✅ Results saved to: {output_csv}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Configuration
    input_directory = "sample"
    output_file = "plaques_extraites.csv"
    confidence = 0.3  # Lower threshold to catch more plates
    
    # Run batch processing
    process_batch(input_directory, output_file, confidence)
