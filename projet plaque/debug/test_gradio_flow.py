import sys
import os
sys.path.append("demo")

from demo.utils.pipeline import ALPRPipeline
from demo.utils.video_processor import create_annotated_video
from demo.utils.cache import VideoCache

# Simulate what Gradio does
video_path = "demo/video/Animation_de_voiture_entrant_dans_un_parking.mp4"
output_path = "outputs/test_annotated.mp4"

print("🔍 Simulating Gradio video processing...")
print(f"Input: {video_path}")
print(f"Output: {output_path}")

# Initialize
pipeline = ALPRPipeline()
video_cache = VideoCache()

# Check cache
params = {"mode": "annotate", "conf_threshold": 0.5, "num_samples": 10, "detect_brand": False}
video_hash = video_cache.get_hash(video_path, params)
print(f"Video hash: {video_hash[:16]}...")

cached_path = video_cache.get_cached_video(video_hash, ".webm")
if cached_path:
    print(f"⚡ Found in cache: {cached_path}")
else:
    print("📹 Processing video...")
    try:
        stats = create_annotated_video(pipeline, video_path, output_path, conf_threshold=0.5, max_fps=10, detect_brand=False)
        print(f"✅ Success!")
        print(f"   Processed: {stats['processed_frames']}/{stats['total_frames']} frames")
        print(f"   Detections: {stats['total_detections']}")
        
        # Add to cache
        video_cache.add_to_cache(video_hash, output_path)
        print(f"💾 Added to cache")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
