import sys
sys.path.append("demo")

from demo.utils.pipeline import ALPRPipeline
from demo.utils.video_processor import sample_video_frames

# Test with existing video
video_path = "demo/video/Animation_de_voiture_entrant_dans_un_parking.mp4"

print("🔍 Testing video processing...")
pipeline = ALPRPipeline()

try:
    results = sample_video_frames(pipeline, video_path, num_samples=3, conf_threshold=0.5, detect_brand=False)
    print(f"✅ Success! Processed {len(results)} frames")
    for r in results:
        print(f"  Frame {r['frame_number']}: {len(r.get('step4_ocr', []))} plates")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
