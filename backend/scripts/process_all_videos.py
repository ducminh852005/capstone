import os
import sys
import glob
from pathlib import Path

# Add backend directory to sys.path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.video_processor import standardize_video_fps

def process_all_videos():
    project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    raw_dir = project_root / "data" / "raw"
    cfr_dir = project_root / "data" / "cfr"
    
    # Ensure cfr_dir exists
    cfr_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all mp4 videos in raw_dir
    video_files = glob.glob(str(raw_dir / "*.mp4"))
    
    if not video_files:
        print(f"No video files found in {raw_dir}")
        print("Please make sure to name your files like 'tran01_cam1.mp4' and put them in the raw folder.")
        return

    print(f"Found {len(video_files)} videos. Starting FPS standardization to 60 FPS...")
    
    for input_file in video_files:
        filename = os.path.basename(input_file)
        output_file = str(cfr_dir / filename)
        
        print(f"\nProcessing: {filename}")
        success = standardize_video_fps(input_file, output_file, target_fps=60)
        
        if success:
            print(f"Successfully processed {filename}")
        else:
            print(f"Failed to process {filename}")
            
    print("\nAll tasks completed.")

if __name__ == "__main__":
    process_all_videos()
