import glob
import os

import _common
from core import config
from core.video_processor import standardize_video_fps

def process_all_videos():
    raw_dir = config.DATA_DIR / "raw"
    cfr_dir = config.DATA_DIR / "cfr"

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
    _common.setup_logging()
    process_all_videos()
