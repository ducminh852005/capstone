import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def standardize_video_fps(input_path: str, output_path: str, target_fps: int = 60) -> bool:
    """
    Standardize video to constant frame rate (CFR) using FFmpeg.
    Following the guideline: ffmpeg -i raw/... -vf fps=60 -c:v libx264 -crf 18 -preset medium -c:a aac cfr/...
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return False

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-y", # Overwrite output if exists
        "-i", input_path,
        "-vf", f"fps={target_fps}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        output_path
    ]

    try:
        logger.info(f"Running FFmpeg to standardize FPS for {input_path} to {target_fps} fps")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed with error:\n{result.stderr}")
            return False
        
        logger.info(f"Successfully processed video: {output_path}")
        return True
    except FileNotFoundError:
        logger.error("FFmpeg is not installed or not in the system PATH.")
        return False
    except Exception as e:
        logger.error(f"An error occurred while processing video: {str(e)}")
        return False

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    # standardize_video_fps("../../data/raw/tran01_cam1.mp4", "../../data/cfr/tran01_cam1.mp4")
