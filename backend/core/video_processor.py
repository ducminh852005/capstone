import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def standardize_video_fps(input_path: str, output_path: str, target_fps: int = 60) -> bool:
    """
    Standardize video to constant frame rate (CFR) using FFmpeg.
    Following the guideline: ffmpeg -i raw/... -vf fps=60 -c:v libx264 -crf 18 -preset medium -c:a aac cfr/...
    """
    if not os.path.exists(input_path):
        logger.error("Input file does not exist: %s", input_path)
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
        logger.info("Running FFmpeg to standardize FPS for %s to %s fps", input_path, target_fps)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg failed with error:\n%s", result.stderr)
            return False

        logger.info("Successfully processed video: %s", output_path)
        return True
    except FileNotFoundError:
        logger.error("FFmpeg is not installed or not in the system PATH.")
        return False
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("An error occurred while processing video: %s", e)
        return False

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    # standardize_video_fps("../../data/raw/tran01_cam1.mp4", "../../data/cfr/tran01_cam1.mp4")
