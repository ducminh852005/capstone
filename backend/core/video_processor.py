import os
import subprocess
import logging
from typing import List

from . import config

logger = logging.getLogger(__name__)

STDERR_TAIL_LINES = 10
"""Last lines of ffmpeg's stderr quoted in the error of a failed clip cut."""


def standardize_video_fps(input_path: str, output_path: str, target_fps: float = config.CFR_FPS) -> bool:
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
        "-vf", f"fps={target_fps:g}",
        "-c:v", "libx264",
        "-crf", str(config.STANDARDIZE_CRF),
        "-preset", config.STANDARDIZE_PRESET,
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


def build_cut_command(src: str, dst: str, start_frame: int, n_frames: int, fps: float,
                      crf: int = config.CLIP_CUT_CRF, preset: str = config.CLIP_CUT_PRESET,
                      gop_frames: int = config.CLIP_CUT_GOP_FRAMES) -> List[str]:
    """
    ffmpeg command that re-encodes `n_frames` frames of `src` starting at source frame `start_frame`.

    Seeking before -i with a re-encode is frame-accurate. The seek point sits half a frame before
    the wanted frame so timestamp rounding can neither drop it nor keep the previous one. The clip
    has no audio, a short GOP and no B-frames (cheap, exact frame stepping in the browser), and an
    explicit constant frame rate so clip frame k is source frame start_frame + k.
    """
    if start_frame < 0 or n_frames <= 0 or fps <= 0:
        raise ValueError(f"Invalid cut: start_frame={start_frame}, n_frames={n_frames}, fps={fps}")
    seek_s = max(start_frame - 0.5, 0.0) / fps
    return [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-ss", f"{seek_s:.6f}",
        "-i", str(src),
        "-frames:v", str(n_frames),
        "-an",
        "-vf", "format=yuv420p",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-g", str(gop_frames), "-bf", "0",
        "-r", f"{fps:g}",
        "-movflags", "+faststart",
        str(dst),
    ]


def cut_clip(src: str, dst: str, start_frame: int, n_frames: int, fps: float) -> str:
    """
    Cut `n_frames` frames of `src` from `start_frame` into `dst` (see build_cut_command).
    Raises FileNotFoundError if `src` or the ffmpeg binary is missing, RuntimeError if ffmpeg fails.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Video not found: {src}")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    command = build_cut_command(src, dst, start_frame, n_frames, fps)
    logger.info("Cutting %d frames from frame %d of %s into %s", n_frames, start_frame, src, dst)
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise FileNotFoundError("ffmpeg is not installed or not on PATH (install it, e.g. "
                                "`winget install ffmpeg`, then restart the terminal)") from e
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{tail}")
    return str(dst)
