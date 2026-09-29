"""
Print the shuttle tracker state frame by frame for a window of a video, plus every hit
measured by the smash detector. Use it to understand why a track broke or restarted.

Usage:
    python scripts/debug_track_state.py [video] --start 380 --end 410 [--backend tracknet]
"""
import argparse

from _common import create_detector, load_video_and_calibration, setup_logging
from core import config
from core.smash import SmashDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=config.DEFAULT_VIDEO)
    ap.add_argument("--start", type=int, default=0, help="first frame to print")
    ap.add_argument("--end", type=int, default=500, help="last frame to process")
    ap.add_argument("--backend", default=config.DEFAULT_BACKEND)
    args = ap.parse_args()

    reader, _, _, roi = load_video_and_calibration(args.video)
    detector = create_detector(args.backend)
    smash = SmashDetector()

    for frame_idx, frame in reader:
        if frame_idx > args.end:
            break
        pt, _ = detector.detect(frame, roi=roi)
        hit = smash.update(frame_idx, pt, detector.track_active, detector.track_len, detector.kf.x[:2])
        if hit is not None:
            print(f"Hit at frame {hit.frame_idx}: speed={hit.speed:.1f} angle={hit.angle_deg:.1f} "
                  f"y={hit.pt[1]} smash={hit.is_smash}")
        if frame_idx >= args.start:
            n_cands = len(detector._recent_candidates[-1]) if detector._recent_candidates else 0
            print(f"Frame {frame_idx:4}: pt={pt} active={detector.track_active} "
                  f"len={detector.track_len} misses={detector.misses} cands={n_cands}")
    reader.release()


if __name__ == "__main__":
    setup_logging()
    main()
