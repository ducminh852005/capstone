"""
Demo for smash (dap cau) detection in badminton videos.

Detects smashes by measuring the shuttle's speed right after each new hit (see
core/smash.py): a hit is recognised when the Kalman tracker starts a new track, and its
speed is measured from the first two real detections of that track.

Usage:
    python scripts/demo_smash.py [video_path] [backend]

Controls:
    q       - Quit
    Space   - Pause / Resume
"""
import sys

import cv2
import numpy as np

# Shared utilities (also sets up sys.path)
from _common import (
    create_detector, load_video_and_calibration, setup_logging,
    draw_shuttle_trajectory, draw_court_overlay, LiveTuner,
)
from core import config
from core.smash import SmashDetector


def run_smash_detection(video_path, backend=config.DEFAULT_BACKEND):
    print(f"Opening video: {video_path}")
    reader, H, H_inv, roi = load_video_and_calibration(video_path)
    detector = create_detector(backend)
    print(f"Shuttle backend: {detector.backend}")

    smash = SmashDetector()   # thresholds read from config at call time, so the tuner works

    tuner = LiveTuner("Smash Tuner")
    tuner.bind("Smash Speed", config, "SMASH_SPEED_THRESHOLD", 5.0, 40.0, 1.0)
    tuner.bind("Smash Min Y", config, "SMASH_MIN_Y", 0.0, 500.0, 10.0)
    tuner.bind("Smash Min Angle", config, "SMASH_MIN_ANGLE", 0.0, 90.0, 5.0)
    tuner.bind("Smash Max Angle", config, "SMASH_MAX_ANGLE", 90.0, 180.0, 5.0)
    tuner.bind("Max Speed Jump (x)", detector, "max_speed_ratio", 1.0, 10.0, 0.5)
    tuner.bind("Min Cos Angle", detector, "min_cos_angle", -1.0, 1.0, 0.1)

    print("Starting processing... Press 'q' to quit.")
    paused = False

    for frame_idx, frame in reader:
        pt, _ = detector.detect(frame, roi=roi)

        hit = smash.update(frame_idx, pt, detector.track_active, detector.track_len, detector.kf.x[:2])
        if hit is not None:
            print(f"[DEBUG] Frame {hit.frame_idx}: Track restarted "
                  f"(Speed={hit.speed:.1f}, Angle={hit.angle_deg:.1f} deg, Y={hit.pt[1]})")
            if hit.is_smash:
                print(f"\n[SMASH #{hit.smash_number} DETECTED] Frame {hit.frame_idx}: "
                      f"Speed = {hit.speed:.1f} px/frame")

        if frame_idx % config.DEBUG_LOG_INTERVAL == 0:
            print(f"[DEBUG] Processing Frame {frame_idx}...")

        draw_shuttle_trajectory(frame, detector.trajectory)
        draw_court_overlay(frame, H)
        _draw_smash_overlays(frame, frame_idx, smash.events)

        cv2.imshow("Smash Detection", cv2.resize(frame, config.DISPLAY_SIZE))
        tuner.render()

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print(f"[{'PAUSED' if paused else 'RESUMED'}] at Frame {frame_idx}")

    reader.release()
    cv2.destroyAllWindows()

    print(f"\n{'=' * 50}")
    print(f"Finished. Total smashes detected: {smash.count}")
    if smash.events:
        speeds = [s.speed for s in smash.events]
        print(f"   Average smash speed: {np.mean(speeds):.1f} px/frame")
        print(f"   Max smash speed:     {np.max(speeds):.1f} px/frame")
    print("=" * 50)


def _draw_smash_overlays(frame, frame_idx, smash_events):
    """Draw 'SMASH!' labels near each recent smash and the running total."""
    for s in smash_events:
        if frame_idx - s.frame_idx < config.SMASH_DISPLAY_DURATION:
            label = f"SMASH #{s.smash_number}! ({s.speed:.1f})"
            cv2.putText(frame, label, (s.pt[0] - 60, s.pt[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    cv2.putText(frame, f"Total Smashes: {len(smash_events)}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)


if __name__ == "__main__":
    setup_logging()
    target_video = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_VIDEO
    backend = sys.argv[2] if len(sys.argv) > 2 else config.DEFAULT_BACKEND
    run_smash_detection(target_video, backend)
