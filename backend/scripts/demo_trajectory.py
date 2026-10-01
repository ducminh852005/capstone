"""
Test shuttle trajectory tracking with umpire line calls and a 2D minimap.

Usage:
    python scripts/demo_trajectory.py [video_path] [backend]

Controls:
    q       - Quit
    Space   - Pause / Resume
    s       - Report a missed bounce (logged to data/missed_bounces.txt)
"""
import sys

import cv2
import numpy as np

# Shared utilities (also sets up sys.path)
from _common import (
    create_detector, load_video_and_calibration, setup_logging, setup_gui,
    draw_shuttle_trajectory, draw_court_overlay, court_overlay_segments, draw_bounce_markers, Minimap,
    LiveTuner,
)
from core import config, court_model, gap_fill
from core.config import DISPLAY_SIZE, DEFAULT_VIDEO, DEFAULT_BACKEND
from core.umpire import RallyUmpire, match_type_from_metadata

MISSED_BOUNCES_LOG = config.DATA_DIR / "missed_bounces.txt"


def run_trajectory(video_path, backend=DEFAULT_BACKEND):
    print(f"Opening video: {video_path}")
    reader, H, H_inv, roi = load_video_and_calibration(video_path)
    detector = create_detector(backend)
    print(f"Shuttle backend: {detector.backend}")
    print("Starting processing... Press 'q' to quit.")

    tuner = LiveTuner("Trajectory Tuner")
    tuner.bind("Tail Length", config, "TRAJECTORY_TAIL_LENGTH", 5, 200, 5)

    # Initialize Umpire for landing detection
    umpire = None
    if H_inv is not None:
        w, h = reader.size
        umpire = RallyUmpire(H_inv=H_inv, match_type=match_type_from_metadata(video_path),
                             frame_size=(h, w), roi=roi,
                             net_top_y=court_model.net_top_threshold_y(H))
        print("Umpire initialized.")
    else:
        print("WARNING: No calibration data, Umpire disabled.")

    bounce_events = []
    minimap = Minimap()
    court_segments = court_overlay_segments(H) if H is not None else None
    display = np.empty((DISPLAY_SIZE[1], DISPLAY_SIZE[0] + minimap.map_w, 3), np.uint8)   # reused every frame
    fill_constraint = gap_fill.court_constraint(H_inv=H_inv, roi=roi)
    filled = []
    paused = False

    for frame_idx, frame in reader:
        pt, mask = detector.detect(frame, roi=roi if umpire is not None else None)

        # Umpire tracks landings
        if umpire is not None:
            call = umpire.update(frame_idx - detector.frame_lag, pt, detector.track_active,
                                 flight_id=detector.flight_id)
            if call is not None:
                bounce_events.append(call)
                print(f"\n[LANDING] Frame {call.frame_idx}: "
                      f"image {call.image_pt}, court (m) {call.world_pt}")
                print(f"Result: {call.result} (Half: {call.half}, Margin: {call.margin_m:.3f}m, "
                      f"1 px = {call.uncertainty_m:.3f}m, method: {call.method})\n")

        # Terminal log
        if pt is not None:
            print(f"Frame {frame_idx}: Shuttle detected at {pt}")
        elif frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: Processing... (no shuttle detected)")

        # --- Drawing ---
        # Estimates for the frames the detector could not see. They only change when a new
        # detection arrives, so recompute then (a fit per gap costs ~1.5 ms every frame otherwise).
        if pt is not None:
            filled = detector.fill_gaps(allowed=fill_constraint, tail=config.TRAJECTORY_TAIL_LENGTH + 40)
        draw_shuttle_trajectory(frame, detector.trajectory, filled=filled)
        draw_bounce_markers(frame, bounce_events)
        draw_court_overlay(frame, H, court_segments)

        display[:, :DISPLAY_SIZE[0]] = cv2.resize(frame, DISPLAY_SIZE)
        display[:, DISPLAY_SIZE[0]:] = minimap.render(bounce_events)

        cv2.imshow("Test Trajectory", display)

        # Keyboard controls
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"[{'PAUSED' if paused else 'RESUMED'}] at Frame {frame_idx}")
        elif key == ord('s'):
            print(f"[MISS REPORT] Missed bounce around Frame {frame_idx}")
            with open(MISSED_BOUNCES_LOG, "a", encoding="utf-8") as f:
                f.write(f"Missed bounce around frame {frame_idx}\n")

    reader.release()
    cv2.destroyAllWindows()
    print("Test finished.")


if __name__ == "__main__":
    setup_logging()
    setup_gui()
    target_video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    backend = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BACKEND
    run_trajectory(target_video, backend)
