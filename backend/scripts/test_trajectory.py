"""
Test shuttle trajectory tracking with umpire line calls and a 2D minimap.

Usage:
    python test_trajectory.py [video_path] [backend]

Controls:
    q       - Quit
    Space   - Pause / Resume
    s       - Report a missed bounce (logged to missed_bounces.txt)
"""
import os
import sys
import cv2

# Shared utilities (also sets up sys.path)
from _common import (
    create_detector, load_video_and_calibration,
    draw_shuttle_trajectory, draw_court_overlay, draw_bounce_markers, Minimap,
)
from core import court_model
from core.config import DISPLAY_SIZE, DEFAULT_VIDEO, DEFAULT_BACKEND
from core.umpire import RallyUmpire


def test_trajectory(video_path, backend=DEFAULT_BACKEND):
    print(f"Opening video: {video_path}")
    reader, H, H_inv, roi = load_video_and_calibration(video_path)
    detector = create_detector(backend)
    print(f"Shuttle backend: {detector.backend}")
    print("Starting processing... Press 'q' to quit.")

    # Initialize Umpire for landing detection
    umpire = None
    if H_inv is not None:
        w, h = reader.size
        stride = getattr(detector._source, 'batch_stride', 15) if hasattr(detector, '_source') else 15
        rest_frm = 3 if stride < 15 else 2

        umpire = RallyUmpire(H_inv=H_inv, match_type="singles",
                             frame_size=(h, w), roi=roi,
                             net_top_y=court_model.net_top_threshold_y(H),
                             rest_frames=rest_frm)
        print(f"Umpire initialized. Adjusted for stride={stride} (frames={rest_frm})")
    else:
        print("WARNING: No calibration data, Umpire disabled.")

    bounce_events = []
    minimap = Minimap()

    for frame_idx, frame in reader:
        pt, mask = detector.detect(frame, roi=roi if umpire is not None else None)

        # Umpire tracks landings
        if umpire is not None:
            call = umpire.update(frame_idx, pt, detector.track_active)
            if call is not None:
                bounce_events.append(call)
                print(f"\n💥 [CẦU RƠI] Frame {call.frame_idx}: "
                      f"Tọa độ ảnh {call.image_pt}, Tọa độ sân (m) {call.world_pt}")
                print(f"👉 Kết quả: {call.result} (Half: {call.half}, Margin: {call.margin_m:.3f}m)\n")

        # Terminal log
        if pt is not None:
            print(f"Frame {frame_idx}: Shuttle detected at {pt}")
        elif frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: Processing... (no shuttle detected)")

        # --- Drawing ---
        draw_shuttle_trajectory(frame, detector.trajectory)
        draw_bounce_markers(frame, bounce_events)
        draw_court_overlay(frame, H)

        frame_resized = cv2.resize(frame, DISPLAY_SIZE)
        minimap_img = minimap.render(bounce_events)
        display_img = cv2.hconcat([frame_resized, minimap_img])

        cv2.imshow("Test Trajectory", display_img)

        # Keyboard controls
        key = cv2.waitKey(0 if getattr(detector, 'paused', False) else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            detector.paused = not getattr(detector, 'paused', False)
            state = "PAUSED" if detector.paused else "RESUMED"
            print(f"[{state}] at Frame {frame_idx}")
        elif key == ord('s'):
            print(f"🚨 [MISS REPORT] Missed bounce around Frame {frame_idx}")
            with open("missed_bounces.txt", "a", encoding="utf-8") as f:
                f.write(f"Missed bounce around frame {frame_idx}\n")

    reader.release()
    cv2.destroyAllWindows()
    print("Test finished.")


if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    backend = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BACKEND
    test_trajectory(target_video, backend)
