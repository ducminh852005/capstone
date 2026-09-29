"""
Test script for smash (đập cầu) detection in badminton videos.

Detects smashes by measuring the shuttle's speed right after each new hit.
A "hit" is recognized when the Kalman tracker starts a new track (the old
trajectory broke because the shuttle changed direction sharply on contact
with the racket). Speed is computed from the displacement between the first
two real detections in the new track.

Usage:
    python test_smash.py [video_path] [backend]

Controls:
    q       - Quit
    Space   - Pause / Resume
"""
import os
import sys
import cv2
import numpy as np

# Shared utilities (also sets up sys.path)
from _common import (
    create_detector, load_video_and_calibration,
    FPSCounter, draw_shuttle_trajectory, draw_court_overlay, Minimap,
)
from core.config import (
    SMASH_SPEED_THRESHOLD, SMASH_DISPLAY_DURATION,
    DISPLAY_SIZE, DEBUG_LOG_INTERVAL, DEFAULT_VIDEO, DEFAULT_BACKEND,
)


def test_smash_detection(video_path, backend=DEFAULT_BACKEND):
    print(f"Opening video: {video_path}")
    reader, H, H_inv, roi = load_video_and_calibration(video_path)
    detector = create_detector(backend)
    print(f"Shuttle backend: {detector.backend}")
    print("Starting processing... Press 'q' to quit.")

    smash_count = 0
    smash_events = []

    # State for two-point speed measurement:
    # When a new track starts we record the first detection point. Once a
    # second real detection arrives we compute speed = distance / dt.
    pending_hit = None
    was_active = False

    for frame_idx, frame in reader:
        pt, mask = detector.detect(frame, roi=roi)

        # --- Hit detection & speed measurement ---

        # A new track just started (old trajectory broke at a racket hit)
        if not was_active and detector.track_active:
            start_pt = pt if pt is not None else detector.kf.x[:2]
            pending_hit = {"start_frame": frame_idx, "start_pt": start_pt}

        # Waiting for the second real detection to compute speed
        if detector.track_active and pending_hit is not None:
            if pt is not None and frame_idx > pending_hit["start_frame"]:
                dt = frame_idx - pending_hit["start_frame"]
                dx = pt[0] - pending_hit["start_pt"][0]
                dy = pt[1] - pending_hit["start_pt"][1]

                speed = np.hypot(dx, dy) / dt  # px/frame
                vx, vy = dx / dt, dy / dt
                angle_deg = np.degrees(np.arctan2(vy, vx))

                print(
                    f"[DEBUG] Frame {frame_idx}: Hit Confirmed! "
                    f"Speed = {speed:.1f} px/frame | Angle = {angle_deg:.1f}°"
                )

                if speed > SMASH_SPEED_THRESHOLD:
                    smash_count += 1
                    smash_events.append({
                        "frame": pending_hit["start_frame"],
                        "pt": (
                            int(pending_hit["start_pt"][0]),
                            int(pending_hit["start_pt"][1]),
                        ),
                        "speed": speed,
                        "angle": angle_deg,
                        "count": smash_count,
                    })
                    print(
                        f"\n🏸🔥 [SMASH #{smash_count} DETECTED] "
                        f"Frame {pending_hit['start_frame']}: "
                        f"Speed = {speed:.1f} px/frame"
                    )

                pending_hit = None  # done measuring

        # Track lost before we got a second point — discard
        if not detector.track_active:
            pending_hit = None

        was_active = detector.track_active

        # --- Debug heartbeat ---
        if frame_idx % DEBUG_LOG_INTERVAL == 0:
            print(f"[DEBUG] Processing Frame {frame_idx}...")

        # --- Drawing ---
        draw_shuttle_trajectory(frame, detector.trajectory)
        draw_court_overlay(frame, H)
        _draw_smash_overlays(frame, frame_idx, smash_events, smash_count)

        # Display
        frame_resized = cv2.resize(frame, DISPLAY_SIZE)
        cv2.imshow("Test Smash Detection", frame_resized)

        # Keyboard controls
        key = cv2.waitKey(0 if getattr(detector, "paused", False) else 1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            detector.paused = not getattr(detector, "paused", False)
            state = "PAUSED" if detector.paused else "RESUMED"
            print(f"[{state}] at Frame {frame_idx}")

    reader.release()
    cv2.destroyAllWindows()

    # Final summary
    print(f"\n{'='*50}")
    print(f"✅ Test finished. Total smashes detected: {smash_count}")
    if smash_events:
        speeds = [s["speed"] for s in smash_events]
        print(f"   Average smash speed: {np.mean(speeds):.1f} px/frame")
        print(f"   Max smash speed:     {np.max(speeds):.1f} px/frame")
    print(f"{'='*50}")


def _draw_smash_overlays(frame, frame_idx, smash_events, smash_count):
    """Draw 'SMASH!' labels near each detected smash and the running total."""
    for smash in smash_events:
        age = frame_idx - smash["frame"]
        if age < SMASH_DISPLAY_DURATION:
            label = f"SMASH #{smash['count']}! ({smash['speed']:.1f})"
            pos = (smash["pt"][0] - 60, smash["pt"][1] - 30)
            cv2.putText(frame, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

    cv2.putText(frame, f"Total Smashes: {smash_count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)


if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    backend = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BACKEND
    test_smash_detection(target_video, backend)
