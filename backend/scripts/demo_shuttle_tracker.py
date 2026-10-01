"""
Test raw shuttle tracking with side-by-side video and mask display.

Usage:
    python scripts/demo_shuttle_tracker.py [video_path]

Controls:
    q - Quit
"""
import sys
import cv2

# Shared utilities (also sets up sys.path)
from _common import create_detector, setup_logging, setup_gui, FPSCounter, LiveTuner
from core.config import DEFAULT_VIDEO


def run_shuttle(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    detector = create_detector(backend="cv")
    fps_counter = FPSCounter()

    tuner = LiveTuner("Tracker Tuner")
    tuner.bind("Min Gate (px)", detector, "min_gate_px", 5.0, 200.0, 5.0)
    tuner.bind("Max Track Len", detector, "max_track_len", 30, 400, 10)
    tuner.bind("Max Speed Jump (x)", detector, "max_speed_ratio", 1.0, 10.0, 0.5)
    tuner.bind("Min Cos Angle", detector, "min_cos_angle", -1.0, 1.0, 0.1)

    print("Starting shuttlecock tracking playback... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pt, fg_mask = detector.detect(frame)
        annotated = detector.draw_trajectory(frame)

        current_fps = fps_counter.tick()

        # Convert mask to BGR for side-by-side display
        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        mask_label = "Background Subtractor Mask" if detector.backend == "cv" else "TrackNet Heatmap"
        cv2.putText(mask_bgr, mask_label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        annotated_resized = cv2.resize(annotated, (720, 405))
        mask_resized = cv2.resize(mask_bgr, (720, 405))
        combined = cv2.hconcat([annotated_resized, mask_resized])

        cv2.imshow("Left: Final Tracking | Right: CV2 Motion Mask", combined)
        tuner.render()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    setup_gui()
    target_video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    run_shuttle(target_video)
