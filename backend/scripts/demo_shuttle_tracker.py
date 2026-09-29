"""
Test raw shuttle tracking with side-by-side video and mask display.

Usage:
    python test_shuttle_tracker.py [video_path]

Controls:
    q - Quit
"""
import sys
import cv2

# Shared utilities (also sets up sys.path)
from _common import create_detector, FPSCounter
from core.config import DEFAULT_VIDEO


def test_shuttle(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    detector = create_detector(backend="cv")
    fps_counter = FPSCounter()

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    test_shuttle(target_video)
