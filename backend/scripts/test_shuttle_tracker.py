import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.shuttle_tracker import ShuttleDetector

import time

def test_shuttle(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    detector = ShuttleDetector()
    
    print("Starting shuttlecock tracking playback... Press 'q' to stop.")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect shuttlecock
        pt, fg_mask = detector.detect(frame)
        
        # Draw the tail (trajectory)
        annotated = detector.draw_trajectory(frame)
        
        # --- FPS Calculation ---
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed > 1.0:
            current_fps = fps_frame_count / elapsed
            fps_start_time = time.time()
            fps_frame_count = 0
            
        # Convert mask to BGR so we can stack them side-by-side
        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        # Add labels and FPS
        cv2.putText(mask_bgr, "Background Subtractor Mask", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Resize for display
        annotated_resized = cv2.resize(annotated, (720, 405)) # 16:9 ratio
        mask_resized = cv2.resize(mask_bgr, (720, 405))
        
        # Stack horizontally
        combined = cv2.hconcat([annotated_resized, mask_resized])
        
        cv2.imshow("Left: Final Tracking | Right: CV2 Motion Mask", combined)
        
        # Wait key for speed control. 1 = max speed. 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    target_video = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cfr", "tran04_cam1.mp4")
    test_shuttle(target_video)
