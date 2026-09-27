import os
import sys
import cv2
import json
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.shuttle_tracker import ShuttleDetector
from core.court_calibration import CourtCalibrator

def get_last_valid_pt(trajectory):
    for i in range(len(trajectory)-1, -1, -1):
        if trajectory[i] is not None:
            return trajectory[i]
    return None

def test_auto_umpire(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    # 1. Load Calibration
    calibrator = CourtCalibrator()
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calibration.json")
    H = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
            image_points = data.get("image_points", [])
            # Map for FULL COURT: X(0 to 13.4m), Y(0 to 6.1m)
            # The calibration points the user clicked were: 
            # Bottom-Left, Bottom-Right, Net-Left, Net-Right
            # Which corresponds to the Near Half-Court.
            world_points = [
                (0, 0),         # Bottom-Left
                (0, 6.10),      # Bottom-Right
                (6.70, 0),      # Net-Left
                (6.70, 6.10)    # Net-Right
            ]
            H = calibrator.get_rough_homography(None, image_points, world_points)
            H_inv = np.linalg.inv(H)
    else:
        print("ERROR: Please run test_calibration.py first!")
        return

    detector = ShuttleDetector()
    
    # Umpire States
    score_near = 0
    score_far = 0
    shuttle_lost_frames = 0
    rally_ongoing = False
    
    # UI Overlay States
    show_alert_until = 0
    alert_text = ""
    alert_color = (255, 255, 255)
    last_landing_pt = None
    
    # We will draw the full court for visualization
    full_court_world = np.array([[0, 0], [13.4, 0], [13.4, 6.1], [0, 6.1]], dtype=np.float32).reshape(-1, 1, 2)
    full_court_img = cv2.perspectiveTransform(full_court_world, H)
    full_court_img = np.int32(full_court_img)
    
    # Net line
    net_world = np.array([[6.7, 0], [6.7, 6.1]], dtype=np.float32).reshape(-1, 1, 2)
    net_img = cv2.perspectiveTransform(net_world, H)
    net_img = np.int32(net_img)
    
    # Calculate a safe threshold for the Net's Top (Y is inverted in images)
    # The shuttle must go higher (lower Y value) than this to be considered a valid shot
    net_base_y = (net_img[0][0][1] + net_img[1][0][1]) / 2
    net_top_threshold_y = net_base_y - 50 
    
    # FPS Variables
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    min_y_in_rally = float('inf')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- DYNAMIC ROI CROPPING ---
        # Ignore audiences walking on the sides or bottom/top edges
        h, w = frame.shape[:2]
        x_min = int(w * 0.15)
        x_max = int(w * 0.85)
        y_min = int(h * 0.10)
        y_max = h # Keep bottom in case it lands very low
        
        roi_frame = frame[y_min:y_max, x_min:x_max]
        
        # 1. Detect Shuttle inside ROI
        pt, fg_mask_roi = detector.detect(roi_frame)
        
        # Pad the mask back to the original size so display logic doesn't break
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        fg_mask[y_min:y_max, x_min:x_max] = fg_mask_roi
        
        if pt is not None:
            # Map point back to original frame coordinate
            global_pt = (pt[0] + x_min, pt[1] + y_min)
            # Update the trajectory with the global point for drawing!
            # Since ShuttleDetector appended local 'pt', we need to overwrite it!
            detector.trajectory[-1] = global_pt
            
            shuttle_lost_frames = 0
            if not rally_ongoing:
                rally_ongoing = True
                min_y_in_rally = global_pt[1] # Reset for new rally
            else:
                if global_pt[1] < min_y_in_rally:
                    min_y_in_rally = global_pt[1] # Track highest point
        else:
            if rally_ongoing:
                shuttle_lost_frames += 1
                
        # 2. Check for Landing Event (Lost for 15 frames)
        if rally_ongoing and shuttle_lost_frames > 15:
            rally_ongoing = False
            
            # --- HEIGHT FILTER (ANTI-PICKUP SENSOR) ---
            # If the shuttle never flew higher than the net top, it was likely just
            # being picked up or rolled on the floor. Ignore it!
            if min_y_in_rally > net_top_threshold_y:
                continue
                
            last_pt = get_last_valid_pt(detector.trajectory)
            
            if last_pt is not None:
                last_landing_pt = last_pt
                
                # Transform to world
                pt_img = np.array([[[last_pt[0], last_pt[1]]]], dtype=np.float32)
                pt_world = cv2.perspectiveTransform(pt_img, H_inv)[0][0]
                world_x, world_y = pt_world[0], pt_world[1]
                
                # Umpire Logic (Focused on NEAR COURT only)
                is_near = (world_x < 6.7)
                
                if is_near:
                    # Check if it's within the bounds of the Near Court
                    is_in = (0 <= world_x <= 6.7) and (0 <= world_y <= 6.1)
                    if is_in:
                        score_far += 1
                        alert_text = "IN! (Near Court) - Point for FAR SIDE"
                        alert_color = (0, 255, 0)
                    else:
                        score_near += 1
                        alert_text = "OUT! (Near Court) - Point for NEAR SIDE"
                        alert_color = (0, 0, 255)
                        
                    # Show alert for 60 frames (~2 seconds)
                    show_alert_until = 60 

        # 3. Draw visuals
        annotated = detector.draw_trajectory(frame)
        
        # --- FPS Calculation ---
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed > 1.0:
            current_fps = fps_frame_count / elapsed
            fps_start_time = time.time()
            fps_frame_count = 0
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Draw full court bounds (White)
        cv2.polylines(annotated, [full_court_img], isClosed=True, color=(255, 255, 255), thickness=2)
        # Draw Net (Red)
        cv2.line(annotated, tuple(net_img[0][0]), tuple(net_img[1][0]), (0, 0, 255), 2)
        
        # Draw Scoreboard
        scoreboard_text = f"NEAR: {score_near}  |  FAR: {score_far}"
        cv2.rectangle(annotated, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(annotated, scoreboard_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Draw Landing Alert
        if show_alert_until > 0:
            show_alert_until -= 1
            cv2.putText(annotated, alert_text, (400, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, alert_color, 4)
            if last_landing_pt:
                # Flash a giant circle at the landing spot
                radius = 10 + (show_alert_until % 10)
                cv2.circle(annotated, last_landing_pt, radius, alert_color, -1)

        # Resize and prepare Mask display
        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_bgr, "Background Subtractor Mask", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
        # Resize both to fit the screen side-by-side (Shrunk to fit smaller screens)
        display_frame = cv2.resize(annotated, (640, 360))
        display_mask = cv2.resize(mask_bgr, (640, 360))
        
        combined = cv2.hconcat([display_frame, display_mask])
        
        cv2.imshow("Left: Auto Umpire (Hawk-Eye) | Right: Mask", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else r"..\data\cfr\tran04_cam1.mp4"
    test_auto_umpire(target_video)
