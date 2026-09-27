import os
import sys
import cv2
import json
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.player_tracker import PlayerTracker
from core.court_calibration import CourtCalibrator

def create_tactical_board():
    # 6.1m width -> 610 px, 6.7m length -> 670 px
    board = np.zeros((670, 610, 3), dtype=np.uint8)
    board[:] = (40, 150, 40) # Green background
    
    cv2.rectangle(board, (0, 0), (610, 670), (255, 255, 255), 4) # Outer boundary
    cv2.line(board, (46, 0), (46, 670), (255, 255, 255), 2)      # Singles sideline
    cv2.line(board, (564, 0), (564, 670), (255, 255, 255), 2)    # Singles sideline
    cv2.line(board, (0, 198), (610, 198), (255, 255, 255), 2)    # Service line
    cv2.line(board, (305, 198), (305, 670), (255, 255, 255), 2)  # Center line
    return board

def test_tracking(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Load calibration points if they exist
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calibration.json")
    H = None
    calibrator = CourtCalibrator()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
            image_points = data.get("image_points")
            world_points = [
                (0, 0),         # Bottom-Left
                (0, 6.10),      # Bottom-Right
                (6.70, 0),      # Net-Left
                (6.70, 6.10)    # Net-Right
            ]
            H = calibrator.get_rough_homography(None, image_points, world_points)
            H_inv = np.linalg.inv(H)
    else:
        print("WARNING: calibration.json not found. Court lines will not be drawn.")

    # Initialize tracker (It will automatically download yolov8n.pt if not exists)
    tracker = PlayerTracker(model_path="yolov8n.pt", conf_thresh=0.5)

    print("Starting video playback... Press 'q' to stop.")
    
    rough_court_bounds_y = (300, 1080)
    
    frame_count = 0
    # Store the last valid tracking result so we can draw it on skipped frames
    last_annotated_frame = None
    
    tactical_board = create_tactical_board()
    heatmap_layer = np.zeros_like(tactical_board)
    
    # Variables for FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        frame_count += 1
        
        # Frame Skipping: Process every 2nd frame to x2 the speed
        if frame_count % 2 == 0 and last_annotated_frame is not None:
            # We must re-render the combined view so the skip frame displays correctly
            display_video = cv2.resize(last_annotated_frame, (1024, 576))
            display_board = cv2.addWeighted(tactical_board, 1.0, heatmap_layer, 0.5, 0)
            display_board = cv2.resize(display_board, (400, 576))
            combined = cv2.hconcat([display_video, display_board])
            
            cv2.imshow("YOLOv8 + ByteTrack: Player Tracking", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 1. Draw Court Lines (If calibrated)
        if H is not None:
            frame = calibrator.draw_court_frame(frame, H)

        # --- DYNAMIC ROI CROPPING FOR YOLO ---
        # We cut 15% from left and right, and 10% from the top.
        # This gives players a comfortable padding to move around,
        # but eliminates the side audiences and ceiling to speed up YOLO.
        h, w = frame.shape[:2]
        x_min = int(w * 0.15)
        x_max = int(w * 0.85)
        y_min = int(h * 0.10)
        y_max = h
        
        roi_frame = frame[y_min:y_max, x_min:x_max]

        # 2. Feed cropped frame to YOLO + ByteTrack
        results = tracker.track_frame(roi_frame, persist=True)
        
        # 3. Extract bounding boxes and track IDs
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            
            # Map boxes back to the original frame coordinates
            boxes[:, 0] += x_min # x1
            boxes[:, 2] += x_min # x2
            boxes[:, 1] += y_min # y1
            boxes[:, 3] += y_min # y2
            
            ids = results.boxes.id.cpu().numpy()
            
            # 3. Filter out audience/referees outside the court bounds
            valid_players = tracker.filter_players_by_roi(frame, boxes, ids, rough_court_bounds_y, H)
            
            # --- TACTICAL BOARD HEATMAP ---
            if H_inv is not None:
                for track_id, _ in valid_players.items():
                    if track_id in tracker.player_foot_points:
                        foot_x, foot_y = tracker.player_foot_points[track_id]
                        
                        # Project Image coordinate to 3D World Coordinate
                        pt_img = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                        pt_world = cv2.perspectiveTransform(pt_img, H_inv)[0][0]
                        world_x, world_y = pt_world[0], pt_world[1]
                        
                        # Map to 2D Minimap
                        px = int(world_y * 100)
                        py = 670 - int(world_x * 100)
                        
                        if 0 <= px < 610 and 0 <= py < 670:
                            # Draw a translucent red circle to build up a heatmap
                            cv2.circle(heatmap_layer, (px, py), 5, (0, 0, 255), -1)
            
            # 5. Draw the results
            annotated_frame = tracker.draw_tracking(frame, valid_players)
        else:
            annotated_frame = frame
            
        # --- FPS Calculation ---
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed > 1.0: # Update FPS every 1 second
            current_fps = fps_frame_count / elapsed
            fps_start_time = time.time()
            fps_frame_count = 0
            
        # Draw FPS on frame
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
        last_annotated_frame = annotated_frame

        # Resize video for viewing
        display_video = cv2.resize(annotated_frame, (1024, 576))
        
        # Render tactical board
        display_board = cv2.addWeighted(tactical_board, 1.0, heatmap_layer, 0.5, 0)
        display_board = cv2.resize(display_board, (400, 576)) # Match video height
        
        cv2.putText(display_board, "2D Movement Heatmap", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # Stack side by side
        combined = cv2.hconcat([display_video, display_board])
        
        cv2.imshow("YOLOv8 + ByteTrack: Player Tracking", combined)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with the smallest CFR video
    default_video = r"..\data\cfr\tran04_cam1.mp4"
    target_video = sys.argv[1] if len(sys.argv) > 1 else default_video
    test_tracking(target_video)
