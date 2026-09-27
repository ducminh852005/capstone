import cv2
import logging
from ultralytics import YOLO
from .pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)

class PlayerTracker:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.4, min_frames_in_court=60):
        """
        Initializes the YOLOv8 model for player tracking.
        min_frames_in_court: the object must stay in the court for this many frames to get an ID.
        """
        logger.info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.tracker_config = "bytetrack.yaml"
        
        # State management for IDs
        self.min_frames = min_frames_in_court
        self.player_presence_counter = {} # {track_id: frames_inside}
        self.confirmed_players = set()    # IDs that have passed the threshold
        
        # Initialize MediaPipe Pose Estimator
        self.pose_estimator = PoseEstimator(model_complexity=1) # Complexity 1 is faster than 2 for real-time
        self.player_foot_points = {} # Cache for drawing: {track_id: (x, y)}
        
        # Performance Optimization: Cache offset to avoid running MediaPipe every frame
        self.frame_counter = 0
        self.foot_offsets = {} # {track_id: (offset_x, offset_y)}

    def track_frame(self, frame, persist=True):
        """
        Processes a single frame and returns tracking results.
        persist=True tells the tracker to remember IDs from the previous frames.
        """
        # YOLOv8 class 0 is "person". We filter out everything else.
        results = self.model.track(
            frame, 
            persist=persist, 
            classes=[0],           # Only detect persons
            conf=self.conf_thresh,
            tracker=self.tracker_config,
            verbose=False
        )
        return results[0]

    def filter_players_by_roi(self, frame, boxes, ids, rough_court_bounds_y, H=None):
        """
        Filters out people outside the main court area.
        If H is provided, it strictly checks if the foot is inside the 3D projected court polygon.
        Otherwise, falls back to the rough Y bounds.
        """
        valid_players = {}
        self.player_foot_points.clear()
        
        if boxes is None or ids is None:
            return valid_players
            
        court_polygon = None
        if H is not None:
            # Near Half-Court world boundaries: X(0 to 6.7), Y(0 to 6.1)
            # This completely ignores players on the far side of the net.
            import numpy as np
            pts_world = np.array([[0, 0], [6.7, 0], [6.7, 6.1], [0, 6.1]], dtype=np.float32).reshape(-1, 1, 2)
            court_polygon = cv2.perspectiveTransform(pts_world, H)
            if court_polygon is not None:
                court_polygon = np.int32(court_polygon)

        self.frame_counter += 1
        
        for box, track_id in zip(boxes, ids):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2
            
            # -------------------------------------------------------------
            # PERFORMANCE OPTIMIZATION: LAZY MEDIAPIPE (Offset Caching)
            # -------------------------------------------------------------
            foot_pt = None
            
            # Only run MediaPipe once every 10 frames for this ID (or if we haven't seen them)
            if track_id not in self.foot_offsets or self.frame_counter % 10 == 0:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    foot_pt = self.pose_estimator.extract_foot_point(person_crop, bbox_offset=(x1, y1))
                    if foot_pt:
                        # Calculate and cache the offset from the bottom-center of the bbox
                        offset_x = foot_pt[0] - x_center
                        offset_y = foot_pt[1] - y2
                        self.foot_offsets[track_id] = (offset_x, offset_y)
            
            # If MediaPipe didn't run this frame (or failed), use the cached offset
            if not foot_pt and track_id in self.foot_offsets:
                offset_x, offset_y = self.foot_offsets[track_id]
                foot_pt = (x_center + offset_x, y2 + offset_y)
                
            if foot_pt:
                foot_x, foot_y = foot_pt
                self.player_foot_points[track_id] = foot_pt
            else:
                # Absolute fallback if everything fails
                foot_x, foot_y = x_center, y2
                self.player_foot_points[track_id] = (foot_x, foot_y)
            # -------------------------------------------------------------
            
            # Strict Court Checking using Polygon if H is available
            if court_polygon is not None:
                # measureDist=False returns +1 (inside), 0 (edge), -1 (outside)
                is_inside = cv2.pointPolygonTest(court_polygon, (foot_x, foot_y), False) >= 0
            else:
                is_inside = (rough_court_bounds_y[0] <= foot_y <= rough_court_bounds_y[1])
            
            if is_inside:
                self.player_presence_counter[track_id] = self.player_presence_counter.get(track_id, 0) + 1
                if self.player_presence_counter[track_id] >= self.min_frames:
                    self.confirmed_players.add(track_id)
            else:
                if track_id not in self.confirmed_players:
                    self.player_presence_counter[track_id] = 0
            
            # Draw logic: 
            # If they are confirmed, we still verify they aren't completely off-screen.
            # Using a slightly expanded bounding box tolerance for jumping.
            if track_id in self.confirmed_players:
                if court_polygon is not None:
                    # Tolerance: Distance to polygon (True returns distance)
                    dist = cv2.pointPolygonTest(court_polygon, (foot_x, foot_y), True)
                    if dist >= -100: # allow them to step 100 pixels out of bounds
                        valid_players[track_id] = [x1, y1, x2, y2]
                else:
                    expanded_bounds_y = (rough_court_bounds_y[0] - 100, rough_court_bounds_y[1] + 100)
                    if expanded_bounds_y[0] <= foot_y <= expanded_bounds_y[1]:
                        valid_players[track_id] = [x1, y1, x2, y2]
                
        return valid_players

    def draw_tracking(self, frame, valid_players):
        """
        Draws bounding boxes, IDs, and precise foot points on the frame for valid players.
        """
        annotated_frame = frame.copy()
        
        for track_id, box in valid_players.items():
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID background
            label = f"ID: {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw ID text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw exact foot point from MediaPipe
            if track_id in self.player_foot_points:
                foot_pt = self.player_foot_points[track_id]
                cv2.circle(annotated_frame, foot_pt, 6, (0, 0, 255), -1) # Red dot for foot
            
        return annotated_frame
