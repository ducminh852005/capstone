import cv2
import numpy as np
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, model_complexity=2):
        """
        Initialize MediaPipe Pose using the new Tasks API.
        We expect 'pose_landmarker_heavy.task' to be downloaded in the backend directory.
        """
        # Determine model path
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, 'pose_landmarker_lite.task')
        
        if not os.path.exists(model_path):
            logger.error(f"MediaPipe Model not found at {model_path}. Please download it.")
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            
        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def extract_foot_point(self, person_crop, bbox_offset):
        """
        Extract the foot point of the player using MediaPipe Pose.
        person_crop: cropped image of the person
        bbox_offset: (x, y) offset of the crop in the original image
        """
        if person_crop is None or person_crop.size == 0:
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process image
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            return None
            
        h, w, _ = person_crop.shape
        # We only process the first detected person in the crop (should be only 1 anyway)
        landmarks = detection_result.pose_landmarks[0]
        
        def get_pt(idx):
            lm = landmarks[idx]
            # presence and visibility checks
            if lm.visibility > 0.5 and getattr(lm, 'presence', 1.0) > 0.5:
                return (int(lm.x * w), int(lm.y * h))
            return None

        # Landmarks: 29: left heel, 30: right heel, 31: left toe, 32: right toe
        l_heel = get_pt(29)
        r_heel = get_pt(30)
        l_toe = get_pt(31)
        r_toe = get_pt(32)
        
        candidates = []
        if l_heel and l_toe:
            candidates.append( ((l_heel[0] + l_toe[0]) // 2, (l_heel[1] + l_toe[1]) // 2) )
        if r_heel and r_toe:
            candidates.append( ((r_heel[0] + r_toe[0]) // 2, (r_heel[1] + r_toe[1]) // 2) )
            
        if not candidates:
            return None
            
        lowest_foot = max(candidates, key=lambda pt: pt[1])
        
        global_foot_x = lowest_foot[0] + bbox_offset[0]
        global_foot_y = lowest_foot[1] + bbox_offset[1]
        
        return (global_foot_x, global_foot_y)

    def draw_landmarks(self, image, crop_img, bbox_offset):
        """
        Draws the full skeleton on the original image.
        """
        if crop_img is None or crop_img.size == 0:
            return image
            
        image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            return image
            
        h, w, _ = crop_img.shape
        landmarks = detection_result.pose_landmarks[0]
        
        # Define skeleton connections
        POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        
        # Convert landmarks to global pixel coordinates
        pts = {}
        for idx, lm in enumerate(landmarks):
            if lm.visibility > 0.4 and getattr(lm, 'presence', 1.0) > 0.4:
                px = int(lm.x * w) + bbox_offset[0]
                py = int(lm.y * h) + bbox_offset[1]
                pts[idx] = (px, py)
                
        # Draw connections
        for p1, p2 in POSE_CONNECTIONS:
            if p1 in pts and p2 in pts:
                cv2.line(image, pts[p1], pts[p2], (0, 255, 0), 2)
                
        # Draw joints
        for idx, pt in pts.items():
            cv2.circle(image, pt, 3, (255, 0, 0), -1)
            
        return image
