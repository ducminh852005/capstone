import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CourtCalibrator:
    def __init__(self, court_model_points=None):
        """
        Initialize the calibrator. 
        court_model_points is a dictionary mapping point indices to (x, y) real-world coordinates.
        Standard half-court model (x in [0, 13.4], y in [0, 6.1]):
        - (0, 0): bottom left corner of half court
        - (13.4, 6.1): top right corner of full court
        """
        # Default court points can be defined here based on the guideline
        self.court_model_points = court_model_points or self._get_default_points()

    def _get_default_points(self):
        # According to the guideline: x: 0-13.40m, y: 0-6.10m
        # Coordinates will be built here
        return {
            "bottom_left": (0, 0),
            "bottom_right": (0, 6.10),
            "net_left": (6.70, 0),
            "net_right": (6.70, 6.10)
        }

    def extract_clean_background(self, video_path: str, num_frames=60):
        """
        Step 2: Clean background extraction by taking the median of N random/spread frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            num_frames = total_frames

        frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            return None
        
        # Calculate median along the time axis (axis=0)
        median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
        return median_frame

    def get_rough_homography(self, bg_image, keypoints_image, keypoints_world):
        """
        Step 3: Rough homography using CourtKeyNet outputs or classical detection.
        keypoints_image: list of points (x,y) in image space.
        keypoints_world: list of corresponding points in world space.
        """
        if len(keypoints_image) < 4 or len(keypoints_world) < 4:
            logger.error("At least 4 points are required for homography.")
            return None
            
        src_pts = np.array(keypoints_world).reshape(-1, 1, 2)
        dst_pts = np.array(keypoints_image).reshape(-1, 1, 2)
        
        # Find homography matrix H mapping World -> Image
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def extract_white_lines(self, bg_image):
        """
        Step 4: Extract white lines.
        Top-hat transform + Otsu thresholding + masking.
        """
        gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Top-hat transform highlights bright objects (lines) on dark background (court)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Otsu thresholding
        _, thresholded = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresholded

    def refine_homography(self, thresholded_image, rough_H):
        """
        Step 5: Refine homography using white pixels.
        (Draft logic for accumulating white pixels along lines).
        """
        # Implement homography refinement
        pass

    def validate_calibration(self, H, test_image_points, test_world_points, threshold=5.0):
        """
        Step 7: Validation.
        Check if reprojection error is acceptable.
        """
        if H is None:
            return False
            
        src_pts = np.array(test_world_points).reshape(-1, 1, 2)
        projected_pts = cv2.perspectiveTransform(src_pts, H)
        
        errors = np.linalg.norm(test_image_points - projected_pts.squeeze(), axis=1)
        mean_error = np.mean(errors)
        
        if mean_error > threshold:
            logger.warning(f"Calibration warning: Mean error {mean_error:.2f} > threshold {threshold}")
            return False
        
        return True

    def calculate_intersection_from_lines(self, line1_pts, line2_pts):
        """
        Calculate intersection of two lines using homogeneous coordinates cross product.
        Useful when a corner is out of frame.
        line1_pts: list of points [(x1,y1), (x2,y2)] on line 1
        line2_pts: list of points [(x3,y3), (x4,y4)] on line 2
        """
        # Convert to homogeneous points
        p1 = np.array([line1_pts[0][0], line1_pts[0][1], 1])
        p2 = np.array([line1_pts[1][0], line1_pts[1][1], 1])
        p3 = np.array([line2_pts[0][0], line2_pts[0][1], 1])
        p4 = np.array([line2_pts[1][0], line2_pts[1][1], 1])
        
        # Line equations: l = p1 x p2
        l1 = np.cross(p1, p2)
        l2 = np.cross(p3, p4)
        
        # Intersection point: x = l1 x l2
        pt_homogeneous = np.cross(l1, l2)
        
        if pt_homogeneous[2] == 0:
            # Parallel lines
            return None
            
        # Convert back to Cartesian coordinates
        x = pt_homogeneous[0] / pt_homogeneous[2]
        y = pt_homogeneous[1] / pt_homogeneous[2]
        
        return (int(x), int(y))
        
    def draw_court_frame(self, image, H):
        """
        Draw the court frame on the image using the homography matrix H.
        """
        if H is None:
            return image
            
        result = image.copy()
        
        # Define world coordinates of the court lines to draw
        # Outer boundary of half court (near side)
        pts_world = np.array([
            [0, 0], [13.4, 0], [13.4, 6.1], [0, 6.1]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Project world coordinates to image coordinates
        pts_image = cv2.perspectiveTransform(pts_world, H)
        
        if pts_image is not None:
            pts_image = np.int32(pts_image)
            cv2.polylines(result, [pts_image], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw corners
            for pt in pts_image:
                cv2.circle(result, tuple(pt[0]), 5, (0, 0, 255), -1)
                
        return result

if __name__ == "__main__":
    pass
