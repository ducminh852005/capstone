import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ShuttleTrajectoryProcessor:
    def __init__(self, max_gap_frames=60):
        """
        max_gap_frames: Maximum number of consecutive missing frames to interpolate.
                        If the shuttle is out of frame for longer than this (e.g. 1 second at 60fps), 
                        we might not want to interpolate to avoid wild physics.
        """
        self.max_gap_frames = max_gap_frames
        
        # Precompute Gamma LUT table to save CPU (Values < 1.0 darken the image)
        gamma = 0.5
        invGamma = 1.0 / gamma
        self.gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Precompute Morphological kernel
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def preprocess_frame_for_tracking(self, frame):
        """
        Preprocesses a frame to reduce the impact of glare, ceiling lights.
        Optimized to run in real-time.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gamma Correction (using precomputed table)
        gamma_corrected = cv2.LUT(gray, self.gamma_table)
        
        # 3. Morphological Top-Hat Transform (using precomputed 5x5 kernel)
        # 5x5 is fast and perfectly sized for a shuttlecock
        top_hat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_TOPHAT, self.tophat_kernel)
        
        # 4. Blend and return as grayscale (Background Subtractor works on grayscale too!)
        enhanced = cv2.addWeighted(gamma_corrected, 0.7, top_hat, 0.5, 0)
        return enhanced

    def fill_missing_trajectory(self, trajectory):
        """
        Fills missing shuttlecock coordinates (when it flies out of frame) using Polynomial Fitting (Parabola).
        
        trajectory: A list of tuples (x, y) for each frame. Missing frames should be None.
                    Example: [(100, 200), (105, 190), None, None, (120, 160), ...]
        
        Returns a new trajectory list with the missing gaps interpolated smoothly.
        """
        n_frames = len(trajectory)
        filled_trajectory = list(trajectory)
        
        # Extract indices where data exists
        valid_indices = [i for i, pt in enumerate(trajectory) if pt is not None]
        
        if len(valid_indices) < 3:
            # Not enough data to fit a parabola
            return filled_trajectory

        # Find gaps (consecutive None values)
        gaps = []
        start_gap = None
        for i in range(n_frames):
            if filled_trajectory[i] is None:
                if start_gap is None:
                    start_gap = i
            else:
                if start_gap is not None:
                    gaps.append((start_gap, i - 1))
                    start_gap = None
                    
        # If it ends with a gap, we might extrapolate (but it's risky, so we just ignore trailing Nones)
        
        for gap_start, gap_end in gaps:
            gap_length = gap_end - gap_start + 1
            if gap_length > self.max_gap_frames:
                logger.info(f"Gap of {gap_length} frames is too large, skipping interpolation.")
                continue
                
            # To fit a good parabola, we need points before and after the gap.
            # Let's take up to 10 points before and 10 points after.
            window = 15
            before_indices = [idx for idx in valid_indices if idx < gap_start][-window:]
            after_indices = [idx for idx in valid_indices if idx > gap_end][:window]
            
            fit_indices = before_indices + after_indices
            if len(fit_indices) < 4:
                continue # Need enough points to establish the curve
                
            X_data = np.array(fit_indices)
            Y_x = np.array([trajectory[i][0] for i in fit_indices])
            Y_y = np.array([trajectory[i][1] for i in fit_indices])
            
            # Fit a 2nd degree polynomial (parabola) for both x and y over time (frames)
            # x(t) = a*t^2 + b*t + c (Though x(t) is usually linear, perspective makes it a curve)
            # y(t) = a*t^2 + b*t + c (Gravity makes this a parabola)
            poly_x = np.polyfit(X_data, Y_x, deg=2)
            poly_y = np.polyfit(X_data, Y_y, deg=2)
            
            # Evaluate the polynomial for the missing frames
            for i in range(gap_start, gap_end + 1):
                pred_x = np.polyval(poly_x, i)
                pred_y = np.polyval(poly_y, i)
                filled_trajectory[i] = (int(pred_x), int(pred_y))
                
        return filled_trajectory

    def predict_reentry_point(self, trajectory_history, frame_rate=60):
        """
        Real-time prediction: If shuttle is currently lost, predict where and when it will re-enter.
        This uses the last known ascending points.
        """
        # Implementation for real-time Kalman filtering goes here.
        pass

class ShuttleDetector:
    def __init__(self):
        self.processor = ShuttleTrajectoryProcessor()
        # KNN Subtractor is great for detecting small, fast-moving objects
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=400, detectShadows=False)
        self.trajectory = [] # List of (x, y) coordinates of the shuttle
        
    def detect(self, frame):
        """
        Detects the shuttlecock in the current frame using classical CV techniques.
        """
        # 1. Preprocess: Enhance small bright spots
        processed = self.processor.preprocess_frame_for_tracking(frame)
        
        # 2. Background Subtraction: Isolate moving objects
        fg_mask = self.bg_subtractor.apply(processed)
        
        # 3. Clean up mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        # Dilate to make small shuttlecock more visible
        fg_mask = cv2.dilate(fg_mask, None, iterations=1)
        
        # 4. Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            # Relaxed area limits: Shuttle can be very tiny (blurred) or a bit larger
            if 2 < area < 500: 
                (x, y), radius = cv2.minEnclosingCircle(c)
                candidates.append((int(x), int(y), area))
                
        best_pt = None
        
        # 5. Physics-Based Tracking Heuristics
        if candidates:
            # Extract up to the last 5 valid positions
            valid_history = [pt for pt in self.trajectory[-5:] if pt is not None]
            
            if len(valid_history) >= 1:
                last_pos = valid_history[-1]
                predicted_pos = last_pos
                
                # If we have at least 2 points, we can calculate Momentum (Velocity)
                if len(valid_history) >= 2:
                    prev_pos = valid_history[-2]
                    vx = last_pos[0] - prev_pos[0]
                    vy = last_pos[1] - prev_pos[1]
                    
                    # Physics bias: Add a slight downward gravity pull to the predicted path
                    vy += 2 
                    
                    # Predict where the shuttle SHOULD be based on its momentum
                    predicted_pos = (last_pos[0] + vx, last_pos[1] + vy)
                
                best_score = float('inf')
                
                for c in candidates:
                    cx, cy = c[0], c[1]
                    
                    # Distance to last known position
                    dist_to_last = ((cx - last_pos[0])**2 + (cy - last_pos[1])**2) ** 0.5
                    
                    # Distance to predicted physical position
                    dist_to_pred = ((cx - predicted_pos[0])**2 + (cy - predicted_pos[1])**2) ** 0.5
                    
                    # Cap maximum speed per frame (prevents teleporting to a shoe)
                    # A smash can be very fast, ~400 pixels per frame maximum
                    if dist_to_last > 400:
                        continue
                        
                    # Score heavily favors objects that follow the predicted Parabolic/Momentum path
                    # But leaves 30% weight to dist_to_last to handle sudden bounces/racket hits
                    score = (dist_to_pred * 0.7) + (dist_to_last * 0.3)
                    
                    if score < best_score:
                        best_score = score
                        best_pt = (cx, cy)
            else:
                # No history: Assume shuttle is high in the air (Lowest Y)
                best_candidate = min(candidates, key=lambda c: c[1])
                best_pt = (best_candidate[0], best_candidate[1])
                
        self.trajectory.append(best_pt)
        return best_pt, fg_mask

    def draw_trajectory(self, frame):
        """
        Draws the trailing path of the shuttlecock.
        """
        annotated = frame.copy()
        
        # Draw the last 15 points as a fading tail
        tail_length = 20
        recent_path = self.trajectory[-tail_length:]
        
        for i in range(1, len(recent_path)):
            pt1 = recent_path[i-1]
            pt2 = recent_path[i]
            
            if pt1 is None or pt2 is None:
                continue
                
            # Fading color (Older is darker red, newer is bright yellow)
            thickness = int(np.interp(i, [0, tail_length], [1, 4]))
            cv2.line(annotated, pt1, pt2, (0, 255, 255), thickness)
            
        # Draw current position
        if len(self.trajectory) > 0 and self.trajectory[-1] is not None:
            cv2.circle(annotated, self.trajectory[-1], 6, (0, 0, 255), -1)
            
        return annotated

if __name__ == "__main__":
    # Small test
    processor = ShuttleTrajectoryProcessor()
    
    # Simulate a shuttle going up, going missing for 3 frames, then coming down
    mock_path = [
        (100, 500), (110, 400), (120, 310), (130, 230),  # ascending
        None, None, None,                                # out of frame!
        (170, 230), (180, 310), (190, 400), (200, 500)   # descending
    ]
    
    fixed_path = processor.fill_missing_trajectory(mock_path)
    for i, pt in enumerate(fixed_path):
        print(f"Frame {i}: {pt} {'(Interpolated)' if mock_path[i] is None else ''}")
