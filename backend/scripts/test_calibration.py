import os
import sys
import cv2
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.court_calibration import CourtCalibrator

# Global variables for drag and drop
image_points = []
dragging_idx = -1
hover_idx = -1

def interactive_calibration(image_path):
    global image_points, dragging_idx, hover_idx
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image from {image_path}")
        return

    # Start with 4 default points forming a rectangle in the middle of the screen
    h, w = image.shape[:2]
    image_points = [
        [w//4, h*3//4],     # 0: Bottom-Left
        [w*3//4, h*3//4],   # 1: Bottom-Right
        [w//4, h//4],       # 2: Net-Left
        [w*3//4, h//4]      # 3: Net-Right
    ]
    
    world_points = [
        (0, 0),         # Bottom-Left
        (0, 6.10),      # Bottom-Right
        (6.70, 0),      # Net-Left
        (6.70, 6.10)    # Net-Right
    ]
    
    calibrator = CourtCalibrator()
    
    def mouse_callback(event, x, y, flags, param):
        global image_points, dragging_idx, hover_idx
        
        # Check hover
        hover_idx = -1
        for i, pt in enumerate(image_points):
            if np.linalg.norm(np.array(pt) - np.array([x, y])) < 20:
                hover_idx = i
                break
                
        if event == cv2.EVENT_LBUTTONDOWN:
            if hover_idx != -1:
                dragging_idx = hover_idx
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_idx != -1:
                image_points[dragging_idx] = [x, y]
                
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = -1

    window_name = "Interactive Calibration (Drag the Red Points) - Press SPACE to Save"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("---------------------------------------------------")
    print("INTERACTIVE CALIBRATION MODE")
    print("Drag the 4 RED DOTS to the following corners of the NEAR HALF COURT:")
    print("1. Bottom-Left")
    print("2. Bottom-Right")
    print("3. Net-Left")
    print("4. Net-Right")
    print("Press SPACE or ENTER when you are done.")
    print("---------------------------------------------------")

    while True:
        display_img = image.copy()
        
        # Calculate Homography and draw the green court real-time
        try:
            H = calibrator.get_rough_homography(display_img, image_points, world_points)
            if H is not None:
                display_img = calibrator.draw_court_frame(display_img, H)
        except Exception as e:
            pass
            
        # Draw the draggable points on top
        for i, pt in enumerate(image_points):
            color = (0, 255, 0) if i == hover_idx else (0, 0, 255)
            cv2.circle(display_img, tuple(pt), 8, color, -1)
            # Label
            labels = ["Bottom-Left", "Bottom-Right", "Net-Left", "Net-Right"]
            cv2.putText(display_img, labels[i], (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, labels[i], (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        
        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(15) & 0xFF
        if key == 32 or key == 13: # Space or Enter
            # Save to JSON
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calibration.json")
            with open(config_path, "w") as f:
                json.dump({"image_points": image_points}, f)
            print(f"Calibration points saved to {config_path}!")
            break
        elif key == ord('q') or key == 27: # Q or Esc
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_calibration.py <path_to_image>")
        sys.exit(1)
        
    interactive_calibration(sys.argv[1])
