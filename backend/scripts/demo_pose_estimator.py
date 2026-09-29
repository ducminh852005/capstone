import sys

import cv2

import _common  # noqa: F401  (puts backend/ on sys.path)
from core.pose_estimator import PoseEstimator, select_foot_point

def run_mediapipe_pose(image_path):
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Initialize MediaPipe Pose Estimator
    print("Initializing MediaPipe Heavy model...")
    estimator = PoseEstimator(model_complexity=2)

    # Ask the user to draw a bounding box around a player (simulating YOLO's output)
    print("Please draw a bounding box around a player using your mouse, then press ENTER or SPACE.")
    cv2.namedWindow("Select Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Player", 1280, 720)
    bbox = cv2.selectROI("Select Player", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Player")

    if bbox[2] <= 0 or bbox[3] <= 0:
        print("Invalid or empty bounding box selected. Please drag a valid rectangle. Exiting...")
        return

    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    print(f"Selected Bounding Box: x={x}, y={y}, w={w}, h={h}")

    bbox_xyxy = (x, y, x + w, y + h)

    # Estimate pose on a padded crop and extract the foot point
    print("Extracting foot coordinates...")
    landmarks = estimator.detect_landmarks(image, bbox_xyxy)
    foot_point, source = (None, None)
    if landmarks is not None:
        foot_point, source = select_foot_point(landmarks, h)
        if foot_point is not None:
            foot_point = (int(round(foot_point[0])), int(round(foot_point[1])))

    # Draw the full skeleton on a copy of the original image
    result_image = estimator.draw_landmarks(image.copy(), landmarks)

    # Draw YOLO Bounding Box (Blue)
    cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(result_image, "YOLO BBox", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw the final calculated foot point (Red Dot)
    if foot_point:
        print(f"SUCCESS: Estimated Foot Point at Global Coordinate: {foot_point} (source: {source})")
        cv2.circle(result_image, foot_point, radius=6, color=(0, 0, 255), thickness=-1)
        cv2.putText(result_image, "Foot Point", (foot_point[0] + 10, foot_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        print("WARNING: Could not detect feet in the selected area.")

    # Show result
    print("Displaying result... Press any key to close the window.")
    cv2.namedWindow("Pose Estimation Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Estimation Result", 1280, 720)
    cv2.imshow("Pose Estimation Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_pose_estimator.py <path_to_image>")
        sys.exit(1)

    run_mediapipe_pose(sys.argv[1])
