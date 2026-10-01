import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np

import _common
from core import court_model
from core.court_calibration import CourtCalibrator

# Global variables for drag and drop
image_points = []
dragging_idx = -1
hover_idx = -1

LABELS = ["Bottom-Left", "Bottom-Right", "Net-Left", "Net-Right"]


def interactive_calibration(image_path=None, video_path=None):
    global image_points

    calibrator = CourtCalibrator()
    if video_path:
        print("Building a clean background (median of 60 frames)...")
        image = calibrator.extract_clean_background(video_path)
    else:
        image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image from {video_path or image_path}")
        return

    # Start from the saved corners if any, else a rectangle in the middle of the screen
    h, w = image.shape[:2]
    config_path = court_model.DEFAULT_CALIB_PATH
    image_points = [[w // 4, h * 3 // 4], [w * 3 // 4, h * 3 // 4], [w // 4, h // 4], [w * 3 // 4, h // 4]]
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            image_points = [list(map(int, p)) for p in json.load(f).get("image_points", image_points)]

    world_points = court_model.CALIB_WORLD_POINTS
    refined_H = None  # set by 'R', cleared as soon as a point is dragged
    status = ""

    def mouse_callback(event, x, y, flags, param):
        global dragging_idx, hover_idx
        nonlocal refined_H

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
                refined_H = None
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = -1

    window_name = "Interactive Calibration (Drag the Red Points) - R: refine, SPACE: save"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("---------------------------------------------------")
    print("INTERACTIVE CALIBRATION MODE")
    print("Drag the 4 RED DOTS to the following corners of the NEAR HALF COURT:")
    for i, label in enumerate(LABELS, start=1):
        print(f"{i}. {label}")
    print("Press R to refine the homography on the white lines,")
    print("SPACE or ENTER to save, Q or ESC to quit.")
    print("---------------------------------------------------")

    while True:
        display_img = image.copy()

        H = refined_H if refined_H is not None else calibrator.get_rough_homography(display_img, image_points, world_points)
        if H is not None:
            display_img = calibrator.draw_court_frame(display_img, H)

        for i, pt in enumerate(image_points):
            color = (0, 255, 0) if i == hover_idx else (0, 0, 255)
            cv2.circle(display_img, tuple(pt), 8, color, -1)
            cv2.putText(display_img, LABELS[i], (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, LABELS[i], (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        if status:
            cv2.putText(display_img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(15) & 0xFF
        if key in (ord('r'), ord('R')) and H is not None:
            rough_H = calibrator.get_rough_homography(image, image_points, world_points)
            mask = calibrator.extract_white_lines(image, rough_H)
            before = calibrator.line_overlap_score(mask, rough_H)
            refined_H, after = calibrator.refine_homography(mask, rough_H)
            status = f"Line overlap: {100 * before:.1f}% -> {100 * after:.1f}%"
            print(status)
        elif key in (32, 13):  # Space or Enter
            # Read existing data first to preserve court_dimensions
            data = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError) as e:
                    logging.getLogger(__name__).warning("Could not read existing %s (%s); overwriting", config_path, e)

            data["image_points"] = image_points
            if refined_H is not None:
                data["H"] = refined_H.tolist()
            elif "H" in data:
                del data["H"]

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Calibration saved to {config_path}{' (with refined H)' if refined_H is not None else ''}!")
            break
        elif key in (ord('q'), 27):  # Q or Esc
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _common.setup_logging()
    _common.setup_gui()
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="sample frame of the video")
    ap.add_argument("--video", help="build a clean background from this video instead (recommended for refinement)")
    args = ap.parse_args()
    if not args.image and not args.video:
        print("Usage: python calibrate_court.py <path_to_image> | --video <path_to_video>")
        sys.exit(1)
    interactive_calibration(args.image, args.video)
