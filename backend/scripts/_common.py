"""
Shared utilities for test/demo scripts in the scripts/ directory.

Extracts boilerplate that was duplicated across test_trajectory.py,
test_smash.py, test_shuttle_tracker.py, test_auto_umpire.py, etc.
"""
import os
import sys
import time

import cv2
import numpy as np

# Add the backend directory to sys.path so `from core import ...` works
# when scripts are run directly (python scripts/test_xxx.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import court_model
from core.config import (
    DEFAULT_BACKEND,
    MINIMAP_BG_COLOR,
    MINIMAP_SCALE,
    MINIMAP_SIZE,
    TRAJECTORY_TAIL_LENGTH,
)


# ========================= Detector Init ==============================

def create_detector(backend=DEFAULT_BACKEND, **kwargs):
    """
    Create a ShuttleDetector with automatic fallback from tracknet to cv
    if the model weights file is missing.
    """
    from core.shuttle_tracker import ShuttleDetector

    try:
        return ShuttleDetector(backend=backend, **kwargs)
    except FileNotFoundError as e:
        if backend != "tracknet":
            raise
        print(f"WARNING: {e}\nFalling back to backend='cv'.")
        return ShuttleDetector(backend="cv", **kwargs)


# ========================= Video + Calibration ========================

def load_video_and_calibration(video_path):
    """
    Open a video and load court calibration in one call.

    Returns (reader, H, H_inv, roi) where roi is the shuttle detection ROI
    derived from the calibration (None if no calibration is available).
    """
    from core.video_io import ThreadedVideoReader

    reader = ThreadedVideoReader(video_path)
    H, H_inv = court_model.load_calibration()
    roi = None
    if H is not None:
        w, h = reader.size
        roi = court_model.shuttle_roi(H, (h, w))
    else:
        print("WARNING: No calibration data found.")
    return reader, H, H_inv, roi


# ========================= FPS Counter ================================

class FPSCounter:
    """Lightweight FPS tracker that updates once per second."""

    def __init__(self):
        self._start = time.time()
        self._count = 0
        self.fps = 0.0

    def tick(self):
        self._count += 1
        elapsed = time.time() - self._start
        if elapsed > 1.0:
            self.fps = self._count / elapsed
            self._start = time.time()
            self._count = 0
        return self.fps


# ========================= Drawing Helpers ============================

def draw_shuttle_trajectory(frame, trajectory, tail_length=TRAJECTORY_TAIL_LENGTH):
    """
    Draw the shuttle's recent flight path as a gradient-colored tail.
    Draws in-place on `frame` (no copy).
    """
    recent = [p for p in trajectory[-tail_length:] if p is not None]
    if not recent:
        return
    for i in range(1, len(recent)):
        t = i / len(recent)  # 0..1, older to newer
        color = (0, int(80 + 175 * t), 255)
        thickness = int(1 + 3 * t)
        cv2.line(frame, recent[i - 1], recent[i], color, thickness)
    cv2.circle(frame, recent[-1], 6, (0, 0, 255), -1)


def draw_court_overlay(frame, H):
    """Draw all court lines (full court) onto the frame using the homography."""
    if H is None:
        return
    for a, b in court_model.full_court_lines():
        pts = court_model.world_to_img([a, b], H)
        pt1 = (int(pts[0][0]), int(pts[0][1]))
        pt2 = (int(pts[1][0]), int(pts[1][1]))
        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)


def draw_bounce_markers(frame, bounce_events):
    """Draw IN/OUT cross markers at each confirmed bounce location."""
    for event in bounce_events:
        color = (0, 255, 0) if event.result == "IN" else (0, 0, 255)
        cv2.drawMarker(frame, event.image_pt, color,
                       markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
        cv2.putText(frame, event.result,
                    (event.image_pt[0] + 15, event.image_pt[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


# ========================= Minimap ====================================

class Minimap:
    """
    2D bird's-eye-view minimap of the court.

    World coordinate mapping:
        x (court length, 0..13.4) → vertical axis (bottom to top)
        y (court width,  0..6.1)  → horizontal axis (left to right)
    """

    def __init__(self, scale=MINIMAP_SCALE, size=MINIMAP_SIZE, bg_color=MINIMAP_BG_COLOR):
        self.scale = scale
        self.map_w, self.map_h = size
        self.bg_color = bg_color
        self.offset_x = (self.map_w - court_model.COURT_WIDTH * scale) / 2
        self.offset_y = (self.map_h - court_model.COURT_LENGTH * scale) / 2

    def world_to_minimap(self, x, y):
        """Convert world (x, y) in meters to minimap pixel coordinates."""
        px = int(y * self.scale + self.offset_x)
        py = int(self.map_h - (x * self.scale + self.offset_y))
        return (px, py)

    def render(self, bounce_events=None):
        """Create a fresh minimap image with court lines and optional bounce markers."""
        canvas = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        # Draw court lines
        for a, b in court_model.full_court_lines():
            pt1 = self.world_to_minimap(*a)
            pt2 = self.world_to_minimap(*b)
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

        # Draw bounce events
        if bounce_events:
            for event in bounce_events:
                color = (0, 255, 0) if event.result == "IN" else (0, 0, 255)
                pt = self.world_to_minimap(event.world_pt[0], event.world_pt[1])
                cv2.circle(canvas, pt, 6, color, -1)
                cv2.circle(canvas, pt, 12, color, 2)

        return canvas
