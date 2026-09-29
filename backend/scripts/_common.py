"""
Shared utilities for demo/tool scripts in the scripts/ directory.

Every script does `from _common import ...` first: importing this module puts backend/
on sys.path (so `from core import ...` works when a script is run directly, from any
working directory) and provides the boilerplate shared by the demos.
"""
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, court_model
from core.config import (
    DEFAULT_BACKEND,
    MINIMAP_BG_COLOR,
    MINIMAP_SCALE,
    MINIMAP_SIZE,
)

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Show core/ log messages on the console; call once at the top of a script's main."""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


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
        logger.warning("%s\nFalling back to backend='cv'.", e)
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
        logger.warning("No calibration data found (run scripts/calibrate_court.py).")
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

def draw_shuttle_trajectory(frame, trajectory, tail_length=None):
    """
    Draw the shuttle's recent flight path as a gradient-colored tail.
    Draws in-place on `frame` (no copy). tail_length defaults to config.TRAJECTORY_TAIL_LENGTH,
    read at call time so a live tuner can change it.
    """
    if tail_length is None:
        tail_length = config.TRAJECTORY_TAIL_LENGTH
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


# ========================= Live Tuner =================================

class TunerBinding:
    """
    One tunable value: maps a trackbar position (0..steps) to obj.attr in [min_val, max_val].
    Kept free of any GUI call so it can be unit-tested.
    """

    def __init__(self, label, obj, attr, min_val, max_val, step=1.0):
        if not hasattr(obj, attr):
            raise AttributeError(
                f"Cannot tune {label!r}: {type(obj).__name__} has no attribute {attr!r}")
        if max_val <= min_val or step <= 0:
            raise ValueError(f"Cannot tune {label!r}: need min_val < max_val and step > 0")
        self.label = label
        self.obj = obj
        self.attr = attr
        self.min_val = min_val
        self.step = step
        self.is_float = any(isinstance(v, float) for v in (min_val, max_val, step))
        self.steps = max(1, int(round((max_val - min_val) / step)))

    @property
    def value(self):
        return getattr(self.obj, self.attr)

    def position_for(self, value):
        """Trackbar position of `value`, clamped to the slider range."""
        pos = int(round((value - self.min_val) / self.step))
        return max(0, min(self.steps, pos))

    def initial_position(self):
        return self.position_for(self.value)

    def set_position(self, pos):
        """Write the value for trackbar position `pos` back to the bound attribute."""
        val = self.min_val + pos * self.step
        if not self.is_float:
            val = int(val)
        setattr(self.obj, self.attr, val)
        return val

    def text(self):
        return f"{self.label}: {self.value:.2f}" if self.is_float else f"{self.label}: {self.value}"


class LiveTuner:
    """
    Live Tuning Board using OpenCV Trackbars.
    Data binding: moving a slider sets an attribute on the bound object immediately.

    The bound attribute must already exist (a typo raises AttributeError instead of silently
    creating a new attribute), and the code under tuning must READ it at call time: an
    instance attribute (`detector.max_speed_ratio`) or a `config.X` module attribute read
    inside a function. A value copied at import (`from core.config import X`) or bound as a
    default argument will not react.
    """

    def __init__(self, window_name="Live Tuning Board"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._bindings = []

    def bind(self, label, obj, attr, min_val, max_val, step=1.0):
        """
        obj: the object to mutate (an instance, or a module such as `config`).
        attr: attribute name, e.g. 'SMASH_SPEED_THRESHOLD'. The slider starts at its current value.
        """
        binding = TunerBinding(label, obj, attr, min_val, max_val, step)
        self._bindings.append(binding)

        def on_change(pos):
            binding.set_position(pos)
            self.render()

        cv2.createTrackbar(label[:25], self.window_name, binding.initial_position(), binding.steps, on_change)
        self.render()

    def render(self):
        """Draw the current values to the OpenCV window as an image."""
        height = max(150, len(self._bindings) * 50 + 20)
        img = np.zeros((height, 450, 3), dtype=np.uint8)
        img[:] = (35, 40, 45)  # dark slate gray

        for i, b in enumerate(self._bindings):
            cv2.putText(img, b.text(), (20, 40 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

        cv2.imshow(self.window_name, img)


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
