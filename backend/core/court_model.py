"""
Badminton court geometry and homography helpers shared by the whole pipeline.

World frame (meters), following the technical guideline:
- x along the court length: 0 = near baseline, 6.70 = net, 13.40 = far baseline
- y along the court width: 0 .. 6.10 (doubles sidelines)
Line coordinates are the CENTRE of each painted line (lines are ~4 cm wide).
"""
import json
import logging
import os

import cv2
import numpy as np

from . import config

logger = logging.getLogger(__name__)

DEFAULT_CALIB_PATH = config.CALIBRATION_PATH

# Default values (standard badminton court dimensions in meters)
COURT_LENGTH = 13.40
COURT_WIDTH = 6.10
NET_X = 6.70
BASELINE_X = 0.0
LONG_SERVICE_DOUBLES_X = 0.76
SHORT_SERVICE_X = 4.72
SIDELINE_DOUBLES_Y = (0.0, 6.10)
SIDELINE_SINGLES_Y = (0.46, 5.64)
CENTER_Y = 3.05
LINE_HALF_WIDTH = 0.02

# Court dimensions may be overridden by calibration.json. This is read ONCE at import because
# the derived constants below (regions, calibration corners) are computed from them; it is
# the only file access allowed at import time in core/.
if os.path.exists(DEFAULT_CALIB_PATH):
    try:
        with open(DEFAULT_CALIB_PATH, "r", encoding="utf-8") as f:
            _calib_data = json.load(f)
            if "court_dimensions" in _calib_data:
                _cd = _calib_data["court_dimensions"]
                COURT_LENGTH = _cd.get("COURT_LENGTH", COURT_LENGTH)
                COURT_WIDTH = _cd.get("COURT_WIDTH", COURT_WIDTH)
                NET_X = _cd.get("NET_X", NET_X)
                BASELINE_X = _cd.get("BASELINE_X", BASELINE_X)
                LONG_SERVICE_DOUBLES_X = _cd.get("LONG_SERVICE_DOUBLES_X", LONG_SERVICE_DOUBLES_X)
                SHORT_SERVICE_X = _cd.get("SHORT_SERVICE_X", SHORT_SERVICE_X)
                if "SIDELINE_DOUBLES_Y" in _cd:
                    SIDELINE_DOUBLES_Y = tuple(_cd["SIDELINE_DOUBLES_Y"])
                if "SIDELINE_SINGLES_Y" in _cd:
                    SIDELINE_SINGLES_Y = tuple(_cd["SIDELINE_SINGLES_Y"])
                CENTER_Y = _cd.get("CENTER_Y", CENTER_Y)
                LINE_HALF_WIDTH = _cd.get("LINE_HALF_WIDTH", LINE_HALF_WIDTH)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Failed to parse court dimensions from %s: %s", DEFAULT_CALIB_PATH, e)

# Corners clicked in scripts/calibrate_court.py: Bottom-Left, Bottom-Right, Net-Left, Net-Right
CALIB_WORLD_POINTS = [
    (BASELINE_X, SIDELINE_DOUBLES_Y[0]),
    (BASELINE_X, SIDELINE_DOUBLES_Y[1]),
    (NET_X, SIDELINE_DOUBLES_Y[0]),
    (NET_X, SIDELINE_DOUBLES_Y[1])
]

# (x_min, x_max, y_min, y_max): near half-court plus the buffer from the guideline
# (~1 m behind the baseline, ~0.5 m on each side). Used for player gating and heatmaps.
REGION_BUFFERED = (BASELINE_X - 1.0, NET_X, SIDELINE_DOUBLES_Y[0] - 0.5, SIDELINE_DOUBLES_Y[1] + 0.5)
# Wider region used only as a cheap pre-filter before running pose estimation.
REGION_PREFILTER = (BASELINE_X - 2.0, NET_X + 0.5, SIDELINE_DOUBLES_Y[0] - 1.5, SIDELINE_DOUBLES_Y[1] + 1.5)


def near_half_lines():
    """Painted line segments of the near half-court as ((x1, y1), (x2, y2)) in meters."""
    y0, y1 = SIDELINE_DOUBLES_Y
    lines = [
        ((BASELINE_X, y0), (BASELINE_X, y1)),
        ((LONG_SERVICE_DOUBLES_X, y0), (LONG_SERVICE_DOUBLES_X, y1)),
        ((SHORT_SERVICE_X, y0), (SHORT_SERVICE_X, y1)),
        ((BASELINE_X, CENTER_Y), (SHORT_SERVICE_X, CENTER_Y)),
    ]
    for y in (*SIDELINE_DOUBLES_Y, *SIDELINE_SINGLES_Y):
        lines.append(((BASELINE_X, y), (NET_X, y)))
    return lines


def full_court_lines():
    """All painted line segments of the full court as ((x1, y1), (x2, y2)) in meters."""
    y0, y1 = SIDELINE_DOUBLES_Y
    ys0, ys1 = SIDELINE_SINGLES_Y
    far_baseline = COURT_LENGTH
    far_long_service = COURT_LENGTH - LONG_SERVICE_DOUBLES_X
    far_short_service = COURT_LENGTH - SHORT_SERVICE_X

    lines = [
        # Baselines
        ((BASELINE_X, y0), (BASELINE_X, y1)),
        ((far_baseline, y0), (far_baseline, y1)),
        # Long service lines (doubles)
        ((LONG_SERVICE_DOUBLES_X, y0), (LONG_SERVICE_DOUBLES_X, y1)),
        ((far_long_service, y0), (far_long_service, y1)),
        # Short service lines
        ((SHORT_SERVICE_X, y0), (SHORT_SERVICE_X, y1)),
        ((far_short_service, y0), (far_short_service, y1)),
        # Net line
        ((NET_X, y0), (NET_X, y1)),
        # Doubles sidelines (full length)
        ((BASELINE_X, y0), (far_baseline, y0)),
        ((BASELINE_X, y1), (far_baseline, y1)),
        # Singles sidelines (full length)
        ((BASELINE_X, ys0), (far_baseline, ys0)),
        ((BASELINE_X, ys1), (far_baseline, ys1)),
        # Centre lines (near and far halves, between short service lines)
        ((BASELINE_X, CENTER_Y), (SHORT_SERVICE_X, CENTER_Y)),
        ((far_short_service, CENTER_Y), (far_baseline, CENTER_Y)),
    ]
    return lines


def sample_line_points(step_m=0.05, lines=None):
    """Evenly spaced world points along the model lines, shape (N, 2)."""
    pts = []
    for (xa, ya), (xb, yb) in (lines or near_half_lines()):
        n = max(int(np.hypot(xb - xa, yb - ya) / step_m), 1) + 1
        t = np.linspace(0.0, 1.0, n)
        pts.append(np.stack([xa + (xb - xa) * t, ya + (yb - ya) * t], axis=1))
    return np.concatenate(pts).astype(np.float32)


def world_to_img(pts, H):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def img_to_world(pts, H_inv):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)


def homography_from_points(image_points, world_points=CALIB_WORLD_POINTS):
    """World -> image homography. Exact DLT for 4 points, RANSAC when more are given."""
    src = np.asarray(world_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    if len(src) < 4:
        return None
    method = 0 if len(src) == 4 else cv2.RANSAC
    H, _ = cv2.findHomography(src, dst, method, 3.0)
    return H


def load_calibration(path=DEFAULT_CALIB_PATH):
    """Returns (H, H_inv) or (None, None). Prefers the refined "H" when it was saved."""
    if not os.path.exists(path):
        return None, None
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("H") is not None:
        H = np.asarray(data["H"], dtype=np.float64)
    else:
        H = homography_from_points(data.get("image_points", []))
    if H is None:
        return None, None
    return H, np.linalg.inv(H)


def in_region(xy, region=REGION_BUFFERED):
    x_min, x_max, y_min, y_max = region
    return x_min <= xy[0] <= x_max and y_min <= xy[1] <= y_max


def region_polygon_img(H, region=REGION_BUFFERED):
    x_min, x_max, y_min, y_max = region
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return np.int32(world_to_img(corners, H)).reshape(-1, 1, 2)


def _clamp_roi(x0, y0, x1, y1, frame_shape):
    h, w = frame_shape[:2]
    return max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)


def player_roi(H, frame_shape, region=REGION_BUFFERED, person_height_m=2.2, margin_px=40):
    """
    Image ROI that contains every player standing inside `region`: the projected
    region plus room above each corner for a standing person. The vertical pixel
    scale is approximated by the largest local ground-plane scale at that corner.
    """
    x_min, x_max, y_min, y_max = region
    corners = np.float32([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    base = world_to_img(corners, H)
    along_x = world_to_img(corners + [1.0, 0.0], H)
    along_y = world_to_img(corners + [0.0, 1.0], H)
    px_per_m = np.maximum(np.linalg.norm(along_x - base, axis=1), np.linalg.norm(along_y - base, axis=1))
    tops = base - np.stack([np.zeros_like(px_per_m), px_per_m * person_height_m], axis=1)
    allp = np.vstack([base, tops])
    x0, y0 = allp.min(axis=0) - margin_px
    x1, y1 = allp.max(axis=0) + margin_px
    return _clamp_roi(x0, y0, x1, y1, frame_shape)


def shuttle_roi(H, frame_shape, margin_frac=0.03, top_frac=0.10):
    """
    ROI for shuttle detection: horizontal span of the full-court lines (both halves,
    plus a margin), from `top_frac` of the frame height down to the bottom. The top
    band is dropped to ignore ceiling lights, the sides to ignore spectators.
    """
    h, w = frame_shape[:2]
    poly = region_polygon_img(H, (0.0, COURT_LENGTH, 0.0, COURT_WIDTH)).reshape(-1, 2)
    x0 = poly[:, 0].min() - margin_frac * w
    x1 = poly[:, 0].max() + margin_frac * w
    return _clamp_roi(x0, top_frac * h, x1, h, frame_shape)


def net_top_threshold_y(H, offset_px=50):
    """Image row the shuttle must rise above for a flight to count as a real shot."""
    net = world_to_img([(NET_X, 0.0), (NET_X, COURT_WIDTH)], H)
    return float(net[:, 1].mean()) - offset_px
