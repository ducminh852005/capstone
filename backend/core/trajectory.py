"""
Player trajectory cleaning, distance and heatmap, following the "Làm sạch quỹ đạo"
section of the technical guideline. All positions are court coordinates in meters.
"""
import math

import cv2
import numpy as np
from scipy.signal import savgol_filter

from . import court_model
from . import config


def _to_dense(frame_idx, xy):
    """Frame indices + positions -> dense (T, 2) array over the full frame range, NaN where missing."""
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    start = int(frame_idx.min())
    dense = np.full((int(frame_idx.max()) - start + 1, 2), np.nan)
    dense[frame_idx - start] = xy
    return start, dense


def _segments(valid):
    """(start, end_exclusive) of every run of True values."""
    edges = np.diff(np.concatenate([[0], valid.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def reject_outliers(xy, fps, region=court_model.REGION_BUFFERED,
                    max_speed=config.TRAJECTORY_MAX_SPEED, reset_after_s=0.25):
    """
    NaN out points outside `region` and points implying a speed above `max_speed` m/s
    relative to the last accepted point. If rejections persist for `reset_after_s`, the
    anchor itself was probably the outlier, so the next point is accepted as a new anchor.
    """
    out = xy.copy()
    x_min, x_max, y_min, y_max = region
    outside = ~((out[:, 0] >= x_min) & (out[:, 0] <= x_max) & (out[:, 1] >= y_min) & (out[:, 1] <= y_max))
    out[outside] = np.nan

    last_i, rejected_since = None, None
    reset_frames = max(int(reset_after_s * fps), 1)
    xs, ys = out[:, 0].tolist(), out[:, 1].tolist()      # plain floats: math.hypot is ~100x faster than np.linalg.norm on a pair
    for i in np.flatnonzero(~np.isnan(out[:, 0])).tolist():
        if last_i is not None:
            speed = math.hypot(xs[i] - xs[last_i], ys[i] - ys[last_i]) * fps / (i - last_i)
            if speed > max_speed and (rejected_since is None or i - rejected_since < reset_frames):
                rejected_since = rejected_since if rejected_since is not None else i
                out[i] = np.nan
                continue
        last_i, rejected_since = i, None
    return out


def interpolate_short_gaps(xy, fps, max_gap_s=config.TRAJECTORY_MAX_GAP_S):
    """Linear interpolation over interior gaps shorter than max_gap_s; longer gaps stay NaN."""
    out = xy.copy()
    valid = ~np.isnan(out[:, 0])
    max_gap = int(max_gap_s * fps)
    for start, end in _segments(~valid):
        if start == 0 or end == len(out) or end - start > max_gap:
            continue
        t = np.arange(start, end)
        for k in range(2):
            out[start:end, k] = np.interp(t, [start - 1, end], [out[start - 1, k], out[end, k]])
    return out


def smooth(xy, fps, window_s=config.TRAJECTORY_SMOOTH_WINDOW_S,
           polyorder=config.TRAJECTORY_SMOOTH_POLYORDER):
    """Savitzky-Golay per continuous segment (segments shorter than the window are left as is)."""
    out = xy.copy()
    window = int(round(window_s * fps)) | 1  # odd
    if window <= polyorder:
        return out
    for start, end in _segments(~np.isnan(out[:, 0])):
        if end - start >= window:
            out[start:end] = savgol_filter(out[start:end], window, polyorder, axis=0)
    return out


def clean_player_track(frame_idx, xy, fps,
                       max_speed=config.TRAJECTORY_MAX_SPEED,
                       max_gap_s=config.TRAJECTORY_MAX_GAP_S,
                       smooth_window_s=config.TRAJECTORY_SMOOTH_WINDOW_S):
    """
    Guideline order: reject out-of-region / too-fast points -> interpolate gaps < 0.5 s ->
    Savitzky-Golay smoothing. Returns (first_frame, dense_xy) with NaN for missing frames.
    """
    start, dense = _to_dense(frame_idx, xy)
    dense = reject_outliers(dense, fps, max_speed=max_speed)
    dense = interpolate_short_gaps(dense, fps, max_gap_s)
    dense = smooth(dense, fps, smooth_window_s)
    return start, dense


def path_length(xy):
    """Total distance in meters; steps touching a NaN (a long gap) are not counted."""
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return float(np.nansum(steps))


def valid_ratio(xy):
    return float(np.mean(~np.isnan(xy[:, 0]))) if len(xy) else 0.0


def occupancy_heatmap(xy, fps, cell_m=0.25, region=court_model.REGION_BUFFERED, blur_m=0.25):
    """
    Seconds spent per cell (rows = x / depth from the baseline, cols = y / width).
    Every valid frame adds 1/fps s; a Gaussian blur of about the position error avoids
    showing detail the system cannot measure.
    """
    x_min, x_max, y_min, y_max = region
    rows = int(np.ceil((x_max - x_min) / cell_m))
    cols = int(np.ceil((y_max - y_min) / cell_m))
    grid = np.zeros((rows, cols), np.float32)
    pts = xy[~np.isnan(xy[:, 0])]
    r = ((pts[:, 0] - x_min) / cell_m).astype(int)
    c = ((pts[:, 1] - y_min) / cell_m).astype(int)
    ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    np.add.at(grid, (r[ok], c[ok]), 1.0 / fps)
    if blur_m > 0:
        grid = cv2.GaussianBlur(grid, (0, 0), blur_m / cell_m)
    return grid
