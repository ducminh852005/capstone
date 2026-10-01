"""
Filling the frames in which the shuttle was not detected, from the detections around them.

The tracker only sees the shuttle every TrackNet stride (config.TRACKNET_BATCH_STRIDE) frames, and sometimes misses
one or more detections on top of that. Between two detections the shuttle is in ballistic
flight, so the missing positions follow inertia (the velocity implied by the detections on
both sides) plus gravity: one parabola per image axis, fitted to the neighbouring detections.

Detections form *chains*: runs of detections at most a stride or so apart within one flight.
Two kinds of gaps are filled:
  - "stride": the frames between consecutive detections of a chain (always empty at stride > 1);
  - "bridge": the gap between the end of one chain and the start of the next, only when both
    chains have at least `min_chain_len` detections (so each side gives a direction), the gap
    is not longer than `max_gap_frames`, the next chain did not start at a racket hit, and a
    single parabola fits the detections on both sides (`max_rms_px`): a hit inside the gap
    changes the flight and leaves no fit, so the gap stays empty.

A filled point is only accepted if it is plausible for the court (see court_constraint):
inside the shuttle ROI and not below the floor. All points of a gap must pass, otherwise the
whole gap stays empty. The court homography only describes the floor plane, so it can rule out
a point that would be under the floor but says nothing about the shuttle's height: a shuttle
high in the air projects onto the floor far behind the court, which is perfectly fine.

Filled points are estimates: they are for display and analysis, never fed back into the
tracker.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from functools import lru_cache

import numpy as np

from . import config, court_model


@dataclass(frozen=True)
class Detection:
    frame_idx: int
    pt: Tuple[float, float]              # (x, y) full-frame pixels
    flight_id: Optional[int] = None      # ShuttleDetector.flight_id at that frame (None = unknown)
    start: Optional[str] = None          # ShuttleDetector.start_source if this detection started a track


@dataclass(frozen=True)
class FilledPoint:
    frame_idx: int
    pt: Tuple[float, float]
    kind: str                            # "stride" (inside a chain) or "bridge" (between two chains)


def split_chains(dets: Sequence[Detection], max_link_frames: int) -> List[List[Detection]]:
    """Runs of detections at most max_link_frames apart that belong to the same flight."""
    chains: List[List[Detection]] = []
    cur: List[Detection] = []
    for d in dets:
        if cur:
            prev = cur[-1]
            same_flight = d.flight_id is None or prev.flight_id is None or d.flight_id == prev.flight_id
            if d.frame_idx - prev.frame_idx > max_link_frames or not same_flight:
                chains.append(cur)
                cur = []
        cur.append(d)
    if cur:
        chains.append(cur)
    return chains


def court_constraint(H_inv=None, roi=None,
                     margin_m: float = config.UMPIRE_FLOOR_BUFFER_M) -> Callable[[Tuple[float, float]], bool]:
    """
    Plausibility check for a filled image point.
      - roi (x0, y0, x1, y1): the point must lie inside it.
      - H_inv (image -> world homography of the court floor): the point must not be beyond the
        image line of the court's near edge (pushed margin_m metres out) on the far side from
        the court. Anything there would be under the floor, whatever the shuttle's height.
    A check is skipped when its input is None.
    """
    edge = None
    if H_inv is not None:
        H = np.linalg.inv(H_inv)
        p1, p2, centre = court_model.world_to_img(
            [(-margin_m, -margin_m), (-margin_m, court_model.COURT_WIDTH + margin_m),
             (court_model.COURT_LENGTH / 2, court_model.COURT_WIDTH / 2)], H)
        d = p2 - p1
        side_of_court = float(d[0] * (centre[1] - p1[1]) - d[1] * (centre[0] - p1[0]))
        edge = (p1, d, 1.0 if side_of_court >= 0 else -1.0)

    def allowed(pt) -> bool:
        if roi is not None:
            x0, y0, x1, y1 = roi
            if not (x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1):
                return False
        if edge is not None:
            p1, d, sign = edge
            if sign * float(d[0] * (pt[1] - p1[1]) - d[1] * (pt[0] - p1[0])) < 0:
                return False
        return True

    return allowed


def _fit(window: Sequence[Detection]):
    """Per-axis parabola (line if only two detections) through the window: (cx, cy, t0, rms_px).
    Results are cached by the window's (frame, x, y) values: when a new detection arrives only
    the last gaps change, the others are looked up (callers must not modify the arrays)."""
    return _fit_cached(tuple((d.frame_idx, d.pt[0], d.pt[1]) for d in window))


@lru_cache(maxsize=1024)
def _fit_cached(points):
    t = np.array([p[0] for p in points], dtype=float)
    t0 = float(t.mean())
    tt = t - t0
    xs = np.array([p[1] for p in points], dtype=float)
    ys = np.array([p[2] for p in points], dtype=float)
    deg = min(2, len(points) - 1)
    cx, cy = np.polyfit(tt, xs, deg), np.polyfit(tt, ys, deg)
    resid = np.hypot(np.polyval(cx, tt) - xs, np.polyval(cy, tt) - ys)
    return cx, cy, t0, float(np.sqrt(np.mean(resid ** 2)))


def _fill_between(a: Detection, b: Detection, window, kind, allowed, max_rms_px,
                  trusted_gap_frames) -> List[FilledPoint]:
    cx, cy, t0, rms = _fit(window)
    gap = b.frame_idx - a.frame_idx
    limit = max_rms_px * min(1.0, trusted_gap_frames / gap)      # the longer the gap, the tighter
    if len(window) >= 4 and rms > limit:
        return []                       # not one continuous flight: leave the gap empty
    frames = np.arange(a.frame_idx + 1, b.frame_idx)
    tt = frames - t0
    pts = [(float(x), float(y)) for x, y in zip(np.polyval(cx, tt), np.polyval(cy, tt))]
    if allowed is not None and not all(allowed(p) for p in pts):
        return []
    return [FilledPoint(int(f), p, kind) for f, p in zip(frames, pts)]


def fill_gaps(dets: Sequence[Detection],
              allowed: Optional[Callable[[Tuple[float, float]], bool]] = None,
              max_link_frames: int = config.SHUTTLE_FILL_LINK_FRAMES,
              max_gap_frames: int = config.SHUTTLE_FILL_MAX_GAP_FRAMES,
              min_chain_len: int = config.SHUTTLE_FILL_MIN_CHAIN_LEN,
              stride_fit_points: int = config.SHUTTLE_FILL_STRIDE_FIT_POINTS,
              bridge_fit_points: int = config.SHUTTLE_FILL_BRIDGE_FIT_POINTS,
              max_rms_px: float = config.SHUTTLE_FILL_MAX_RMS_PX,
              trusted_gap_frames: int = config.SHUTTLE_FILL_TRUSTED_GAP_FRAMES) -> List[FilledPoint]:
    """
    Estimated positions for the frames between detections (see the module docstring).
    dets: detections in increasing frame order. Returns the filled points, in frame order.
    """
    chains = split_chains(dets, max_link_frames)
    out: List[FilledPoint] = []

    for chain in chains:                                     # gaps inside a chain
        for k in range(len(chain) - 1):
            a, b = chain[k], chain[k + 1]
            if b.frame_idx - a.frame_idx <= 1:
                continue
            window = chain[max(0, k + 1 - stride_fit_points):k + 1 + stride_fit_points]
            out += _fill_between(a, b, window, "stride", allowed, max_rms_px, trusted_gap_frames)

    for left, right in zip(chains[:-1], chains[1:]):        # gaps between chains
        a, b = left[-1], right[0]
        if (len(left) < min_chain_len or len(right) < min_chain_len
                or b.frame_idx - a.frame_idx > max_gap_frames
                or b.frame_idx - a.frame_idx <= 1
                or b.start == "physics"):
            continue
        window = left[-bridge_fit_points:] + right[:bridge_fit_points]
        out += _fill_between(a, b, window, "bridge", allowed, max_rms_px, trusted_gap_frames)

    return sorted(out, key=lambda p: p.frame_idx)
