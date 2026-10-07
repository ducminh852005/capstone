"""
Movement statistics of one player over a stretch of a clip, from his foot positions on the court
(metres, court frame of court_model.py): distance, speed, time per zone, waiting position, recovery
after each hit and rest. Follows the "Làm sạch quỹ đạo, chỉ số và heatmap" section of the guideline.

The track is cleaned over the whole clip (so the smoothing sees the pre-roll too) and measured only
between first_frame_idx and last_frame_idx. Frames are clip-local, speeds in m/s.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .trajectory import clean_player_track
from .zones import ZONE_IDS, zone_counts

logger = logging.getLogger(__name__)

BASE_FROM_RALLY = "rally"      # waiting position measured while a rally was on
BASE_FROM_CLIP = "clip"        # no usable stillness in a rally: measured over the whole analysed stretch


@dataclass
class MovementParams:
    """Thresholds of the movement statistics; MovementParams.from_config() gives the configured ones."""
    max_speed_mps: float
    max_gap_s: float
    smooth_window_s: float
    speed_levels_mps: Tuple[float, ...]
    still_speed_mps: float
    rest_min_s: float
    base_min_s: float
    home_radius_m: float
    recovery_max_s: float
    min_valid_ratio: float
    small_sample_s: float
    depth_edges_m: Tuple[float, float]
    speed_percentile: float = config.MOVE_SPEED_PERCENTILE

    @classmethod
    def from_config(cls) -> "MovementParams":
        return cls(config.TRAJECTORY_MAX_SPEED, config.TRAJECTORY_MAX_GAP_S, config.TRAJECTORY_SMOOTH_WINDOW_S,
                   tuple(config.MOVE_SPEED_LEVELS_MPS), config.MOVE_STILL_SPEED_MPS, config.MOVE_REST_MIN_S,
                   config.MOVE_BASE_MIN_S, config.MOVE_HOME_RADIUS_M, config.MOVE_RECOVERY_MAX_S,
                   config.MOVE_MIN_VALID_RATIO, config.MOVE_SMALL_SAMPLE_S, tuple(config.ZONE_DEPTH_EDGES_M))


@dataclass
class RecoveryEvent:
    """How the player got back to his waiting position after a hit."""
    hit_frame_idx: int
    farthest_frame_idx: int                # where he was farthest from the waiting position before the next hit
    excursion_m: float                     # that distance
    recovery_s: Optional[float]            # seconds from there until back within the home radius; None = not back in time


@dataclass
class MovementStats:
    first_frame_idx: int
    last_frame_idx: int
    duration_s: float                      # length of the analysed stretch
    valid_ratio: float                     # fraction of its frames with a usable position
    reliable: bool                         # valid_ratio >= MOVE_MIN_VALID_RATIO
    small_sample: bool                     # shorter than MOVE_SMALL_SAMPLE_S
    distance_m: float
    distance_along_m: float                # part of it along the court (x)
    distance_across_m: float               # and across it (y)
    mean_speed_mps: Optional[float]        # None without two consecutive usable positions
    p95_speed_mps: Optional[float]
    speed_levels_mps: List[float]          # edges of the levels below
    speed_level_share: List[float]         # share of the moving time per level (len(edges) + 1, sums to 1)
    zone_time_s: Dict[str, float]          # seconds per zone (zones.ZONE_IDS)
    base_position_m: Optional[List[float]]
    base_source: Optional[str]
    recoveries: List[RecoveryEvent] = field(default_factory=list)
    mean_recovery_s: Optional[float] = None
    rest_s: float = 0.0
    rest_intervals_frames: List[Tuple[int, int]] = field(default_factory=list)   # inclusive frame ranges


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _cleaned_segment(foot_world_m, fps, first, last, params) -> np.ndarray:
    """Cleaned (T, 2) positions of frames first..last, NaN where unusable."""
    n = len(foot_world_m)
    track = np.full((n, 2), np.nan)
    idx = [i for i, p in enumerate(foot_world_m) if p is not None]
    if len(idx) >= 2:
        start, dense = clean_player_track(idx, [foot_world_m[i] for i in idx], fps, params.max_speed_mps,
                                          params.max_gap_s, params.smooth_window_s)
        track[start:start + len(dense)] = dense
    return track[first:last + 1]


def _waiting_position(seg, still, in_rally, fps, params):
    """(position, source): median of the still frames inside rallies, else of all still frames."""
    for mask, source in ((still & in_rally, BASE_FROM_RALLY), (still, BASE_FROM_CLIP)):
        if mask.sum() / fps >= params.base_min_s:
            return [float(v) for v in np.median(seg[mask], axis=0)], source
    return None, None


def _recoveries(seg, base, hit_rel, fps, first, params) -> List[RecoveryEvent]:
    """One RecoveryEvent per hit (frames relative to the segment) with a usable track after it."""
    out = []
    horizon = int(params.recovery_max_s * fps)
    for k, rel in enumerate(hit_rel):
        end = min(rel + horizon, len(seg) - 1)
        if k + 1 < len(hit_rel):
            end = min(end, hit_rel[k + 1])
        dist = np.linalg.norm(seg[rel:end + 1] - base, axis=1)
        if np.all(np.isnan(dist)):
            continue
        far = int(np.nanargmax(dist))
        recovery = None
        if dist[far] <= params.home_radius_m:
            recovery = 0.0                                       # never left the waiting position
        else:
            back = np.flatnonzero(dist[far + 1:] <= params.home_radius_m)
            if len(back):
                recovery = (int(back[0]) + 1) / fps
        out.append(RecoveryEvent(first + rel, first + rel + far, float(dist[far]), recovery))
    return out


def _empty_stats(seg: np.ndarray, ok: np.ndarray, fps: float, first: int, last: int,
                 params: MovementParams) -> MovementStats:
    duration_s = len(seg) / fps
    levels = list(params.speed_levels_mps)
    stats = MovementStats(
        first_frame_idx=first, last_frame_idx=last, duration_s=duration_s,
        valid_ratio=float(ok.mean()) if len(seg) else 0.0, reliable=False,
        small_sample=duration_s < params.small_sample_s, distance_m=0.0, distance_along_m=0.0,
        distance_across_m=0.0, mean_speed_mps=None, p95_speed_mps=None, speed_levels_mps=levels,
        speed_level_share=[0.0] * (len(levels) + 1), zone_time_s={z: 0.0 for z in ZONE_IDS},
        base_position_m=None, base_source=None)
    stats.reliable = stats.valid_ratio >= params.min_valid_ratio
    return stats


def _fill_motion(stats: MovementStats, seg: np.ndarray, ok: np.ndarray, fps: float, params: MovementParams):
    """Distance, speed statistics and zone times. Returns (step length per frame pair, step is usable)."""
    step = np.diff(seg, axis=0)
    step_ok = ok[1:] & ok[:-1]
    step = np.where(step_ok[:, None], step, 0.0)
    length = np.linalg.norm(step, axis=1)
    stats.distance_m = float(length.sum())
    stats.distance_along_m = float(np.abs(step[:, 0]).sum())
    stats.distance_across_m = float(np.abs(step[:, 1]).sum())
    speed = length[step_ok] * fps
    if len(speed):
        stats.mean_speed_mps = float(speed.mean())
        stats.p95_speed_mps = float(np.percentile(speed, params.speed_percentile))
        counts = np.bincount(np.digitize(speed, stats.speed_levels_mps), minlength=len(stats.speed_levels_mps) + 1)
        stats.speed_level_share = [float(c) / len(speed) for c in counts]
    for zone, n in zone_counts(seg[ok], params.depth_edges_m).items():
        stats.zone_time_s[zone] = n / fps
    return length, step_ok


def _fill_rest(stats: MovementStats, still: np.ndarray, first: int, fps: float, params: MovementParams) -> None:
    for a, b in _runs(still):
        if (b - a + 1) / fps >= params.rest_min_s:
            stats.rest_s += (b - a + 1) / fps
            stats.rest_intervals_frames.append((first + a, first + b))


def _fill_waiting_and_recovery(stats: MovementStats, seg: np.ndarray, still: np.ndarray,
                               hit_frames: Sequence[int], rally_spans: Sequence[Tuple[int, int]], fps: float,
                               params: MovementParams) -> None:
    first, last = stats.first_frame_idx, stats.last_frame_idx
    in_rally = np.zeros(len(seg), bool)
    for a, b in rally_spans:
        in_rally[max(a - first, 0):max(b - first + 1, 0)] = True
    stats.base_position_m, stats.base_source = _waiting_position(seg, still, in_rally, fps, params)
    if stats.base_position_m is None:
        return
    hit_rel = sorted(f - first for f in hit_frames if first <= f <= last)
    stats.recoveries = _recoveries(seg, np.asarray(stats.base_position_m), hit_rel, fps, first, params)
    done = [r.recovery_s for r in stats.recoveries if r.recovery_s is not None]
    stats.mean_recovery_s = float(np.mean(done)) if done else None


def compute_movement(foot_world_m: Sequence[Optional[Sequence[float]]], fps: float, hit_frames: Sequence[int],
                     rally_spans: Sequence[Tuple[int, int]], first_frame_idx: int, last_frame_idx: int,
                     params: MovementParams) -> MovementStats:
    """
    foot_world_m: foot position (x, y) in metres per clip frame, None where the player was not tracked.
    hit_frames: contact frames of the player's own hits.  rally_spans: (start, end) frames of rallies.
    first_frame_idx..last_frame_idx: the stretch that is measured (the clip without its pre-roll).
    """
    seg = _cleaned_segment(foot_world_m, fps, first_frame_idx, last_frame_idx, params)
    ok = ~np.isnan(seg[:, 0])
    stats = _empty_stats(seg, ok, fps, first_frame_idx, last_frame_idx, params)
    if ok.sum() < 2:
        return stats
    length, step_ok = _fill_motion(stats, seg, ok, fps, params)
    still = np.zeros(len(seg), bool)
    still[1:] = step_ok & (length * fps < params.still_speed_mps)
    _fill_rest(stats, still, first_frame_idx, fps, params)
    _fill_waiting_and_recovery(stats, seg, still, hit_frames, rally_spans, fps, params)
    return stats
