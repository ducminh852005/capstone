"""
Zones of the near half-court (the half the calibration covers and the tracked player stays in):
three depth bands (rear / mid / front) times left / right, in the court frame of court_model.py.
"""
from typing import Dict, Sequence, Tuple

import numpy as np

from . import court_model

REAR, MID, FRONT = "rear", "mid", "front"
LEFT, RIGHT = "left", "right"

ZONE_IDS = tuple(f"{depth}_{side}" for depth in (REAR, MID, FRONT) for side in (LEFT, RIGHT))
"""The six zones: rear_left, rear_right, mid_left, ..., front_right."""


def zone_of(world_xy_m: Sequence[float], depth_edges_m: Tuple[float, float],
            center_y_m: float = court_model.CENTER_Y) -> str:
    """
    Zone id of a court point (x, y) in metres. x is the distance from the near baseline: below
    depth_edges_m[0] is the rear, below depth_edges_m[1] the middle, beyond it the front (also past
    the net: only the near half is meaningful). Left / right is seen by a player facing the net
    (y grows to his right); points outside the lines fall in the nearest zone.
    """
    side = LEFT if world_xy_m[1] < center_y_m else RIGHT
    return f"{depth_of(world_xy_m, depth_edges_m)}_{side}"


def depth_of(world_xy_m: Sequence[float], depth_edges_m: Tuple[float, float]) -> str:
    """REAR | MID | FRONT of a court point (see zone_of for the bands)."""
    x = world_xy_m[0]
    return REAR if x < depth_edges_m[0] else MID if x < depth_edges_m[1] else FRONT


def zone_counts(xy_m: np.ndarray, depth_edges_m: Tuple[float, float],
                center_y_m: float = court_model.CENTER_Y) -> Dict[str, int]:
    """Number of points of an (N, 2) array of court positions (m, no NaN) in each zone of ZONE_IDS:
    the same bands as zone_of, for a whole track at once."""
    depth = (xy_m[:, 0] >= depth_edges_m[0]).astype(int) + (xy_m[:, 0] >= depth_edges_m[1])
    index = depth * 2 + (xy_m[:, 1] >= center_y_m)               # ZONE_IDS order: depth major, left before right
    counts = np.bincount(index, minlength=len(ZONE_IDS))
    return {zone: int(n) for zone, n in zip(ZONE_IDS, counts)}
