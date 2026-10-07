"""
Per-player shot statistics of a clip: how many shots of each type, their share, flight time and
speed, where they landed, how often they went out and how many ended a rally. Works on any stretch of
frames (first_frame_idx..), so it can serve whole videos later. Every figure carries its sample size
(n): over a short clip these are small numbers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import court_model
from .hit_events import NEAR
from .rally import Rally
from .shots import SHOT_TYPES, UNKNOWN, Shot
from .umpire import RESULT_OUT
from .zones import ZONE_IDS, zone_of


@dataclass
class ShotTypeStats:
    type: str
    n: int
    share_pct: float                             # of all the player's shots in the stretch
    mean_flight_s: Optional[float]               # over the shots with a known flight time
    mean_speed_body_per_s: Optional[float]       # image-plane speed after the hit, in body heights per second
    mean_ground_speed_mps: Optional[float]       # floor distance / flight time: a lower bound of the real speed
    n_with_landing: int                          # shots that were the last of a rally ended by a landing call
    out_rate: Optional[float]                    # OUT / n_with_landing; None when no shot of the type landed
    landing_zone_counts: Dict[str, int] = field(default_factory=dict)   # zones.ZONE_IDS, from the receiving baseline
    n_rally_winners: int = 0                     # shots that were the last of a rally the player won
    n_rally_errors: int = 0                      # ... or lost


@dataclass
class ShotSummary:
    player_id: int
    n_shots: int
    n_classified: int                            # shots with a type other than "unknown"
    by_type: List[ShotTypeStats]                 # types with at least one shot, in shots.SHOT_TYPES order


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    known = [v for v in values if v is not None]
    return float(np.mean(known)) if known else None


def landing_zone(world_xy_m: Sequence[float], depth_edges_m) -> str:
    """Zone of a landing, measured from the baseline of the half it fell in (the far half is mirrored)."""
    x = world_xy_m[0]
    if x > court_model.NET_X:
        x = court_model.COURT_LENGTH - x
    return zone_of((x, world_xy_m[1]), depth_edges_m)


def summarize_shots(shots: Sequence[Shot], rallies: Sequence[Rally], player_id: int, first_frame_idx: int,
                    depth_edges_m) -> ShotSummary:
    """Statistics of the shots of `player_id` from first_frame_idx on (the clip without its pre-roll)."""
    mine = [s for s in shots if s.hitter == NEAR and s.player_id == player_id and s.frame_idx >= first_frame_idx]
    last_hit = {r.hit_indices[-1]: r for r in rallies if r.hit_indices}
    groups: Dict[str, List[Shot]] = {}
    for s in mine:
        groups.setdefault(s.type, []).append(s)
    by_type = [_type_stats(kind, groups[kind], len(mine), last_hit, depth_edges_m)
               for kind in SHOT_TYPES if kind in groups]
    return ShotSummary(player_id, len(mine), sum(s.type != UNKNOWN for s in mine), by_type)


def _type_stats(kind: str, group: Sequence[Shot], n_all: int, last_hit: Dict[int, Rally], depth_edges_m) -> ShotTypeStats:
    feats = [s.features for s in group if s.features is not None]
    landed = [f for f in feats if f.landing_result is not None]
    zones = {z: 0 for z in ZONE_IDS}
    for f in landed:
        zones[landing_zone(f.landing_world_m, depth_edges_m)] += 1
    rallies_ended = [last_hit[s.hit_index] for s in group if s.hit_index in last_hit]
    return ShotTypeStats(
        type=kind, n=len(group), share_pct=100.0 * len(group) / n_all,
        mean_flight_s=_mean([f.flight_s for f in feats]),
        mean_speed_body_per_s=_mean([f.speed_body_per_s for f in feats]),
        mean_ground_speed_mps=_mean([f.ground_speed_mps for f in feats]),
        n_with_landing=len(landed),
        out_rate=(sum(f.landing_result == RESULT_OUT for f in landed) / len(landed)) if landed else None,
        landing_zone_counts=zones,
        n_rally_winners=sum(r.winner == NEAR for r in rallies_ended),
        n_rally_errors=sum(r.loser == NEAR for r in rallies_ended))
