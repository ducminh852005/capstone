"""
Rule-based shot types of the near player's hits: serve, smash, drop, clear, lift, net shot, drive.

Each hit (core/hit_events.py) gets features from three sources: the racket arm at the contact
(pose: how high the wrist is, how straight the arm), the shuttle (direction and speed right after
the hit, time of flight to the next hit or the landing) and the court (where the player stood, where
the shuttle landed). A fixed list of rules, tried in priority order, picks the type; the first match
wins and the next match is kept as the runner-up.

These are weak rules, not measurements: the pose is 2D and seen from behind, the shuttle's height
is unknown (the floor homography cannot give it) and a shot flying away from the camera rises in the
image even when it is flat. There are no labelled shots yet, so every threshold is an initial guess
(config SHOT_*); the viewer presents the result as a suggestion. Only hits of the tracked near
player are classified: the opponent is not seen, his hits stay "unknown".
"""
import logging
import math
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

from . import config, court_model
from .hit_events import NEAR, UNKNOWN, ShuttleHit
from .pose_features import body_height_px, contact_pose
from .rally import Rally
from .umpire import METHOD_CONTACT, METHOD_UNCONFIRMED, Call
from .zones import FRONT, MID, REAR, depth_of

logger = logging.getLogger(__name__)

SERVE, SMASH, DROP, CLEAR, LIFT, NET, DRIVE = "serve", "smash", "drop", "clear", "lift", "net", "drive"
SHOT_TYPES = (SERVE, SMASH, DROP, CLEAR, LIFT, NET, DRIVE, UNKNOWN)

OVERHEAD, MID_HEIGHT, LOW = "overhead", "mid", "low"      # height of the racket hand at the contact

GROUND_SPEED_METHODS = (METHOD_CONTACT, METHOD_UNCONFIRMED)
"""Landing calls whose position is a seen impact; the others (lost, resting) are too uncertain to
turn into a speed."""


@dataclass
class ShotThresholds:
    """Thresholds of the rules; ShotThresholds.from_config() gives the configured ones."""
    overhead_min_above_head_body: float
    low_min_below_hip_body: float
    serve_max_x_m: float
    smash_min_elbow_deg: float
    smash_min_speed_body_per_s: float
    flat_elevation_deg: float
    drop_max_landing_from_net_m: float
    drop_max_flight_s: float
    clear_min_flight_s: float
    lift_min_flight_s: float
    net_max_flight_s: float
    drive_min_speed_body_per_s: float
    drive_max_flight_s: float
    no_pose_max_conf: float
    base_strength: float
    dead_time_s: float
    depth_edges_m: Tuple[float, float]
    contact_window_frames: int
    min_pose_score: float

    @classmethod
    def from_config(cls) -> "ShotThresholds":
        return cls(config.SHOT_OVERHEAD_MIN_ABOVE_HEAD_BODY, config.SHOT_LOW_MIN_BELOW_HIP_BODY,
                   config.SHOT_SERVE_MAX_X_M, config.SHOT_SMASH_MIN_ELBOW_DEG, config.SHOT_SMASH_MIN_SPEED_BODY_PER_S,
                   config.SHOT_FLAT_ELEVATION_DEG, config.SHOT_DROP_MAX_LANDING_FROM_NET_M,
                   config.SHOT_DROP_MAX_FLIGHT_S, config.SHOT_CLEAR_MIN_FLIGHT_S, config.SHOT_LIFT_MIN_FLIGHT_S,
                   config.SHOT_NET_MAX_FLIGHT_S, config.SHOT_DRIVE_MIN_SPEED_BODY_PER_S,
                   config.SHOT_DRIVE_MAX_FLIGHT_S, config.SHOT_NO_POSE_MAX_CONF, config.SHOT_BASE_STRENGTH,
                   config.RALLY_DEAD_TIME_S, tuple(config.ZONE_DEPTH_EDGES_M), config.POSE_CONTACT_WINDOW_FRAMES,
                   config.POSE_MIN_SCORE)


@dataclass
class ShotFeatures:
    """What the rules look at. None = could not be measured."""
    pose_missing: bool                            # no skeleton near the contact: pose features are None
    contact_height: Optional[str]                 # OVERHEAD | MID_HEIGHT | LOW
    wrist_above_head_body: Optional[float]
    wrist_below_hip_body: Optional[float]
    elbow_angle_deg: Optional[float]
    elevation_deg: float                          # shuttle direction after the hit; > 0 rises (image plane)
    speed_px_per_frame: float
    speed_body_per_s: Optional[float]             # that speed in body heights per second
    flight_s: Optional[float]                     # contact -> next hit or landing; None without either
    is_rally_opening: bool
    hitter_foot_world_m: Optional[List[float]]
    hitter_depth: Optional[str]                   # REAR | MID | FRONT (zones.py)
    landing_world_m: Optional[List[float]] = None
    landing_result: Optional[str] = None          # "IN" | "OUT"
    landing_from_net_m: Optional[float] = None
    ground_dist_m: Optional[float] = None         # hitter's feet -> landing, on the floor
    ground_speed_mps: Optional[float] = None      # that distance / flight_s: a lower bound of the real speed


@dataclass
class Shot:
    hit_index: int                                # ShuttleHit.index
    frame_idx: int                                # contact frame
    hitter: str                                   # NEAR | FAR | UNKNOWN
    player_id: Optional[int]
    type: str                                     # SHOT_TYPES
    confidence: float                             # 0..1
    runner_up: Optional[str]                      # next rule that also matched
    reasons: List[str] = field(default_factory=list)      # keys of the conditions behind `type`
    features: Optional[ShotFeatures] = None


def elevation_deg(angle_deg: float) -> float:
    """Degrees above the horizontal of a direction given as HitMeasurement.angle_deg (image coordinates,
    atan2(dy, dx) with y down: 90 = straight down): positive = rising, +-90 = vertical."""
    return -math.degrees(math.asin(math.sin(math.radians(angle_deg))))


def _flight_s(hit: ShuttleHit, rally: Optional[Rally], hits: Sequence[ShuttleHit], calls: Sequence[Call],
              fps: float) -> Optional[float]:
    """Seconds from the contact to the next event of the rally: the following hit, else the landing."""
    if rally is None:
        return None
    position = rally.hit_indices.index(hit.index)
    if position + 1 < len(rally.hit_indices):
        end = hits[rally.hit_indices[position + 1]].contact_frame_idx
    elif rally.call_index is not None:
        end = calls[rally.call_index].frame_idx
    else:
        return None
    return (end - hit.contact_frame_idx) / fps if end > hit.contact_frame_idx else None


def _nearest_with(players_by_frame, pid: int, contact: int, window: int, need_landmarks: bool):
    """The observation of player `pid` closest in time to `contact` (within `window` frames)."""
    for d in range(window + 1):
        for f in {contact - d, contact + d}:
            obs = players_by_frame[f].get(pid) if 0 <= f < len(players_by_frame) else None
            if obs is not None and (not need_landmarks or obs.landmarks is not None):
                return obs
    return None


def extract_features(hit: ShuttleHit, rally: Optional[Rally], hits: Sequence[ShuttleHit], calls: Sequence[Call],
                     players_by_frame: Sequence[Mapping[int, object]], fps: float, is_opening: bool,
                     t: ShotThresholds) -> ShotFeatures:
    """Features of a near-player hit; see ShotFeatures."""
    skeleton = _nearest_with(players_by_frame, hit.player_id, hit.contact_frame_idx, t.contact_window_frames, True)
    obs = _nearest_with(players_by_frame, hit.player_id, hit.contact_frame_idx, t.contact_window_frames, False)
    pose = None if skeleton is None else contact_pose(skeleton.landmarks, skeleton.bbox, hit.hand, t.min_pose_score)
    height_px = None if obs is None else body_height_px(obs.bbox)

    contact_height = None
    if pose is not None:
        contact_height = (OVERHEAD if pose.wrist_above_head_body >= t.overhead_min_above_head_body
                          else LOW if pose.wrist_below_hip_body >= t.low_min_below_hip_body else MID_HEIGHT)
    foot = None if obs is None or obs.foot_world is None else [float(v) for v in obs.foot_world]
    feats = ShotFeatures(
        pose_missing=pose is None, contact_height=contact_height,
        wrist_above_head_body=None if pose is None else pose.wrist_above_head_body,
        wrist_below_hip_body=None if pose is None else pose.wrist_below_hip_body,
        elbow_angle_deg=None if pose is None else pose.elbow_angle_deg,
        elevation_deg=elevation_deg(hit.angle_deg), speed_px_per_frame=hit.speed_px_per_frame,
        speed_body_per_s=None if not height_px else hit.speed_px_per_frame * fps / height_px,
        flight_s=_flight_s(hit, rally, hits, calls, fps), is_rally_opening=is_opening,
        hitter_foot_world_m=foot, hitter_depth=None if foot is None else depth_of(foot, t.depth_edges_m))

    last_of_rally = rally is not None and rally.hit_indices[-1] == hit.index and rally.call_index is not None
    if last_of_rally:
        call = calls[rally.call_index]
        feats.landing_world_m = [float(v) for v in call.world_pt]
        feats.landing_result = call.result
        feats.landing_from_net_m = abs(float(call.world_pt[0]) - court_model.NET_X)
        if foot is not None and call.method in GROUND_SPEED_METHODS:
            feats.ground_dist_m = math.hypot(call.world_pt[0] - foot[0], call.world_pt[1] - foot[1])
            if feats.flight_s:
                feats.ground_speed_mps = feats.ground_dist_m / feats.flight_s
    return feats


def _rule(required: Sequence[Tuple[str, Optional[bool]]], bonus: Sequence[Tuple[str, Optional[bool]]],
          base_strength: float) -> Optional[Tuple[List[str], float]]:
    """(reasons, strength) when every required condition holds, else None. A condition that is None
    could not be evaluated: a required one is skipped, a bonus one does not count either way."""
    if any(ok is False for _, ok in required):
        return None
    reasons = [key for key, ok in required if ok]
    known = [(key, ok) for key, ok in bonus if ok is not None]
    reasons += [key for key, ok in known if ok]
    if not known:
        return reasons, base_strength
    return reasons, base_strength + (1.0 - base_strength) * sum(ok for _, ok in known) / len(known)


def match_rules(f: ShotFeatures, t: ShotThresholds) -> List[Tuple[str, List[str], float]]:
    """Every rule that matches, in priority order: (type, reasons, strength). A rule whose evidence
    is missing (no flight time and no landing for a drop, ...) does not match."""
    c = f.contact_height
    overhead = None if c is None else c == OVERHEAD
    low = None if c is None else c == LOW
    mid = None if c is None else c == MID_HEIGHT
    rises = f.elevation_deg > t.flat_elevation_deg
    falls = f.elevation_deg < -t.flat_elevation_deg
    flat = not rises and not falls
    fast = None if f.speed_body_per_s is None else f.speed_body_per_s >= t.smash_min_speed_body_per_s
    drive_fast = None if f.speed_body_per_s is None else f.speed_body_per_s >= t.drive_min_speed_body_per_s
    back = None if f.hitter_depth is None else f.hitter_depth in (REAR, MID)
    front = None if f.hitter_depth is None else f.hitter_depth == FRONT
    fl = f.flight_s
    short_landing = None if f.landing_from_net_m is None else f.landing_from_net_m <= t.drop_max_landing_from_net_m
    elbow_out = None if f.elbow_angle_deg is None else f.elbow_angle_deg >= t.smash_min_elbow_deg
    behind_line = None if f.hitter_foot_world_m is None else f.hitter_foot_world_m[0] <= t.serve_max_x_m
    out = []

    def add(kind, result):
        if result is not None:
            out.append((kind, *result))

    add(SERVE, _rule([("rally_opening", f.is_rally_opening or False), ("contact_low", low), ("behind_service_line", behind_line)],
                     [], t.base_strength))
    add(SMASH, _rule([("contact_overhead", overhead), ("shuttle_fast", fast), ("not_rising", not rises), ("hit_from_back", back)],
                     [("arm_extended", elbow_out), ("shuttle_falling", falls)], t.base_strength))
    add(DROP, _rule([("contact_overhead", overhead), ("shuttle_slow", None if fast is None else not fast),
                     ("hit_from_back", back),
                     ("short_flight_or_landing", any((short_landing, None if fl is None else fl <= t.drop_max_flight_s)))],
                    [("not_rising", not rises)], t.base_strength))
    add(CLEAR, _rule([("contact_overhead", overhead), ("hit_from_back", back),
                      ("rising_or_long_flight", any((rises, None if fl is None else fl >= t.clear_min_flight_s)))],
                     [("long_flight", None if fl is None else fl >= t.clear_min_flight_s), ("shuttle_rising", rises)],
                     t.base_strength))
    add(LIFT, _rule([("contact_low", low),
                     ("rising_or_long_flight", any((rises, None if fl is None else fl >= t.lift_min_flight_s)))],
                    [("long_flight", None if fl is None else fl >= t.lift_min_flight_s), ("shuttle_rising", rises)],
                    t.base_strength))
    add(NET, _rule([("contact_low_or_mid", None if c is None else c != OVERHEAD), ("hit_at_net", front),
                    ("shuttle_slow", None if drive_fast is None else not drive_fast)],
                   [("short_flight", None if fl is None else fl <= t.net_max_flight_s), ("not_rising", not rises)],
                   t.base_strength))
    add(DRIVE, _rule([("contact_mid", mid), ("shuttle_flat", flat), ("shuttle_fast", drive_fast)],
                     [("short_flight", None if fl is None else fl <= t.drive_max_flight_s)], t.base_strength))
    return out


def classify_shot(hit: ShuttleHit, features: ShotFeatures, t: ShotThresholds) -> Shot:
    """The shot type of a near-player hit from its features (first matching rule wins)."""
    matches = match_rules(features, t)
    shot = Shot(hit.index, hit.contact_frame_idx, hit.hitter, hit.player_id, UNKNOWN, 0.0, None, ["no_rule_matches"],
                features)
    if not matches:
        return shot
    kind, reasons, strength = matches[0]
    confidence = hit.confidence * strength
    if features.pose_missing:
        confidence = min(confidence, t.no_pose_max_conf)
        reasons = reasons + ["pose_missing"]
    shot.type, shot.reasons, shot.confidence = kind, reasons, confidence
    shot.runner_up = matches[1][0] if len(matches) > 1 else None
    return shot


def build_shots(hits: Sequence[ShuttleHit], rallies: Sequence[Rally], calls: Sequence[Call],
                players_by_frame: Sequence[Mapping[int, object]], fps: float, t: ShotThresholds) -> List[Shot]:
    """One Shot per valid hit, in order. Hits of the opponent or of an unidentified player are
    "unknown": they are not seen, so there is no pose to classify from."""
    rally_of = {i: r for r in rallies for i in r.hit_indices}
    shots: List[Shot] = []
    previous_end: Optional[int] = None
    opening_rallies = set()
    for r in rallies:                     # a rally opens with a serve when play had stopped for a while before it
        if previous_end is None or (r.start_frame_idx - previous_end) / fps >= t.dead_time_s:
            opening_rallies.add(r.index)
        previous_end = r.end_frame_idx
    for hit in hits:
        if not hit.valid:
            continue
        if hit.hitter != NEAR or hit.player_id is None:
            reason = "opponent_not_tracked" if hit.hitter != UNKNOWN else "hitter_unknown"
            shots.append(Shot(hit.index, hit.contact_frame_idx, hit.hitter, hit.player_id, UNKNOWN, 0.0, None, [reason]))
            continue
        rally = rally_of.get(hit.index)
        opening = rally is not None and rally.index in opening_rallies and rally.hit_indices[0] == hit.index
        features = extract_features(hit, rally, hits, calls, players_by_frame, fps, opening, t)
        shots.append(classify_shot(hit, features, t))
    return shots
