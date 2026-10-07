"""
Racket hits of a clip: which player hit the shuttle, and when.

The shuttle tracker only reports that a track (re)started (core/smash.py HitMeasurement). That is
not a clean list of hits: a floor bounce restarts the track just like a racket does, and a shuttle
seen for the first time ("new") may simply be falling. A start is attributed to a tracked
near-half player when one of his hand points (wrist, index fingertip; core/pose_features.py) comes
close to the shuttle's path across the contact. Starts with no hand nearby are the opponent's hits,
inferred (the opponent is not tracked: the calibration covers only the near half), or rejected
when they are more likely a bounce or a shuttle first seen in flight.

Frames are clip-local; pixel quantities carry a _px suffix, distances relative to the player's size
a _body suffix (pixels / height of his box).
"""
import logging
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import config
from .pose_features import LEFT, RIGHT, body_height_px, hand_points, infer_racket_hand
from .smash import HitMeasurement
from .umpire import SINGLES, Call

logger = logging.getLogger(__name__)

NEAR = "near"          # the tracked near-half player
FAR = "far"            # the opponent (inferred, never seen)
UNKNOWN = "unknown"

EVIDENCE_WRIST = "wrist"                  # a hand point of a tracked player was near the shuttle
EVIDENCE_NO_NEAR_HAND = "no_near_hand"    # hands were measured and none was near: someone else hit it
EVIDENCE_NONE = "none"                    # no hand could be measured (no player or no skeleton)

REJECT_NEW_NO_HITTER = "new_track_no_hitter"
REJECT_LANDING_BOUNCE = "landing_bounce"

HIT_SOURCES = ("physics", "new")
"""Track starts that can be a hit. A "stop" (the shuttle came to rest) never is."""


@dataclass
class HitterParams:
    """Thresholds of the attribution; HitterParams.from_config() gives the configured ones."""
    max_wrist_dist_body: float
    min_near_confidence: float
    window_margin_frames: int
    default_window_frames: int
    max_prev_gap_frames: int
    landing_exclude_frames: int
    far_confidence: float
    far_break_confidence: float
    min_pose_score: float
    hand: Optional[str] = None        # LEFT / RIGHT: only this hand counts; None = either hand

    @classmethod
    def from_config(cls, hand: Optional[str] = None) -> "HitterParams":
        return cls(config.HITTER_MAX_WRIST_DIST_BODY, config.HITTER_MIN_NEAR_CONFIDENCE,
                   config.HITTER_WINDOW_MARGIN_FRAMES, config.HITTER_DEFAULT_WINDOW_FRAMES,
                   config.HITTER_MAX_PREV_GAP_FRAMES, config.HITTER_LANDING_EXCLUDE_FRAMES,
                   config.HITTER_FAR_CONFIDENCE, config.HITTER_FAR_BREAK_CONFIDENCE,
                   config.POSE_MIN_SCORE, hand)


@dataclass
class ShuttleHit:
    index: int                                   # position in the attributed list (also its id in rallies)
    contact_frame_idx: int                       # best estimate of the racket contact
    start_frame_idx: int                         # frame the shuttle track started on (first detection after it)
    window_frames: Tuple[int, int]               # frames searched for the hand (inclusive)
    shuttle_pt_px: Tuple[float, float]           # shuttle at the track start
    prev_pt_px: Optional[Tuple[float, float]]    # last shuttle detection before the hit
    source: str                                  # "physics" | "new"
    speed_px_per_frame: float
    angle_deg: float                             # image coordinates, 90 = straight down
    is_smash: bool
    dt_frames: int                               # frames between the two detections the speed was measured on
    hitter: str                                  # NEAR | FAR | UNKNOWN
    evidence: str
    confidence: float                            # 0..1
    low_confidence: bool
    valid: bool
    player_id: Optional[int] = None
    hand: Optional[str] = None                   # hand point that came closest (NEAR only)
    wrist_dist_body: Optional[float] = None      # that distance, in body heights
    reject_reason: Optional[str] = None
    alternation_break: bool = False              # two hits of the near player in a row inside one rally


def point_segment_distance(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    """Distance (same unit as the inputs) from point p to the segment a-b. Plain float arithmetic: this
    runs for every hand point of every frame of every contact window."""
    ax, ay = a[0], a[1]
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _within(sorted_frames: Sequence[int], frame: int, tol: int) -> bool:
    """Whether a frame of the sorted list lies within +-tol of `frame`."""
    i = bisect_left(sorted_frames, frame - tol)
    return i < len(sorted_frames) and sorted_frames[i] <= frame + tol


def _any_between(sorted_frames: Sequence[int], low_exclusive: int, high_inclusive: int) -> bool:
    i = bisect_right(sorted_frames, low_exclusive)
    return i < len(sorted_frames) and sorted_frames[i] <= high_inclusive


def previous_detection(shuttle_px: Sequence[Optional[Sequence[float]]], start_frame: int,
                       start_pt: Sequence[float], max_gap_frames: int
                       ) -> Optional[Tuple[int, Tuple[float, float]]]:
    """
    (frame, point) of the last shuttle detection before `start_frame`, looking back at most
    `max_gap_frames`. The two frames before a track start hold copies of the start point (the
    tracker back-fills them when it starts a track), so entries equal to it are skipped.
    """
    last = max(start_frame - max_gap_frames, 0)
    for f in range(min(start_frame, len(shuttle_px)) - 1, last - 1, -1):
        pt = shuttle_px[f]
        if pt is None or (pt[0], pt[1]) == (start_pt[0], start_pt[1]):
            continue
        return f, (float(pt[0]), float(pt[1]))
    return None


def _closest_hand(players_by_frame: Sequence[Mapping[int, object]], first: int, last: int,
                  a: Sequence[float], b: Sequence[float], params: HitterParams):
    """
    Closest hand point to the segment a-b over frames first..last: ((dist_body, frame, player_id,
    hand), measured). `measured` tells whether any hand point was visible at all.
    """
    sides = (LEFT, RIGHT) if params.hand is None else (params.hand,)
    best, measured = None, False
    for t in range(first, last + 1):
        for pid, obs in players_by_frame[t].items():
            height = body_height_px(obs.bbox)
            if height <= 0:
                continue
            for side in sides:
                for pt in hand_points(obs.landmarks, side, params.min_pose_score):
                    measured = True
                    dist = point_segment_distance(pt, a, b) / height
                    if best is None or dist < best[0]:
                        best = (dist, t, pid, side)
    return best, measured


def _contact_window(m: HitMeasurement, shuttle_px, n_frames: int, params: HitterParams):
    """(prev_pt, raw_first, first, last): the last detection before the start, the frame the contact
    window opens at (before the margin) and its frames first..last, clipped to the clip."""
    prev = previous_detection(shuttle_px, m.frame_idx, m.pt, params.max_prev_gap_frames)
    prev_frame, prev_pt = prev if prev is not None else (None, None)
    raw_first = prev_frame if prev_frame is not None else m.frame_idx - params.default_window_frames
    return (prev_pt, max(raw_first, 0), max(raw_first - params.window_margin_frames, 0),
            min(m.frame_idx + params.window_margin_frames, n_frames - 1))


def _grade_hit(hit: ShuttleHit, best, measured: bool, params: HitterParams) -> None:
    """Who hit, from the closest hand point `best` = (dist_body, frame, player_id, hand) (None without
    any) and whether any hand could be measured at all."""
    if best is not None and best[0] <= params.max_wrist_dist_body:
        dist, frame, pid, side = best
        share = 1.0 - dist / params.max_wrist_dist_body
        hit.hitter, hit.evidence, hit.player_id, hit.hand = NEAR, EVIDENCE_WRIST, pid, side
        hit.wrist_dist_body, hit.contact_frame_idx = dist, frame
        hit.confidence = params.min_near_confidence + (1.0 - params.min_near_confidence) * share
        hit.low_confidence = False
    elif measured:
        hit.hitter, hit.evidence = FAR, EVIDENCE_NO_NEAR_HAND
        hit.wrist_dist_body = best[0]
        hit.confidence = params.far_break_confidence


def _reject_reason(hit: ShuttleHit, call_frames: Sequence[int], params: HitterParams) -> Optional[str]:
    """Why a start without a hand near it is not a hit: a shuttle first seen in flight, or a floor bounce."""
    if hit.hitter == NEAR:
        return None
    if hit.source == "new":
        return REJECT_NEW_NO_HITTER
    if _within(call_frames, hit.start_frame_idx, params.landing_exclude_frames):
        return REJECT_LANDING_BOUNCE
    return None


def attribute_hits(measurements: Sequence[HitMeasurement], shuttle_px: Sequence[Optional[Sequence[float]]],
                   players_by_frame: Sequence[Mapping[int, object]], calls: Sequence[Call],
                   params: HitterParams) -> List[ShuttleHit]:
    """
    One ShuttleHit per track start that can be a hit (HIT_SOURCES), in order of start frame.

    shuttle_px: shuttle detection per clip frame (None where there is none).
    players_by_frame: {player_id: PlayerObs} per clip frame (needs .bbox and .landmarks).
    calls: landing calls, used to tell a floor bounce from a hit.
    Rejected starts stay in the list with valid=False and a reject_reason. Opponent hits are
    only inferred here (hitter FAR, low confidence); apply_alternation() refines their confidence.
    """
    call_frames = sorted(c.frame_idx for c in calls)
    hits: List[ShuttleHit] = []
    for m in sorted(measurements, key=lambda m: m.frame_idx):
        if m.source not in HIT_SOURCES:
            continue
        prev_pt, raw_first, first, last = _contact_window(m, shuttle_px, len(players_by_frame), params)
        best, measured = _closest_hand(players_by_frame, first, last, prev_pt if prev_pt is not None else m.pt,
                                       m.pt, params)
        hit = ShuttleHit(
            index=len(hits), contact_frame_idx=(raw_first + m.frame_idx) // 2, start_frame_idx=m.frame_idx,
            window_frames=(first, last), shuttle_pt_px=(float(m.pt[0]), float(m.pt[1])), prev_pt_px=prev_pt,
            source=m.source, speed_px_per_frame=float(m.speed), angle_deg=float(m.angle_deg),
            is_smash=bool(m.is_smash), dt_frames=int(m.dt_frames),
            hitter=UNKNOWN, evidence=EVIDENCE_NONE, confidence=0.0, low_confidence=True, valid=True)
        _grade_hit(hit, best, measured, params)
        reason = _reject_reason(hit, call_frames, params)
        if reason is not None:
            hit.valid, hit.reject_reason = False, reason
        hits.append(hit)
        logger.debug("Hit candidate at frame %d (%s): %s, %s, valid=%s", m.frame_idx, m.source, hit.hitter,
                     hit.evidence, hit.valid)
    return hits


def apply_alternation(hits: Sequence[ShuttleHit], calls: Sequence[Call], match_type: str,
                      params: HitterParams) -> None:
    """
    Use the alternation of a singles rally (near, far, near, ...) to grade the inferred hits, in place.

    A FAR hit right after a near hit is what a rally looks like (HITTER_FAR_CONFIDENCE); one that
    follows another FAR hit, or opens a rally, is less certain (HITTER_FAR_BREAK_CONFIDENCE). Two
    NEAR hits in a row get `alternation_break` (the opponent's hit in between was missed, or one
    of the two is wrong). A landing call starts a new rally. In doubles the opponents cannot be told
    apart from the near player's partner, so inferred hits become UNKNOWN.
    """
    call_frames = sorted(c.frame_idx for c in calls)
    prev: Optional[ShuttleHit] = None
    warned = False
    for hit in sorted((h for h in hits if h.valid), key=lambda h: h.contact_frame_idx):
        if prev is not None and _any_between(call_frames, prev.contact_frame_idx, hit.contact_frame_idx):
            prev = None
        if hit.hitter == FAR:
            if match_type != SINGLES:
                hit.hitter, hit.confidence = UNKNOWN, 0.0
                if not warned:
                    logger.warning("Hits of the opponent cannot be inferred in %s matches; they are left unknown",
                                   match_type)
                    warned = True
            elif prev is not None and prev.hitter == NEAR:
                hit.confidence = params.far_confidence
            else:
                hit.confidence = params.far_break_confidence
                hit.alternation_break = prev is not None and prev.hitter == FAR
        elif hit.hitter == NEAR:
            hit.alternation_break = prev is not None and prev.hitter == NEAR
        prev = hit


def racket_hands(hits: Sequence[ShuttleHit]) -> Dict[int, Tuple[Optional[str], float, int]]:
    """{player_id: (racket hand, share of the votes, number of votes)}: each valid near hit votes
    for the hand that came closest to the shuttle."""
    votes: Dict[int, List[str]] = {}
    for h in hits:
        if h.valid and h.hitter == NEAR and h.player_id is not None and h.hand is not None:
            votes.setdefault(h.player_id, []).append(h.hand)
    return {pid: (*infer_racket_hand(v), len(v)) for pid, v in votes.items()}
