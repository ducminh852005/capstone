"""
Rallies of a clip: the hits between two landings, who hit last, and who won the point.

A rally ends at a landing call (core/umpire.py), when no hit follows for RALLY_MAX_HIT_GAP_S
(shuttle dead, call missed) or at the end of the clip. With the last hitter L and the half H the
shuttle landed in:
  - landed in the other half, IN   -> L wins   (landed_in)
  - landed in the other half, OUT  -> the opponent wins (landed_out)
  - landed in L's own half         -> L loses  (own_half): a net fault, or as likely an opponent hit
                                      that was not detected, so it carries a low confidence.
Net touches and shuttles hitting the net are not detected, so a rally lost into the net ends in
a timeout without a winner.

Frames are clip-local. Hits are the ShuttleHit list of core/hit_events.py.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from . import config
from .hit_events import FAR, NEAR, ShuttleHit
from .umpire import METHOD_RESTING, RESULT_IN, Call

logger = logging.getLogger(__name__)

END_LANDED_IN = "landed_in"
END_LANDED_OUT = "landed_out"
END_OWN_HALF = "own_half"
END_TIMEOUT = "timeout"
END_CLIP = "clip_end"

_HIT, _CALL = 0, 1      # sweep order at equal frames: the hit comes first


@dataclass
class RallyParams:
    """Thresholds of the segmentation; RallyParams.from_config() gives the configured ones."""
    max_hit_gap_s: float
    close_call_factor: float
    resting_factor: float
    own_half_factor: float
    inferred_hitter_confidence: float

    @classmethod
    def from_config(cls) -> "RallyParams":
        return cls(config.RALLY_MAX_HIT_GAP_S, config.RALLY_CLOSE_CALL_FACTOR, config.RALLY_RESTING_FACTOR,
                   config.RALLY_OWN_HALF_FACTOR, config.RALLY_INFERRED_HITTER_CONFIDENCE)


@dataclass
class Rally:
    index: int
    start_frame_idx: int                 # first hit's contact frame (the call's frame for a rally without hits)
    end_frame_idx: int                   # the landing call's frame, else the last hit's contact frame
    hit_indices: List[int]               # ShuttleHit.index of its valid hits, in order
    n_hits: int
    last_hitter: str                     # NEAR | FAR | UNKNOWN
    call_index: Optional[int]            # index into the `calls` list given to segment_rallies
    end_cause: str                       # END_*
    winner: Optional[str]                # NEAR | FAR | None
    loser: Optional[str]
    winner_confidence: float             # 0..1, 0 when there is no winner
    in_preroll: bool                     # started before the first counted frame (pre-roll of the clip)


def _opposite(side: str) -> str:
    return FAR if side == NEAR else NEAR


def _decide(last_hitter: str, hitter_confidence: float, call: Call, params: RallyParams
            ) -> Tuple[str, Optional[str], float]:
    """(end_cause, winner, confidence) of a rally whose last hitter is `last_hitter` and that ended in `call`."""
    cause = END_LANDED_IN if call.result == RESULT_IN else END_LANDED_OUT
    if last_hitter not in (NEAR, FAR):
        return cause, None, 0.0
    confidence = hitter_confidence
    if call.half == last_hitter:
        cause, winner = END_OWN_HALF, _opposite(last_hitter)
        confidence *= params.own_half_factor
    elif call.result == RESULT_IN:
        winner = last_hitter
    else:
        winner = _opposite(last_hitter)
    if call.close_call:
        confidence *= params.close_call_factor
    if call.method == METHOD_RESTING:
        confidence *= params.resting_factor
    return cause, winner, max(0.0, min(1.0, confidence))


def _finish(index: int, hits: Sequence[ShuttleHit], call: Optional[Tuple[int, Call]], no_call_cause: str,
            params: RallyParams, analysis_start_frame: int) -> Rally:
    if call is None:
        cause, winner, confidence = no_call_cause, None, 0.0
        last_hitter = hits[-1].hitter
        start, end = hits[0].contact_frame_idx, hits[-1].contact_frame_idx
    else:
        call_index, c = call
        if hits:
            last_hitter, hitter_confidence = hits[-1].hitter, hits[-1].confidence
            start = hits[0].contact_frame_idx
        else:
            # no hit was detected: the shuttle must have crossed the net to land where it did
            last_hitter, hitter_confidence, start = _opposite(c.half), params.inferred_hitter_confidence, c.frame_idx
        cause, winner, confidence = _decide(last_hitter, hitter_confidence, c, params)
        end = c.frame_idx
    return Rally(index=index, start_frame_idx=start, end_frame_idx=end, hit_indices=[h.index for h in hits],
                 n_hits=len(hits), last_hitter=last_hitter, call_index=None if call is None else call[0],
                 end_cause=cause, winner=winner, loser=None if winner is None else _opposite(winner),
                 winner_confidence=confidence, in_preroll=start < analysis_start_frame)


def segment_rallies(hits: Sequence[ShuttleHit], calls: Sequence[Call], fps: float, params: RallyParams,
                    analysis_start_frame: int = 0, last_frame_idx: Optional[int] = None) -> List[Rally]:
    """
    Rallies of the clip in order. Only valid hits count. `calls` are the landing calls (any order;
    Rally.call_index refers to this list). A rally that began before `analysis_start_frame` is flagged
    in_preroll. last_frame_idx is the final clip frame: a rally whose last hit is within the maximum hit gap of
    it is cut by the end of the clip (clip_end), otherwise it timed out.
    """
    events = [(h.contact_frame_idx, _HIT, h.index, h) for h in hits if h.valid]
    events += [(c.frame_idx, _CALL, j, c) for j, c in enumerate(calls)]
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    rallies: List[Rally] = []
    current: List[ShuttleHit] = []

    def silent(frame: int) -> bool:
        return bool(current) and (frame - current[-1].contact_frame_idx) / fps > params.max_hit_gap_s

    def close(call=None, cause=END_TIMEOUT):
        rallies.append(_finish(len(rallies), current, call, cause, params, analysis_start_frame))
        current.clear()

    for frame, kind, idx, obj in events:
        if silent(frame):
            close()
        if kind == _HIT:
            current.append(obj)
        else:
            close(call=(idx, obj))
    if current:
        cut = last_frame_idx is not None and (last_frame_idx - current[-1].contact_frame_idx) / fps <= params.max_hit_gap_s
        close(cause=END_CLIP if cut else END_TIMEOUT)
    return rallies
