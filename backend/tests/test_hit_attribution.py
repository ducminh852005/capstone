from types import SimpleNamespace

import numpy as np
import pytest

from core.hit_events import (EVIDENCE_NO_NEAR_HAND, EVIDENCE_NONE, EVIDENCE_WRIST, FAR, NEAR, UNKNOWN,
                             HitterParams, ShuttleHit, apply_alternation, attribute_hits, point_segment_distance,
                             previous_detection, racket_hands)
from core.pose_features import LEFT, LEFT_WRIST, RIGHT, RIGHT_WRIST
from core.smash import HitMeasurement
from core.umpire import DOUBLES, SINGLES, Call

BODY_PX = 200.0        # box height of the test players
N = 60


def params(**kw):
    base = dict(max_wrist_dist_body=0.5, min_near_confidence=0.5, window_margin_frames=2, default_window_frames=5,
                max_prev_gap_frames=15, landing_exclude_frames=8, far_confidence=0.3, far_break_confidence=0.2,
                min_pose_score=0.5, hand=None)
    base.update(kw)
    return HitterParams(**base)


def obs(right=None, left=None, with_pose=True):
    """A player observation with a 200 px tall box; hands at the given image points."""
    lm = np.zeros((33, 3), np.float32)
    if right is not None:
        lm[RIGHT_WRIST] = (*right, 0.9)
    if left is not None:
        lm[LEFT_WRIST] = (*left, 0.9)
    return SimpleNamespace(bbox=(0.0, 0.0, 100.0, BODY_PX), landmarks=lm if with_pose else None)


def frames(per_frame=None, default=None):
    """players_by_frame: N frames of {1: default}, with `per_frame` {frame: observation} overrides."""
    out = [({1: default} if default is not None else {}) for _ in range(N)]
    for f, o in (per_frame or {}).items():
        out[f] = {1: o}
    return out


def measurement(frame, pt, source="physics", speed=10.0):
    return HitMeasurement(frame, pt, speed, 45.0, False, 0, source, 5)


def shuttle(points):
    """Shuttle detection per frame from {frame: (x, y)}."""
    px = [None] * N
    for f, v in points.items():
        px[f] = v
    return px


def call(frame, half="far", result="IN"):
    return Call(frame, result, half, (0, 0), (0.0, 0.0), 0.5, False)


def hit(index, frame, hitter, **kw):
    base = dict(index=index, contact_frame_idx=frame, start_frame_idx=frame, window_frames=(frame, frame),
                shuttle_pt_px=(0.0, 0.0), prev_pt_px=None, source="physics", speed_px_per_frame=1.0, angle_deg=0.0,
                is_smash=False, dt_frames=5, hitter=hitter, evidence=EVIDENCE_WRIST, confidence=0.9,
                low_confidence=False, valid=True)
    base.update(kw)
    return ShuttleHit(**base)


# ---- geometry ---------------------------------------------------------------------------------

def test_point_segment_distance_perpendicular_beyond_the_ends_and_degenerate():
    assert point_segment_distance((5, 3), (0, 0), (10, 0)) == 3.0
    assert point_segment_distance((-3, 4), (0, 0), (10, 0)) == 5.0
    assert point_segment_distance((13, 4), (0, 0), (10, 0)) == 5.0
    assert point_segment_distance((3, 4), (0, 0), (0, 0)) == 5.0


def test_previous_detection_skips_the_copies_the_tracker_back_fills_before_a_start():
    px = shuttle({10: (100, 100), 13: (200, 200), 14: (200, 200)})
    assert previous_detection(px, 15, (200, 200), max_gap_frames=15) == (10, (100.0, 100.0))
    assert previous_detection(px, 15, (200, 200), max_gap_frames=3) is None       # too far back
    assert previous_detection(shuttle({13: (200, 200)}), 15, (200, 200), 15) is None


# ---- attribution ------------------------------------------------------------------------------

def test_a_hand_on_the_shuttle_path_makes_it_a_near_hit_at_the_frame_of_the_closest_approach():
    px = shuttle({10: (100, 100), 15: (200, 200)})
    players = frames({12: obs(right=(100, 160)), 13: obs(right=(150, 150)), 14: obs(right=(170, 120))},
                     default=obs(right=(400, 400)))
    [h] = attribute_hits([measurement(15, (200, 200))], px, players, [], params())
    assert (h.hitter, h.evidence, h.player_id, h.hand) == (NEAR, EVIDENCE_WRIST, 1, RIGHT)
    assert h.contact_frame_idx == 13 and h.wrist_dist_body == pytest.approx(0.0, abs=1e-6)
    assert h.confidence == pytest.approx(1.0) and not h.low_confidence and h.valid
    assert h.prev_pt_px == (100.0, 100.0) and h.start_frame_idx == 15
    assert h.window_frames == (8, 17)                   # prev detection - 2 .. start + 2


def test_confidence_falls_to_the_minimum_at_the_distance_limit():
    px = shuttle({10: (100, 100), 15: (100, 100)})
    players = frames(default=obs(right=(100 + 0.5 * BODY_PX, 100)))      # exactly the limit away
    [h] = attribute_hits([measurement(15, (100, 100))], px, players, [], params())
    assert h.hitter == NEAR and h.confidence == pytest.approx(0.5)


def test_hands_measured_but_none_near_means_the_opponent_hit_it():
    px = shuttle({10: (100, 100), 15: (200, 200)})
    players = frames(default=obs(right=(600, 600), left=(500, 500)))
    [h] = attribute_hits([measurement(15, (200, 200))], px, players, [], params())
    assert (h.hitter, h.evidence, h.valid, h.player_id) == (FAR, EVIDENCE_NO_NEAR_HAND, True, None)
    assert h.low_confidence and h.confidence == 0.2 and h.wrist_dist_body > 0.5


def test_without_any_visible_hand_the_hitter_is_unknown_not_far():
    px = shuttle({10: (100, 100), 15: (200, 200)})
    [h] = attribute_hits([measurement(15, (200, 200))], px, frames(default=obs(with_pose=False)), [], params())
    assert (h.hitter, h.evidence, h.confidence, h.valid) == (UNKNOWN, EVIDENCE_NONE, 0.0, True)
    [h] = attribute_hits([measurement(15, (200, 200))], px, frames(), [], params())    # nobody tracked
    assert h.hitter == UNKNOWN


def test_the_closest_of_two_players_gets_the_hit():
    px = shuttle({10: (100, 100), 15: (200, 200)})
    near_hand, far_hand = obs(right=(150, 150)), obs(right=(400, 400))
    players = [{1: far_hand, 2: near_hand} for _ in range(N)]
    [h] = attribute_hits([measurement(15, (200, 200))], px, players, [], params())
    assert h.player_id == 2


def test_a_forced_racket_hand_ignores_the_other_hand():
    px = shuttle({10: (100, 100), 15: (200, 200)})
    players = frames(default=obs(right=(150, 150), left=(900, 900)))
    [h] = attribute_hits([measurement(15, (200, 200))], px, players, [], params(hand=LEFT))
    assert h.hitter == FAR                                  # only the left hand counts and it is far away
    [h] = attribute_hits([measurement(15, (200, 200))], px, players, [], params(hand=RIGHT))
    assert h.hitter == NEAR and h.hand == RIGHT


def test_a_new_track_without_a_hand_near_is_not_a_hit_but_with_one_it_is():
    px = shuttle({15: (200, 200)})
    [h] = attribute_hits([measurement(15, (200, 200), source="new")], px, frames(default=obs(right=(900, 900))), [],
                         params())
    assert (h.valid, h.reject_reason) == (False, "new_track_no_hitter")
    [h] = attribute_hits([measurement(15, (200, 200), source="new")], px, frames(default=obs(right=(200, 200))), [],
                         params())
    assert h.valid and h.hitter == NEAR


def test_a_restart_next_to_a_landing_without_a_hand_is_a_bounce_but_a_hand_keeps_it():
    px = shuttle({45: (100, 100), 50: (200, 200)})
    away = frames(default=obs(right=(900, 900)))
    [h] = attribute_hits([measurement(50, (200, 200))], px, away, [call(47)], params())
    assert (h.valid, h.reject_reason) == (False, "landing_bounce")
    [h] = attribute_hits([measurement(50, (200, 200))], px, away, [call(20)], params())      # a landing far away in time
    assert h.valid
    [h] = attribute_hits([measurement(50, (200, 200))], px, frames(default=obs(right=(150, 150))), [call(47)], params())
    assert h.valid and h.hitter == NEAR


def test_track_stops_are_not_hits_and_hits_come_back_in_order():
    px = shuttle({10: (1, 1), 20: (2, 2), 30: (3, 3)})
    ms = [measurement(30, (3, 3)), measurement(20, (2, 2), source="stop"), measurement(10, (1, 1))]
    hits = attribute_hits(ms, px, frames(default=obs(right=(1, 1))), [], params())
    assert [h.start_frame_idx for h in hits] == [10, 30] and [h.index for h in hits] == [0, 1]


def test_the_hit_keeps_the_speed_and_direction_of_its_measurement():
    px = shuttle({10: (100, 100), 15: (200, 200)})
    m = HitMeasurement(15, (200, 200), 21.5, 80.0, True, 1, "physics", 5)
    [h] = attribute_hits([m], px, frames(default=obs(right=(150, 150))), [], params())
    assert (h.speed_px_per_frame, h.angle_deg, h.is_smash, h.dt_frames) == (21.5, 80.0, True, 5)


def test_without_an_earlier_detection_the_default_window_is_searched():
    px = shuttle({15: (200, 200)})
    players = frames({11: obs(right=(200, 200))}, default=obs(right=(900, 900)))   # hand there 4 frames before the start
    [h] = attribute_hits([measurement(15, (200, 200))], px, players, [], params())
    assert h.hitter == NEAR and h.contact_frame_idx == 11 and h.prev_pt_px is None
    assert h.window_frames == (8, 17)                       # start - 5 - margin .. start + margin


# ---- alternation ------------------------------------------------------------------------------

def test_an_opponent_hit_after_a_near_hit_is_more_credible_than_one_after_another_opponent_hit():
    hits = [hit(0, 10, NEAR), hit(1, 20, FAR, evidence=EVIDENCE_NO_NEAR_HAND, confidence=0.2),
            hit(2, 30, FAR, evidence=EVIDENCE_NO_NEAR_HAND, confidence=0.2)]
    apply_alternation(hits, [], SINGLES, params())
    assert hits[1].confidence == 0.3 and hits[1].alternation_break is False
    assert hits[2].confidence == 0.2 and hits[2].alternation_break is True


def test_two_near_hits_in_a_row_are_flagged_unless_a_landing_separates_them():
    hits = [hit(0, 10, NEAR), hit(1, 20, NEAR), hit(2, 40, NEAR)]
    apply_alternation(hits, [call(30)], SINGLES, params())
    assert [h.alternation_break for h in hits] == [False, True, False]


def test_the_first_hit_of_a_rally_is_not_an_alternation_break():
    hits = [hit(0, 10, FAR, evidence=EVIDENCE_NO_NEAR_HAND, confidence=0.2)]
    apply_alternation(hits, [], SINGLES, params())
    assert hits[0].alternation_break is False and hits[0].confidence == 0.2


def test_rejected_hits_do_not_take_part_in_the_alternation():
    hits = [hit(0, 10, NEAR), hit(1, 15, FAR, valid=False, evidence=EVIDENCE_NONE, confidence=0.0), hit(2, 20, NEAR)]
    apply_alternation(hits, [], SINGLES, params())
    assert hits[2].alternation_break is True                # the rejected one in between is not an opponent hit


def test_in_doubles_inferred_opponent_hits_become_unknown(caplog):
    hits = [hit(0, 10, NEAR), hit(1, 20, FAR, evidence=EVIDENCE_NO_NEAR_HAND, confidence=0.2),
            hit(2, 30, FAR, evidence=EVIDENCE_NO_NEAR_HAND, confidence=0.2)]
    with caplog.at_level("WARNING", logger="core.hit_events"):
        apply_alternation(hits, [], DOUBLES, params())
    assert [h.hitter for h in hits] == [NEAR, UNKNOWN, UNKNOWN] and hits[1].confidence == 0.0
    assert len([r for r in caplog.records if "doubles" in r.message]) == 1


# ---- racket hand ------------------------------------------------------------------------------

def test_racket_hand_votes_come_from_valid_near_hits_per_player():
    hits = [hit(0, 10, NEAR, player_id=1, hand=RIGHT), hit(1, 20, NEAR, player_id=1, hand=RIGHT),
            hit(2, 30, NEAR, player_id=1, hand=LEFT), hit(3, 40, NEAR, player_id=2, hand=LEFT),
            hit(4, 50, NEAR, player_id=1, hand=LEFT, valid=False), hit(5, 60, FAR, hand=None)]
    result = racket_hands(hits)
    assert result[1] == (RIGHT, pytest.approx(2 / 3), 3)
    assert result[2] == (LEFT, 1.0, 1) and 3 not in result


@pytest.mark.parametrize("call_frame, is_bounce", [(42, True), (41, False), (58, True), (59, False)])
def test_the_landing_exclusion_window_is_inclusive_at_its_edge(call_frame, is_bounce):
    px = shuttle({45: (100, 100), 50: (200, 200)})
    away = frames(default=obs(right=(900, 900)))
    [h] = attribute_hits([measurement(50, (200, 200))], px, away, [call(call_frame)], params())   # landing_exclude_frames = 8
    assert h.valid is not is_bounce
