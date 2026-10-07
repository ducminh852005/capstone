import pytest

from core.hit_events import EVIDENCE_WRIST, FAR, NEAR, UNKNOWN, ShuttleHit
from core.rally import (END_CLIP, END_LANDED_IN, END_LANDED_OUT, END_OWN_HALF, END_TIMEOUT, RallyParams,
                        segment_rallies)
from core.umpire import Call

FPS = 60.0
PARAMS = RallyParams(max_hit_gap_s=3.0, close_call_factor=0.5, resting_factor=0.7, own_half_factor=0.5,
                     inferred_hitter_confidence=0.3)
GAP = int(PARAMS.max_hit_gap_s * FPS) + 1       # frames that are just over the maximum gap between hits


def hit(index, frame, hitter, confidence=0.8, valid=True):
    return ShuttleHit(index=index, contact_frame_idx=frame, start_frame_idx=frame, window_frames=(frame, frame),
                      shuttle_pt_px=(0.0, 0.0), prev_pt_px=None, source="physics", speed_px_per_frame=1.0,
                      angle_deg=0.0, is_smash=False, dt_frames=5, hitter=hitter, evidence=EVIDENCE_WRIST,
                      confidence=confidence, low_confidence=False, valid=valid)


def call(frame, half="far", result="IN", close=False, method="contact"):
    return Call(frame, result, half, (0, 0), (0.0, 0.0), 0.5, close, method)


def one(hits, calls, **kw):
    rallies = segment_rallies(hits, calls, FPS, PARAMS, **kw)
    assert len(rallies) == 1
    return rallies[0]


@pytest.mark.parametrize("last, half, result, cause, winner", [
    (NEAR, "far", "IN", END_LANDED_IN, NEAR),          # the near player's shot lands in the far half: his point
    (NEAR, "far", "OUT", END_LANDED_OUT, FAR),         # it lands out: the opponent's point
    (FAR, "near", "IN", END_LANDED_IN, FAR),
    (FAR, "near", "OUT", END_LANDED_OUT, NEAR),
    (NEAR, "near", "IN", END_OWN_HALF, FAR),           # it falls on his own half: he loses the point
    (FAR, "far", "OUT", END_OWN_HALF, NEAR),
])
def test_winner_and_cause_from_the_last_hitter_and_the_landing(last, half, result, cause, winner):
    r = one([hit(0, 100, NEAR), hit(1, 160, FAR), hit(2, 220, last)], [call(300, half, result)])
    assert (r.end_cause, r.winner, r.last_hitter) == (cause, winner, last)
    assert r.loser == (FAR if winner == NEAR else NEAR)
    assert r.hit_indices == [0, 1, 2] and r.n_hits == 3 and r.call_index == 0
    assert (r.start_frame_idx, r.end_frame_idx) == (100, 300)


def test_winner_confidence_is_the_last_hit_confidence_scaled_by_the_doubts_about_the_call():
    plain = one([hit(0, 100, NEAR, confidence=0.8)], [call(200)])
    assert plain.winner_confidence == pytest.approx(0.8)
    assert one([hit(0, 100, NEAR, 0.8)], [call(200, close=True)]).winner_confidence == pytest.approx(0.4)
    assert one([hit(0, 100, NEAR, 0.8)], [call(200, method="resting")]).winner_confidence == pytest.approx(0.56)
    assert one([hit(0, 100, NEAR, 0.8)], [call(200, half="near")]).winner_confidence == pytest.approx(0.4)
    both = one([hit(0, 100, NEAR, 0.8)], [call(200, close=True, method="resting")])
    assert both.winner_confidence == pytest.approx(0.8 * 0.5 * 0.7)


def test_an_inferred_opponent_hit_gives_a_low_confidence_winner():
    r = one([hit(0, 100, NEAR), hit(1, 160, FAR, confidence=0.2)], [call(220, "near", "OUT")])
    assert (r.winner, r.winner_confidence) == (NEAR, pytest.approx(0.2))


def test_a_landing_with_no_detected_hit_infers_the_hitter_from_where_it_landed():
    r = one([], [call(200, "far", "IN")])
    assert (r.last_hitter, r.winner, r.hit_indices, r.n_hits) == (NEAR, NEAR, [], 0)
    assert r.winner_confidence == pytest.approx(0.3) and r.start_frame_idx == r.end_frame_idx == 200
    r = one([], [call(200, "near", "OUT")])
    assert (r.last_hitter, r.winner) == (FAR, NEAR)         # the far player's shot went out in the near half


def test_an_unknown_last_hitter_gives_no_winner_but_still_the_cause():
    r = one([hit(0, 100, UNKNOWN, confidence=0.0)], [call(200, "far", "OUT")])
    assert (r.winner, r.loser, r.winner_confidence, r.end_cause) == (None, None, 0.0, END_LANDED_OUT)


def test_rejected_hits_are_not_part_of_a_rally():
    r = one([hit(0, 100, NEAR), hit(1, 130, FAR, valid=False)], [call(200)])
    assert r.hit_indices == [0] and r.last_hitter == NEAR


def test_each_landing_ends_a_rally_and_the_next_hit_starts_another():
    rallies = segment_rallies([hit(0, 100, NEAR), hit(1, 160, FAR), hit(2, 400, NEAR)],
                              [call(200, "near"), call(500, "far")], FPS, PARAMS)
    assert [(r.index, r.hit_indices, r.call_index) for r in rallies] == [(0, [0, 1], 0), (1, [2], 1)]
    assert [r.winner for r in rallies] == [FAR, NEAR]


def test_call_index_refers_to_the_list_that_was_passed_in():
    calls = [call(500, "far"), call(200, "near")]            # not in frame order
    rallies = segment_rallies([hit(0, 100, NEAR), hit(1, 400, NEAR)], calls, FPS, PARAMS)
    assert [r.call_index for r in rallies] == [1, 0]


def test_a_long_silence_times_a_rally_out_without_a_winner():
    rallies = segment_rallies([hit(0, 100, NEAR), hit(1, 100 + GAP, FAR)], [call(100 + GAP + 50, "near")], FPS, PARAMS)
    assert [(r.end_cause, r.winner, r.hit_indices) for r in rallies] == [(END_TIMEOUT, None, [0]),
                                                                          (END_LANDED_IN, FAR, [1])]
    assert rallies[0].end_frame_idx == 100                    # ends at its last hit


def test_a_landing_long_after_the_last_hit_is_not_credited_to_that_rally():
    rallies = segment_rallies([hit(0, 100, NEAR)], [call(100 + GAP, "far")], FPS, PARAMS)
    assert [(r.end_cause, r.n_hits) for r in rallies] == [(END_TIMEOUT, 1), (END_LANDED_IN, 0)]


def test_a_rally_cut_by_the_end_of_the_clip_is_not_a_timeout():
    hits = [hit(0, 100, NEAR), hit(1, 160, FAR)]
    assert one(hits, [], last_frame_idx=200).end_cause == END_CLIP
    assert one(hits, [], last_frame_idx=160 + GAP + 5).end_cause == END_TIMEOUT
    assert one(hits, []).end_cause == END_TIMEOUT             # end of the clip unknown
    r = one(hits, [], last_frame_idx=200)
    assert (r.winner, r.winner_confidence, r.end_frame_idx) == (None, 0.0, 160)


def test_rallies_that_began_in_the_preroll_are_flagged():
    rallies = segment_rallies([hit(0, 50, NEAR), hit(1, 300, NEAR)], [call(100), call(400)], FPS, PARAMS,
                              analysis_start_frame=120)
    assert [r.in_preroll for r in rallies] == [True, False]


def test_a_hit_and_a_call_on_the_same_frame_keep_the_hit_in_that_rally():
    r = one([hit(0, 100, NEAR)], [call(100)])
    assert r.hit_indices == [0] and r.call_index == 0


def test_no_hits_and_no_calls_means_no_rallies():
    assert segment_rallies([], [], FPS, PARAMS) == []
