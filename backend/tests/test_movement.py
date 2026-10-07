import pytest

from core.movement import BASE_FROM_CLIP, BASE_FROM_RALLY, MovementParams, compute_movement
from core.zones import ZONE_IDS, zone_of

EDGES = (2.2, 4.4)


def params(**kw):
    base = dict(max_speed_mps=7.0, max_gap_s=0.5, smooth_window_s=0.4, speed_levels_mps=(0.5, 2.0, 4.0),
                still_speed_mps=0.5, rest_min_s=2.0, base_min_s=0.5, home_radius_m=0.75, recovery_max_s=3.0,
                min_valid_ratio=0.6, small_sample_s=10.0, depth_edges_m=EDGES)
    base.update(kw)
    return MovementParams(**base)


def line(n, fps, start, velocity):
    """Positions (m) of a player walking from `start` at `velocity` (m/s), one per frame."""
    return [(start[0] + velocity[0] * i / fps, start[1] + velocity[1] * i / fps) for i in range(n)]


def measure(track, fps=60.0, hits=(), spans=(), first=0, last=None, **kw):
    last = len(track) - 1 if last is None else last
    return compute_movement(track, fps, hits, spans, first, last, params(**kw))


# ---- zones ------------------------------------------------------------------------------------

def test_zone_of_splits_the_near_half_in_six():
    assert zone_of((1.0, 1.0), EDGES) == "rear_left"
    assert zone_of((1.0, 5.0), EDGES) == "rear_right"
    assert zone_of((3.0, 1.0), EDGES) == "mid_left"
    assert zone_of((3.0, 5.0), EDGES) == "mid_right"
    assert zone_of((5.5, 1.0), EDGES) == "front_left"
    assert zone_of((5.5, 5.0), EDGES) == "front_right"
    assert len(set(ZONE_IDS)) == 6


def test_zone_edges_belong_to_the_deeper_zone_and_outside_points_go_to_the_nearest_zone():
    assert zone_of((2.2, 3.0), EDGES) == "mid_left" and zone_of((4.4, 3.05), EDGES) == "front_right"
    assert zone_of((-0.8, -0.3), EDGES) == "rear_left"        # behind the baseline, left of the sideline
    assert zone_of((7.0, 9.0), EDGES) == "front_right"        # beyond the net


# ---- distance, speed, zones -------------------------------------------------------------------

def test_walking_at_one_metre_per_second_along_the_court():
    s = measure(line(301, 60.0, (1.0, 3.0), (1.0, 0.0)))        # 5 s, x from 1 to 6
    assert s.distance_m == pytest.approx(5.0, abs=0.05)
    assert s.distance_along_m == pytest.approx(5.0, abs=0.05) and s.distance_across_m == pytest.approx(0.0, abs=0.01)
    assert s.mean_speed_mps == pytest.approx(1.0, abs=0.02) and s.p95_speed_mps == pytest.approx(1.0, abs=0.02)
    assert s.speed_level_share == pytest.approx([0.0, 1.0, 0.0, 0.0])      # walking: between 0.5 and 2 m/s
    assert s.duration_s == pytest.approx(301 / 60) and s.valid_ratio == 1.0 and s.reliable
    assert s.zone_time_s["rear_left"] == pytest.approx(1.2, abs=0.05)      # x 1.0..2.2
    assert s.zone_time_s["mid_left"] == pytest.approx(2.2, abs=0.05)
    assert s.zone_time_s["front_left"] == pytest.approx(1.6, abs=0.05)
    assert sum(s.zone_time_s.values()) == pytest.approx(301 / 60)
    assert s.small_sample is True


def test_distance_and_speed_do_not_depend_on_the_frame_rate():
    a = measure(line(301, 60.0, (1.0, 3.0), (1.0, 0.5)), fps=60.0)
    b = measure(line(151, 30.0, (1.0, 3.0), (1.0, 0.5)), fps=30.0)
    assert b.distance_m == pytest.approx(a.distance_m, abs=0.05)
    assert b.mean_speed_mps == pytest.approx(a.mean_speed_mps, abs=0.03)
    assert b.duration_s == pytest.approx(a.duration_s, abs=0.05)


def test_speed_levels_follow_the_edges():
    s = measure(line(121, 60.0, (0.0, 3.0), (0.0, 3.0)))        # 3 m/s across the court: running
    assert s.speed_level_share == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=0.02)


def test_a_short_gap_is_bridged_but_a_long_gap_is_not_counted():
    track = line(301, 60.0, (1.0, 3.0), (1.0, 0.0))
    for i in range(100, 112):
        track[i] = None                                          # 0.2 s: interpolated
    assert measure(track).distance_m == pytest.approx(5.0, abs=0.1)
    for i in range(100, 160):
        track[i] = None                                          # 1 s: left empty
    s = measure(track)
    assert s.distance_m == pytest.approx(4.0, abs=0.1) and s.valid_ratio == pytest.approx(0.8, abs=0.03)
    assert s.reliable


def test_few_usable_frames_make_the_numbers_unreliable_and_tracks_without_data_are_empty():
    track = line(301, 60.0, (1.0, 3.0), (1.0, 0.0))
    for i in range(100, 300):
        track[i] = None
    assert measure(track).reliable is False
    empty = measure([None] * 100)
    assert (empty.distance_m, empty.valid_ratio, empty.reliable, empty.mean_speed_mps) == (0.0, 0.0, False, None)
    assert sum(empty.zone_time_s.values()) == 0.0 and empty.base_position_m is None
    assert measure([(2.0, 3.0)] + [None] * 10).distance_m == 0.0


def test_only_the_analysed_stretch_is_measured_but_the_whole_track_is_cleaned():
    track = line(301, 60.0, (1.0, 3.0), (1.0, 0.0))
    s = measure(track, first=120, last=300)
    assert s.distance_m == pytest.approx(3.0, abs=0.05) and s.duration_s == pytest.approx(181 / 60)
    assert (s.first_frame_idx, s.last_frame_idx) == (120, 300)


def test_an_impossible_jump_is_rejected_as_an_outlier():
    track = line(121, 60.0, (1.0, 3.0), (0.0, 0.0))
    track[60] = (6.0, 3.0)                                       # 5 m in one frame
    assert measure(track).distance_m < 0.5


# ---- rest, waiting position, recovery ---------------------------------------------------------

def test_standing_still_for_long_enough_is_a_rest():
    track = [(2.0, 3.0)] * 181 + line(120, 60.0, (2.0, 3.0), (1.0, 0.0))        # 3 s still, then 2 s walking
    s = measure(track)
    assert s.rest_s == pytest.approx(3.0, abs=0.05)
    assert len(s.rest_intervals_frames) == 1 and s.rest_intervals_frames[0][0] <= 2
    short = measure([(2.0, 3.0)] * 61 + line(120, 60.0, (2.0, 3.0), (1.0, 0.0)))   # 1 s: not a rest
    assert short.rest_s == 0.0 and short.rest_intervals_frames == []


def out_and_back():
    """Still at x=2 for 1 s, 1 s out to x=4, 1 s back, then still for 1 s (60 fps)."""
    return ([(2.0, 3.0)] * 60 + line(60, 60.0, (2.0, 3.0), (2.0, 0.0)) + line(60, 60.0, (4.0, 3.0), (-2.0, 0.0))
            + [(2.0, 3.0)] * 60)


def test_waiting_position_and_the_recovery_after_a_hit():
    s = measure(out_and_back(), hits=[60], spans=[(0, 239)])
    assert s.base_source == BASE_FROM_RALLY and s.base_position_m == pytest.approx([2.0, 3.0], abs=0.05)
    [r] = s.recoveries
    assert (r.hit_frame_idx, r.farthest_frame_idx) == (60, pytest.approx(120, abs=3))
    assert r.excursion_m == pytest.approx(2.0, abs=0.15)
    assert r.recovery_s == pytest.approx(0.625, abs=0.1)         # 2 m/s from 2 m out to the 0.75 m home radius
    assert s.mean_recovery_s == r.recovery_s


def test_the_waiting_position_falls_back_to_the_whole_stretch_when_no_rally_covers_stillness():
    s = measure(out_and_back(), hits=[60], spans=[(70, 110)])
    assert s.base_source == BASE_FROM_CLIP and s.base_position_m == pytest.approx([2.0, 3.0], abs=0.05)
    assert measure([(2.0, 3.0)] * 121).base_position_m == pytest.approx([2.0, 3.0])         # a player who stands
    assert measure(line(121, 60.0, (1.0, 0.5), (0.0, 1.5)), hits=[10]).base_position_m is None   # never still


def test_a_recovery_that_is_cut_off_by_the_next_hit_or_the_end_is_not_completed():
    track = out_and_back()
    cut_by_next_hit = measure(track, hits=[60, 100], spans=[(0, 239)])
    assert cut_by_next_hit.recoveries[0].recovery_s is None
    assert cut_by_next_hit.recoveries[0].farthest_frame_idx == pytest.approx(100, abs=3)
    cut_by_end = measure(track[:130], hits=[60], spans=[(0, 129)], base_min_s=0.5)
    assert cut_by_end.recoveries[0].recovery_s is None
    assert cut_by_end.mean_recovery_s is None


def test_a_player_who_never_left_the_waiting_position_recovers_instantly():
    s = measure([(2.0, 3.0)] * 181, hits=[60], spans=[(0, 180)])
    assert s.recoveries[0].recovery_s == 0.0 and s.recoveries[0].excursion_m == pytest.approx(0.0, abs=0.01)


def test_hits_outside_the_analysed_stretch_are_ignored():
    s = measure(out_and_back(), hits=[5, 60], spans=[(0, 239)], first=30)
    assert [r.hit_frame_idx for r in s.recoveries] == [60]


def test_zone_counts_match_zone_of_for_every_point_and_depth_of_names_the_band():
    import numpy as np
    from core.zones import ZONE_IDS, depth_of, zone_counts
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-1.0, 7.0, 500), rng.uniform(-0.5, 6.6, 500)])
    expected = {z: 0 for z in ZONE_IDS}
    for p in pts:
        expected[zone_of(p, EDGES)] += 1
    assert zone_counts(pts, EDGES) == expected
    assert [depth_of((x, 3.0), EDGES) for x in (0.0, 2.2, 4.3, 4.4, 6.5)] == ["rear", "mid", "mid", "front", "front"]
