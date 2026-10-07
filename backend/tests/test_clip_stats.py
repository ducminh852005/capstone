import pytest

from core.clip_stats import landing_zone, summarize_shots
from core.hit_events import FAR, NEAR
from core.rally import Rally
from core.shots import CLEAR, SMASH, UNKNOWN, Shot, ShotFeatures

EDGES = (2.2, 4.4)


def feats(flight=1.0, speed=3.0, ground=None, landing=None, world=None):
    return ShotFeatures(pose_missing=False, contact_height="overhead", wrist_above_head_body=0.2,
                        wrist_below_hip_body=-0.5, elbow_angle_deg=170.0, elevation_deg=0.0, speed_px_per_frame=10.0,
                        speed_body_per_s=speed, flight_s=flight, is_rally_opening=False, hitter_foot_world_m=[1.5, 3.0],
                        hitter_depth="rear", landing_world_m=world, landing_result=landing, ground_speed_mps=ground)


def shot(index, kind, frame=200, player=1, hitter=NEAR, features=None):
    return Shot(index, frame, hitter, player, kind, 0.8, None, [], features if features is not None else feats())


def rally(index, hit_indices, winner=None, loser=None):
    return Rally(index=index, start_frame_idx=0, end_frame_idx=0, hit_indices=hit_indices, n_hits=len(hit_indices),
                 last_hitter=NEAR, call_index=None, end_cause="landed_in", winner=winner, loser=loser,
                 winner_confidence=0.9, in_preroll=False)


def test_counts_and_shares_per_type_in_the_order_of_shot_types():
    shots = [shot(0, CLEAR), shot(1, SMASH), shot(2, CLEAR), shot(3, UNKNOWN)]
    s = summarize_shots(shots, [], 1, 0, EDGES)
    assert (s.player_id, s.n_shots, s.n_classified) == (1, 4, 3)
    assert [(t.type, t.n, t.share_pct) for t in s.by_type] == [(SMASH, 1, 25.0), (CLEAR, 2, 50.0), (UNKNOWN, 1, 25.0)]


def test_only_the_players_own_shots_after_the_preroll_count():
    shots = [shot(0, CLEAR, frame=50), shot(1, CLEAR, frame=200), shot(2, CLEAR, frame=300, player=2),
             shot(3, CLEAR, frame=300, hitter=FAR, player=None)]
    s = summarize_shots(shots, [], 1, 120, EDGES)
    assert s.n_shots == 1 and s.by_type[0].n == 1
    assert summarize_shots([], [], 1, 0, EDGES).by_type == []


def test_means_ignore_shots_without_the_figure():
    shots = [shot(0, SMASH, features=feats(flight=0.5, speed=4.0, ground=10.0)),
             shot(1, SMASH, features=feats(flight=None, speed=2.0, ground=None))]
    t = summarize_shots(shots, [], 1, 0, EDGES).by_type[0]
    assert t.mean_flight_s == pytest.approx(0.5) and t.mean_speed_body_per_s == pytest.approx(3.0)
    assert t.mean_ground_speed_mps == pytest.approx(10.0)
    none = summarize_shots([shot(0, SMASH, features=feats(flight=None, ground=None))], [], 1, 0, EDGES).by_type[0]
    assert none.mean_flight_s is None and none.mean_ground_speed_mps is None


def test_out_rate_and_landing_zones_only_count_shots_that_landed():
    shots = [shot(0, SMASH, features=feats(landing="IN", world=[11.7, 1.0])),
             shot(1, SMASH, features=feats(landing="OUT", world=[8.0, 5.0])),
             shot(2, SMASH)]
    t = summarize_shots(shots, [], 1, 0, EDGES).by_type[0]
    assert (t.n, t.n_with_landing, t.out_rate) == (3, 2, 0.5)
    assert t.landing_zone_counts["rear_left"] == 1 and t.landing_zone_counts["front_right"] == 1
    assert sum(t.landing_zone_counts.values()) == 2
    assert summarize_shots([shot(0, SMASH)], [], 1, 0, EDGES).by_type[0].out_rate is None


def test_far_half_landings_are_measured_from_the_far_baseline():
    assert landing_zone((11.7, 1.0), EDGES) == "rear_left"        # 13.4 - 11.7 = 1.7 m from the far baseline
    assert landing_zone((8.0, 5.0), EDGES) == "front_right"       # 5.4 m from it
    assert landing_zone((1.0, 5.0), EDGES) == "rear_right"        # near half: from the near baseline


def test_a_shot_that_ends_a_rally_is_a_winner_or_an_error():
    shots = [shot(0, SMASH), shot(1, SMASH), shot(2, SMASH), shot(3, CLEAR)]
    rallies = [rally(0, [0], winner=NEAR, loser=FAR), rally(1, [1], winner=FAR, loser=NEAR), rally(2, [2]),
               rally(3, [9, 3], winner=NEAR, loser=FAR)]
    by = {t.type: t for t in summarize_shots(shots, rallies, 1, 0, EDGES).by_type}
    assert (by[SMASH].n_rally_winners, by[SMASH].n_rally_errors) == (1, 1)       # the third ended without a winner
    assert (by[CLEAR].n_rally_winners, by[CLEAR].n_rally_errors) == (1, 0)
