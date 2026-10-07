from types import SimpleNamespace

import numpy as np
import pytest

from core import config
from core.hit_events import EVIDENCE_WRIST, FAR, NEAR, UNKNOWN, ShuttleHit
from core.pose_features import NOSE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST
from core.rally import Rally
from core.shots import (CLEAR, DRIVE, DROP, LIFT, LOW, MID_HEIGHT, NET, OVERHEAD, SERVE, SMASH,
                        UNKNOWN as UNKNOWN_SHOT, ShotFeatures, ShotThresholds, build_shots, classify_shot,
                        elevation_deg, match_rules)
from core.umpire import Call

FPS = 60.0


def thresholds(**kw):
    base = dict(overhead_min_above_head_body=0.0, low_min_below_hip_body=0.0, serve_max_x_m=4.7,
                smash_min_elbow_deg=140.0, smash_min_speed_body_per_s=2.5, flat_elevation_deg=10.0,
                drop_max_landing_from_net_m=2.5, drop_max_flight_s=0.9, clear_min_flight_s=1.1, lift_min_flight_s=1.0,
                net_max_flight_s=0.7, drive_min_speed_body_per_s=2.0, drive_max_flight_s=0.6, no_pose_max_conf=0.4,
                base_strength=0.6, dead_time_s=4.0, depth_edges_m=(2.2, 4.4), contact_window_frames=2,
                min_pose_score=0.5)
    base.update(kw)
    return ShotThresholds(**base)


def features(**kw):
    base = dict(pose_missing=False, contact_height=OVERHEAD, wrist_above_head_body=0.2, wrist_below_hip_body=-0.5,
                elbow_angle_deg=170.0, elevation_deg=0.0, speed_px_per_frame=10.0, speed_body_per_s=1.0, flight_s=1.0,
                is_rally_opening=False, hitter_foot_world_m=[1.5, 3.0], hitter_depth="rear")
    base.update(kw)
    return ShotFeatures(**base)


def kinds(f, t=None):
    return [m[0] for m in match_rules(f, t or thresholds())]


def hit(index=0, frame=100, hitter=NEAR, confidence=0.9, valid=True, player_id=1, hand="right", angle=45.0, speed=10.0):
    return ShuttleHit(index=index, contact_frame_idx=frame, start_frame_idx=frame, window_frames=(frame, frame),
                      shuttle_pt_px=(0.0, 0.0), prev_pt_px=None, source="physics", speed_px_per_frame=speed,
                      angle_deg=angle, is_smash=False, dt_frames=5, hitter=hitter, evidence=EVIDENCE_WRIST,
                      confidence=confidence, low_confidence=False, valid=valid, player_id=player_id, hand=hand)


def test_elevation_is_the_angle_above_the_horizontal_whatever_the_horizontal_direction():
    assert elevation_deg(90.0) == pytest.approx(-90.0) and elevation_deg(-90.0) == pytest.approx(90.0)
    assert elevation_deg(0.0) == pytest.approx(0.0) and elevation_deg(180.0) == pytest.approx(0.0, abs=1e-9)
    assert elevation_deg(30.0) == pytest.approx(-30.0) and elevation_deg(150.0) == pytest.approx(-30.0)
    assert elevation_deg(-30.0) == pytest.approx(30.0)


# ---- rules ------------------------------------------------------------------------------------

def test_a_fast_overhead_hit_that_does_not_rise_is_a_smash_with_its_supporting_evidence():
    f = features(speed_body_per_s=4.0, elevation_deg=-30.0)
    [(kind, reasons, strength), *_] = match_rules(f, thresholds())
    assert kind == SMASH and strength == pytest.approx(1.0)                 # both bonus conditions hold
    assert {"contact_overhead", "shuttle_fast", "not_rising", "hit_from_back", "arm_extended",
            "shuttle_falling"} <= set(reasons)
    flat_bent = match_rules(features(speed_body_per_s=4.0, elbow_angle_deg=90.0, elevation_deg=0.0), thresholds())[0]
    assert flat_bent[0] == SMASH and flat_bent[2] == pytest.approx(0.6)     # neither bonus holds


def test_a_slow_overhead_hit_is_not_a_smash_and_a_rising_fast_one_is_not_either():
    assert SMASH not in kinds(features(speed_body_per_s=1.0, elevation_deg=-30.0))
    assert SMASH not in kinds(features(speed_body_per_s=4.0, elevation_deg=30.0))
    assert SMASH not in kinds(features(speed_body_per_s=4.0, hitter_depth="front"))      # smashes come from the back


def test_a_short_soft_overhead_hit_is_a_drop_by_flight_time_or_by_landing():
    assert kinds(features(speed_body_per_s=1.0, flight_s=0.6))[0] == DROP
    assert kinds(features(speed_body_per_s=1.0, flight_s=None, landing_from_net_m=1.0))[0] == DROP
    assert DROP not in kinds(features(speed_body_per_s=1.0, flight_s=None, landing_from_net_m=4.0))
    assert DROP not in kinds(features(speed_body_per_s=1.0, flight_s=None))             # no evidence at all


def test_an_overhead_hit_that_rises_or_flies_long_is_a_clear():
    assert kinds(features(elevation_deg=30.0, flight_s=1.6))[0] == CLEAR
    assert kinds(features(elevation_deg=0.0, flight_s=1.6))[0] == CLEAR
    assert CLEAR not in kinds(features(elevation_deg=0.0, flight_s=0.8, speed_body_per_s=1.0))
    assert CLEAR not in kinds(features(contact_height=LOW, elevation_deg=30.0))


def test_a_low_rising_hit_is_a_lift():
    assert kinds(features(contact_height=LOW, elevation_deg=30.0, flight_s=1.5))[0] == LIFT
    assert LIFT not in kinds(features(contact_height=LOW, elevation_deg=0.0, flight_s=0.5))


def test_a_soft_hit_at_the_net_is_a_net_shot_and_a_flat_fast_mid_hit_is_a_drive():
    assert kinds(features(contact_height=MID_HEIGHT, hitter_depth="front", speed_body_per_s=0.5, flight_s=0.5))[0] == NET
    assert kinds(features(contact_height=LOW, hitter_depth="front", speed_body_per_s=0.5, flight_s=0.5))[0] == NET
    assert NET not in kinds(features(contact_height=MID_HEIGHT, hitter_depth="mid", speed_body_per_s=0.5))
    drive = features(contact_height=MID_HEIGHT, hitter_depth="mid", speed_body_per_s=3.0, flight_s=0.4)
    assert kinds(drive)[0] == DRIVE
    assert DRIVE not in kinds(features(contact_height=MID_HEIGHT, speed_body_per_s=3.0, elevation_deg=40.0))


def test_a_low_hit_opening_a_rally_from_behind_the_service_line_is_a_serve():
    base = dict(contact_height=LOW, is_rally_opening=True, hitter_foot_world_m=[2.0, 3.0])
    assert kinds(features(**base))[0] == SERVE
    assert SERVE not in kinds(features(**{**base, "hitter_foot_world_m": [5.0, 3.0]}))
    assert SERVE not in kinds(features(**{**base, "is_rally_opening": False}))
    assert SERVE not in kinds(features(**{**base, "contact_height": OVERHEAD}))


def test_the_first_matching_rule_wins_and_the_next_is_the_runner_up():
    f = features(speed_body_per_s=1.0, elevation_deg=30.0, flight_s=0.8)        # a drop by flight time AND a clear (rising)
    assert kinds(f)[:2] == [DROP, CLEAR]
    shot = classify_shot(hit(), f, thresholds())
    assert (shot.type, shot.runner_up) == (DROP, CLEAR)


def test_confidence_is_the_hit_confidence_times_the_strength_of_the_rule():
    f = features(contact_height=LOW, is_rally_opening=True, hitter_foot_world_m=[2.0, 3.0])
    assert classify_shot(hit(confidence=0.9), f, thresholds()).confidence == pytest.approx(0.9 * 0.6)    # no bonus
    smash = features(speed_body_per_s=4.0, elevation_deg=-30.0)
    assert classify_shot(hit(confidence=0.5), smash, thresholds()).confidence == pytest.approx(0.5)


def test_without_a_skeleton_the_rules_use_the_shuttle_and_court_only_with_a_low_ceiling():
    f = features(pose_missing=True, contact_height=None, elbow_angle_deg=None, speed_body_per_s=4.0, elevation_deg=-30.0)
    shot = classify_shot(hit(confidence=0.9), f, thresholds())
    assert shot.type == SMASH and shot.confidence == pytest.approx(0.4) and "pose_missing" in shot.reasons


def test_a_hit_no_rule_matches_is_unknown_with_zero_confidence():
    shot = classify_shot(hit(), features(speed_body_per_s=1.0, flight_s=None, elevation_deg=0.0), thresholds())
    assert (shot.type, shot.confidence, shot.reasons, shot.runner_up) == (UNKNOWN_SHOT, 0.0, ["no_rule_matches"], None)


def test_thresholds_are_read_from_config_when_asked(monkeypatch):
    monkeypatch.setattr(config, "SHOT_SMASH_MIN_SPEED_BODY_PER_S", 9.0)
    assert ShotThresholds.from_config().smash_min_speed_body_per_s == 9.0
    monkeypatch.setattr(config, "SHOT_SMASH_MIN_SPEED_BODY_PER_S", 1.0)
    assert ShotThresholds.from_config().smash_min_speed_body_per_s == 1.0


# ---- features from a clip ----------------------------------------------------------------------

def overhead_skeleton():
    lm = np.zeros((33, 3), np.float32)
    for index, xy in {NOSE: (50, 40), RIGHT_HIP: (45, 120), RIGHT_SHOULDER: (40, 70), RIGHT_ELBOW: (40, 45),
                      RIGHT_WRIST: (40, 20)}.items():
        lm[index] = (*xy, 0.9)
    return lm


def player(with_pose=True, foot=(1.5, 3.0)):
    return SimpleNamespace(bbox=(0.0, 0.0, 100.0, 200.0), landmarks=overhead_skeleton() if with_pose else None,
                           foot_world=foot)


def rally(index, hit_indices, call_index=None, start=100, end=200):
    return Rally(index=index, start_frame_idx=start, end_frame_idx=end, hit_indices=hit_indices, n_hits=len(hit_indices),
                 last_hitter=NEAR, call_index=call_index, end_cause="landed_in", winner=NEAR, loser=FAR,
                 winner_confidence=0.9, in_preroll=False)


def call(frame, method="contact", result="IN", world=(9.0, 3.0)):
    return Call(frame, result, "far", (0, 0), world, 0.5, False, method)


def frames(n=400, per_frame=None):
    out = [{1: player()} for _ in range(n)]
    for f, v in (per_frame or {}).items():
        out[f] = v
    return out


def test_a_hit_is_measured_from_the_arm_the_shuttle_and_the_rally():
    hits = [hit(0, 100, angle=60.0, speed=20.0), hit(1, 130, hitter=FAR, player_id=None, hand=None)]
    shots = build_shots(hits, [rally(0, [0, 1], call_index=0, end=160)], [call(160)], frames(), FPS, thresholds())
    s = shots[0]
    f = s.features
    assert (s.type, s.hitter, s.player_id, s.frame_idx) == (SMASH, NEAR, 1, 100)
    assert f.contact_height == OVERHEAD and f.pose_missing is False and f.elbow_angle_deg == pytest.approx(180.0)
    assert f.elevation_deg == pytest.approx(-60.0) and f.speed_body_per_s == pytest.approx(20 * FPS / 200)
    assert f.flight_s == pytest.approx(30 / FPS) and f.hitter_depth == "rear" and f.landing_result is None
    assert (shots[1].type, shots[1].reasons) == (UNKNOWN_SHOT, ["opponent_not_tracked"])


def test_the_last_hit_of_a_rally_knows_the_landing_and_the_ground_speed():
    shots = build_shots([hit(0, 100)], [rally(0, [0], call_index=0, end=130)], [call(130, world=(9.0, 3.0))], frames(),
                        FPS, thresholds())
    f = shots[0].features
    assert f.flight_s == pytest.approx(0.5) and f.landing_result == "IN"
    assert f.landing_from_net_m == pytest.approx(2.3)
    assert f.ground_dist_m == pytest.approx(7.5) and f.ground_speed_mps == pytest.approx(15.0)
    resting = build_shots([hit(0, 100)], [rally(0, [0], call_index=0, end=130)], [call(130, method="resting")], frames(),
                          FPS, thresholds())[0].features
    assert resting.landing_result == "IN" and resting.ground_speed_mps is None


def test_a_hit_with_no_next_event_has_no_flight_time():
    shot = build_shots([hit(0, 100)], [rally(0, [0])], [], frames(), FPS, thresholds())[0]
    assert shot.features.flight_s is None and shot.features.landing_result is None


def test_the_skeleton_of_a_neighbouring_frame_is_used_when_the_contact_frame_has_none():
    players = frames(per_frame={100: {1: player(with_pose=False)}, 101: {1: player()}})
    shot = build_shots([hit(0, 100)], [rally(0, [0])], [], players, FPS, thresholds())[0]
    assert shot.features.pose_missing is False
    none = frames(per_frame={f: {1: player(with_pose=False)} for f in range(95, 106)})
    assert build_shots([hit(0, 100)], [rally(0, [0])], [], none, FPS, thresholds())[0].features.pose_missing is True


def test_only_the_rally_opening_after_a_pause_can_be_a_serve():
    low = overhead_skeleton()
    low[RIGHT_WRIST] = (40, 160, 0.9)
    players = [{1: SimpleNamespace(bbox=(0.0, 0.0, 100.0, 200.0), landmarks=low, foot_world=(2.0, 3.0))}
               for _ in range(900)]
    hits = [hit(0, 100, angle=-45.0, speed=3.0), hit(1, 160, angle=-45.0, speed=3.0), hit(2, 700, angle=-45.0, speed=3.0)]
    rallies = [rally(0, [0, 1], start=100, end=200), rally(1, [2], start=700, end=800)]
    types = [s.type for s in build_shots(hits, rallies, [], players, FPS, thresholds())]
    assert types[0] == SERVE and types[1] != SERVE and types[2] == SERVE          # 500 frames (8 s) of pause before the third
    close = [rally(0, [0, 1], start=100, end=200), rally(1, [2], start=300, end=400)]
    assert build_shots(hits, close, [], players, FPS, thresholds())[2].type != SERVE


def test_rejected_and_unidentified_hits():
    hits = [hit(0, 100, valid=False), hit(1, 150, hitter=UNKNOWN, player_id=None, hand=None)]
    shots = build_shots(hits, [], [], frames(), FPS, thresholds())
    assert [(s.hit_index, s.type, s.reasons) for s in shots] == [(1, UNKNOWN_SHOT, ["hitter_unknown"])]
