import json

import numpy as np
import pytest

from core import config, court_model, gap_fill
from core.clip_analysis import (HIDDEN_LANDMARK_PX, SCHEMA_VERSION, ClipAnalysisParams, ClipSource, CourtCalibration,
                                VideoMeta, build_clip_analysis, court_block, players_block, shuttle_block,
                                skeleton_row, to_jsonable)
from core.clip_pipeline import ClipRun, FrameRecord
from core.hit_events import HitterParams
from core.movement import MovementParams
from core.player_tracker import PlayerObs
from core.pose_features import RIGHT_WRIST
from core.rally import RallyParams
from core.smash import HitMeasurement
from core.umpire import Call

IMAGE_CORNERS = [(300, 900), (1600, 900), (500, 500), (1400, 500)]    # BL, BR, Net-L, Net-R


@pytest.fixture(scope="module")
def homographies():
    H = court_model.homography_from_points(IMAGE_CORNERS)
    return H, np.linalg.inv(H)


def skeleton(offset=0.0):
    return np.array([[100.0 + i + offset, 200.0 + i, 0.5 + i / 100] for i in range(33)], np.float32)


def obs(pid, tid, with_pose=False):
    return PlayerObs(pid, tid, (10.04, 20.0, 50.0, 120.0), (30.0, 120.0), (2.5, 3.0), "foot",
                     skeleton() if with_pose else None)


def record(i, players=None, pt=None, call=None, measurement=None):
    return FrameRecord(i, players or {}, pt, i, pt is not None, 0, 1, None, measurement, call)


def make_run(n=6):
    records = [record(i, {1: obs(1, 7, with_pose=i % 2 == 0)}, pt=(100 + i, 200 + i) if i in (1, 2, 5) else None)
               for i in range(n)]
    if n > 3:
        records[3].players[2] = obs(2, 9)      # a passer-by seen for one frame only
    run = ClipRun(records=records, shuttle_px=[r.shuttle_pt_px for r in records])
    run.filled = [gap_fill.FilledPoint(3, (103.5, 203.5), "stride"), gap_fill.FilledPoint(1, (999.0, 999.0), "stride")]
    return run


def build(run, homographies, fps=60.0, match_type="singles", source=None, **param_fields):
    """build_clip_analysis with the given ClipAnalysisParams fields (hitter, rally, movement, shots, extra_warnings)."""
    H, H_inv = homographies
    return build_clip_analysis(run, video=VideoMeta("clip.mp4", fps, 1920, 1080, 0),
                               source=source or ClipSource("tran04_cam1.mp4", 130, 120, 120),
                               court=CourtCalibration(H, H_inv, match_type), generator={"git": "abc"},
                               params=ClipAnalysisParams(**param_fields))


def test_shuttle_block_marks_detections_and_estimates_and_never_overwrites_a_detection():
    block = shuttle_block(make_run())
    assert block["px"][1] == [101.0, 201.0] and block["kind"][1] == "d"
    assert block["px"][3] == [103.5, 203.5] and block["kind"][3] == "e"
    assert block["px"][0] is None and block["kind"][0] is None
    assert block["trail_frames"] == config.CLIP_TRAIL_FRAMES


def test_players_block_aligns_every_list_with_the_clip_frames_and_drops_short_tracks():
    run = make_run(6)
    players = players_block(run, min_frames=2)
    assert [p.player_id for p in players] == [1]                    # P2 was seen once
    p = players[0]
    assert (p.n_frames, p.n_pose_frames, p.track_ids) == (6, 3, [7])
    for per_frame in (p.bbox_px, p.foot_px, p.foot_world_m, p.skeleton):
        assert len(per_frame) == 6
    assert p.bbox_px[0] == [10.0, 20.0, 50.0, 120.0]               # rounded to CLIP_EXPORT_PX_DECIMALS
    assert p.skeleton[1] is None and p.skeleton[0].shape == (66,) and p.skeleton[0].dtype == np.int32
    assert p.foot_world_m[0] == [2.5, 3.0]
    assert len(players_block(run, min_frames=1)) == 2


def test_player_without_calibration_has_no_world_track():
    run = make_run(2)
    run.records[0].players[1] = PlayerObs(1, 7, (0.0, 0.0, 10.0, 10.0), (5.0, 10.0), None, "bbox")
    assert players_block(run, 1)[0].foot_world_m[0] is None


def test_court_selftest_round_trips_through_the_inverse_homography(homographies):
    H, H_inv = homographies
    court = court_block(H, H_inv, "singles")
    assert court["calibration_covers"] == "near_half" and court["net_x_m"] == court_model.NET_X
    assert len(court["selftest"]) >= 5
    for pair in court["selftest"]:
        back = court_model.img_to_world([pair["img_px"]], H_inv)[0]
        assert back == pytest.approx(pair["world_m"], abs=1e-4)
    assert len(court["lines_m"]) == len(court_model.full_court_lines())


def test_analysis_serialises_to_plain_json_without_numpy_types(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)       # the 6-frame fixture has players
    run = make_run()
    run.records[2].call = Call(1, "OUT", "far", (11, 12), (np.float32(7.5), np.float32(1.0)), np.float32(-0.2), True,
                               "resting", 0.13)
    analysis = build(run, homographies)
    data = analysis.to_jsonable()
    text = json.dumps(data)                                         # raises on numpy scalars/arrays
    back = json.loads(text)
    assert back["schema_version"] == SCHEMA_VERSION
    assert back["video"]["n_frames"] == 6 and back["source"]["analysis_start_frame"] == 120
    assert back["events"]["landings"][0]["result"] == "OUT"
    assert back["events"]["landings"][0]["margin_m"] == -0.2
    assert set(back["labels"]) >= {"call_result", "call_half", "call_method", "hit_source"}
    assert back["pose"]["connections"][0] == [0, 1]
    assert back["players"][0]["player_id"] == back["default_player"]


def test_warnings_for_far_half_other_fps_and_missing_players(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 100)     # nobody qualifies
    run = make_run()
    codes = [w["code"] for w in build(run, homographies, fps=30.0).warnings]
    assert codes == ["far_half_extrapolated", "fps_not_60", "no_player"]
    extra = [{"code": "x", "message": "m"}]
    assert build(run, homographies, extra_warnings=extra).warnings[-1] == extra[0]
    assert build(run, homographies).default_player is None


def test_records_must_be_consecutive_from_frame_zero(homographies):
    run = make_run(3)
    run.records[1].frame_idx = 5
    with pytest.raises(ValueError, match="consecutive"):
        build(run, homographies)


def test_to_jsonable_handles_numpy_nan_and_tuples():
    out = to_jsonable({"a": np.float32(1.5), "b": (np.int64(2), np.bool_(True)), "c": np.array([[1.0, np.nan]]),
                       "d": float("inf")})
    assert out == {"a": 1.5, "b": [2, True], "c": [[1.0, None]], "d": None}
    assert type(out["b"][0]) is int and type(out["b"][1]) is bool


# ---- hits, rallies and racket hand ------------------------------------------------------------

def hitter_params(**kw):
    base = dict(max_wrist_dist_body=0.5, min_near_confidence=0.5, window_margin_frames=2, default_window_frames=5,
                max_prev_gap_frames=15, landing_exclude_frames=8, far_confidence=0.3, far_break_confidence=0.2,
                min_pose_score=0.5, hand=None)
    base.update(kw)
    return HitterParams(**base)


RALLY_PARAMS = RallyParams(3.0, 0.5, 0.7, 0.5, 0.3)


def rally_run(n=40):
    """A near player whose right wrist sits on the shuttle path of a hit started at frame 20 (previous
    detection at frame 15), a landing in the far half at frame 30, and a second start (frame 5) with
    no hand near it."""
    wrist = np.zeros((33, 3), np.float32)
    wrist[RIGHT_WRIST] = (150, 150, 0.9)
    player = PlayerObs(1, 7, (0.0, 0.0, 100.0, 200.0), (50.0, 200.0), (2.0, 3.0), "foot", wrist)
    records = [record(i, {1: player}) for i in range(n)]
    px = [None] * n
    px[5], px[15], px[20] = (900, 900), (100, 100), (200, 200)
    for i in (5, 15, 20):
        records[i].shuttle_pt_px = px[i]
    records[20].measurement = HitMeasurement(20, (200, 200), 21.0, 80.0, True, 1, "physics", 5)
    records[5].measurement = HitMeasurement(5, (900, 900), 4.0, 10.0, False, 0, "new", 4)
    records[32].call = Call(30, "IN", "far", (11, 12), (9.0, 3.0), 0.8, False)
    return ClipRun(records=records, shuttle_px=px)


def build_rally(run, homographies, **kw):
    kw.setdefault("hitter", hitter_params())
    kw.setdefault("rally", RALLY_PARAMS)
    return build(run, homographies, **kw)


def test_hits_are_attributed_to_the_player_and_rallies_decided(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    analysis = build_rally(rally_run(), homographies)
    hits = analysis.events["hits"]
    assert [(h.start_frame_idx, h.source, h.valid, h.reject_reason) for h in hits] == [
        (5, "new", False, "new_track_no_hitter"), (20, "physics", True, None)]
    assert (hits[1].hitter, hits[1].player_id, hits[1].hand, hits[1].is_smash) == ("near", 1, "right", True)
    [rally] = analysis.events["rallies"]
    assert (rally.hit_indices, rally.end_cause, rally.winner, rally.call_index) == ([1], "landed_in", "near", 0)
    assert rally.winner_confidence == pytest.approx(hits[1].confidence, abs=0.01)


def test_racket_hand_is_voted_per_player_or_forced(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    auto = build_rally(rally_run(), homographies).players[0].racket_hand
    assert (auto.hand, auto.share, auto.votes, auto.source) == ("right", 1.0, 1, "auto")
    forced = build_rally(rally_run(), homographies, hitter=hitter_params(hand="left")).players[0].racket_hand
    assert (forced.hand, forced.source) == ("left", "manual")


def test_a_bystander_who_is_not_offered_does_not_get_hits(homographies, monkeypatch):
    run = rally_run()
    for r in run.records:                                   # P2 stands right on the shuttle path but is seen once
        r.players[2] = r.players[1]
    run.records[0].players[3] = run.records[0].players[1]
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 2)
    analysis = build_rally(run, homographies)
    assert {p.player_id for p in analysis.players} == {1, 2}
    assert {h.player_id for h in analysis.events["hits"] if h.valid} <= {1, 2}


def test_two_near_hits_in_a_rally_raise_the_alternation_warning(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    run = rally_run()
    run.records[26].measurement = HitMeasurement(26, (210, 210), 9.0, 45.0, False, 0, "physics", 5)
    run.shuttle_px[26] = (210, 210)
    codes = [w["code"] for w in build_rally(run, homographies).warnings]
    assert "alternation_break" in codes and "doubles_unsupported" not in codes


def opponent_hit_run():
    """rally_run, with the landing moved to frame 39 (in the near half) and a start at frame 26
    (previous detection at frame 24) far from the player's hand: the opponent returned the shuttle."""
    run = rally_run()
    run.records[32].call = None
    run.records[39].call = Call(39, "IN", "near", (11, 12), (2.0, 3.0), 0.8, False)
    run.records[26].measurement = HitMeasurement(26, (900, 900), 9.0, 45.0, False, 0, "physics", 5)
    run.shuttle_px[24], run.shuttle_px[26] = (850, 850), (900, 900)
    return run


def test_an_opponent_hit_is_inferred_in_singles_and_left_unknown_in_doubles(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    singles = build_rally(opponent_hit_run(), homographies)
    assert [h.hitter for h in singles.events["hits"] if h.valid] == ["near", "far"]
    assert (singles.events["rallies"][0].last_hitter, singles.events["rallies"][0].winner) == ("far", "far")
    assert "doubles_unsupported" not in [w["code"] for w in singles.warnings]

    doubles = build_rally(opponent_hit_run(), homographies, match_type="doubles")
    assert [h.hitter for h in doubles.events["hits"] if h.valid] == ["near", "unknown"]
    assert "doubles_unsupported" in [w["code"] for w in doubles.warnings]


def test_new_events_serialise_to_json(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    data = json.loads(json.dumps(build_rally(rally_run(), homographies).to_jsonable()))
    hit = data["events"]["hits"][1]
    assert hit["hitter"] == "near" and hit["window_frames"] == [13, 22] and hit["prev_pt_px"] == [100.0, 100.0]
    assert data["events"]["rallies"][0]["winner"] == "near"
    assert data["players"][0]["racket_hand"]["hand"] == "right"
    assert data["labels"]["hitter"]["far"] and data["labels"]["end_cause"]["landed_in"]


def test_movement_is_measured_per_player_after_the_preroll(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    movement_params = MovementParams(7.0, 0.5, 0.4, (0.5, 2.0, 4.0), 0.5, 2.0, 0.5, 0.75, 3.0, 0.6, 10.0, (2.2, 4.4))
    analysis = build_rally(rally_run(), homographies, source=ClipSource("v.mp4", 0, 10, 10), movement=movement_params)
    movement = analysis.players[0].movement
    assert (movement.first_frame_idx, movement.last_frame_idx) == (10, 39)
    assert movement.duration_s == pytest.approx(30 / 60) and movement.small_sample is True
    assert movement.zone_time_s["rear_left"] == pytest.approx(30 / 60)       # the fixture player stands at (2.0, 3.0)
    assert movement.distance_m == pytest.approx(0.0, abs=1e-6)
    data = json.loads(json.dumps(analysis.to_jsonable()))
    assert data["players"][0]["movement"]["zone_time_s"]["rear_left"] == pytest.approx(0.5)
    assert set(data["labels"]["zone"]) == set(data["players"][0]["movement"]["zone_time_s"])


def test_shots_are_classified_summarised_and_serialised(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    analysis = build_rally(rally_run(), homographies, source=ClipSource("v.mp4", 0, 0, 0))
    [shot] = [x for x in analysis.events["shots"] if x.hit_index == 1]
    assert (shot.hitter, shot.player_id, shot.frame_idx) == ("near", 1, analysis.events["hits"][1].contact_frame_idx)
    assert shot.type in ("serve", "smash", "drop", "clear", "lift", "net", "drive", "unknown")
    assert shot.features.landing_result == "IN" and shot.features.flight_s > 0
    summary = analysis.players[0].shot_summary
    assert summary.player_id == 1 and summary.n_shots == 1 and summary.by_type[0].n == 1
    data = json.loads(json.dumps(analysis.to_jsonable()))
    assert data["events"]["shots"][-1]["features"]["landing_result"] == "IN"
    assert data["players"][0]["shot_summary"]["by_type"][0]["share_pct"] == 100.0
    assert data["labels"]["shot_type"]["smash"] and data["labels"]["shot_reason"]["pose_missing"]


# ---- skeleton export (schema 2) ----------------------------------------------------------------

def test_skeleton_row_is_a_flat_list_of_integer_pixels_with_hidden_joints_marked():
    lm = np.zeros((33, 3), np.float32)
    lm[:, 0], lm[:, 1], lm[:, 2] = np.arange(33) + 0.4, np.arange(33) + 100.6, 0.9
    lm[5, 2] = 0.1                                                   # not visible
    lm[6, :2] = (-12.3, -4.0)                                        # visible but outside the frame
    row = skeleton_row(lm)
    assert row.shape == (66,) and row.dtype == np.int32
    assert list(row[:4]) == [0, 101, 1, 102]                         # rounded to the nearest pixel
    assert list(row[10:12]) == [HIDDEN_LANDMARK_PX, HIDDEN_LANDMARK_PX]
    assert list(row[12:14]) == [0, 0]                                # clipped to the frame, never the hidden marker


def test_the_hidden_threshold_is_read_from_config_at_call_time(monkeypatch):
    lm = np.zeros((33, 3), np.float32)
    lm[:, 2] = 0.45
    monkeypatch.setattr(config, "CLIP_SKELETON_MIN_SCORE", 0.4)
    assert (skeleton_row(lm) == HIDDEN_LANDMARK_PX).sum() == 0
    monkeypatch.setattr(config, "CLIP_SKELETON_MIN_SCORE", 0.5)
    assert (skeleton_row(lm) == HIDDEN_LANDMARK_PX).all()


def test_integer_arrays_serialise_without_a_per_element_walk_and_float_arrays_still_clean_nan():
    arr = np.arange(66, dtype=np.int32)
    out = to_jsonable({"a": arr})
    assert out["a"] == list(range(66)) and type(out["a"][0]) is int
    assert to_jsonable(np.array([1.5, np.nan])) == [1.5, None]


def test_the_exported_skeleton_is_much_smaller_than_the_old_nested_floats(homographies, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MIN_PLAYER_FRAMES", 1)
    text = json.dumps(build(make_run(), homographies).to_jsonable()["players"][0]["skeleton"], separators=(",", ":"))
    assert len(text) / 3 < 300          # three skeletons: well under the ~620 bytes each of schema 1
    assert build(make_run(), homographies).to_jsonable()["pose"]["hidden"] == HIDDEN_LANDMARK_PX
