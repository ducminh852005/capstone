from core import config
from core.smash import SmashDetector


def make_detector(**kw):
    """The tests feed detections 1-2 frames apart, so measure from the first one (default: >= 4 frames)."""
    kw.setdefault("min_dt_frames", 1)
    return SmashDetector(**kw)


def feed(det, frames, source="physics"):
    """frames: list of (frame_idx, pt, track_active, track_len). Returns non-None results."""
    out = []
    for idx, pt, active, length in frames:
        res = det.update(idx, pt, active, length, (0.0, 0.0), source)
        if res is not None:
            out.append(res)
    return out


def test_fast_downward_hit_is_smash():
    det = make_detector(speed_threshold=12, min_y=50, min_angle=10, max_angle=170)
    # new track at frame 100 (pt given), second detection 4 frames later 80 px below: 20 px/frame, 90 deg
    hits = feed(det, [(100, (500, 300), True, 0), (101, None, True, 1), (104, (500, 380), True, 4)])
    assert len(hits) == 1
    h = hits[0]
    assert h.is_smash and h.smash_number == 1
    assert h.frame_idx == 100 and h.pt == (500, 300)
    assert abs(h.speed - 20.0) < 1e-6 and abs(h.angle_deg - 90.0) < 1e-6
    assert det.count == 1


def test_slow_hit_is_measured_but_not_smash():
    det = make_detector(speed_threshold=12)
    hits = feed(det, [(10, (500, 300), True, 0), (12, (500, 310), True, 2)])
    assert len(hits) == 1 and not hits[0].is_smash and hits[0].smash_number == 0
    assert det.count == 0


def test_y_gate_rejects_hit_above_play_area():
    det = make_detector(speed_threshold=12, min_y=200)
    hits = feed(det, [(10, (500, 100), True, 0), (11, (500, 180), True, 1)])
    assert len(hits) == 1 and not hits[0].is_smash


def test_angle_gate_rejects_upward_and_horizontal():
    det = make_detector(speed_threshold=12, min_angle=10, max_angle=170)
    up = feed(det, [(10, (500, 300), True, 0), (11, (500, 250), True, 1)])         # -90 deg: upward
    assert not up[0].is_smash
    flat = feed(det, [(20, (500, 300), True, 0), (21, (540, 300), True, 1)])       # 0 deg: horizontal
    assert not flat[0].is_smash


def test_no_detection_on_start_frame_uses_track_pos():
    det = make_detector(speed_threshold=12, min_y=0)
    det.update(50, None, True, 0, (400.0, 200.0), "physics")
    hit = det.update(52, (400, 260), True, 2, (0.0, 0.0), "physics")
    assert hit is not None and hit.pt == (400, 200)
    assert abs(hit.speed - 30.0) < 1e-6


def test_track_lost_discards_pending_hit():
    det = make_detector(speed_threshold=12)
    feed(det, [(10, (500, 300), True, 0), (11, None, False, 0)])
    # a later detection must not be paired with the discarded start
    hits = feed(det, [(20, (500, 400), True, 3)])
    assert hits == []


def test_no_measurement_on_start_frame_itself():
    det = make_detector()
    assert det.update(10, (500, 300), True, 0, (0.0, 0.0), "physics") is None


def test_defaults_read_config_at_call_time(monkeypatch):
    det = make_detector()
    monkeypatch.setattr(config, "SMASH_SPEED_THRESHOLD", 5.0)
    assert det.speed_threshold == 5.0
    monkeypatch.setattr(config, "SMASH_SPEED_THRESHOLD", 50.0)
    assert det.speed_threshold == 50.0
    assert make_detector(speed_threshold=7.0).speed_threshold == 7.0


def test_new_track_is_measured_but_not_a_smash_by_default():
    det = make_detector(speed_threshold=12, min_y=50)
    frames = [(1738, (1502, 447), True, 0), (1746, (1538, 547), True, 8)]   # fast, downward, but not a hit
    (h,) = feed(det, frames, source="new")
    assert h.source == "new" and h.dt_frames == 8
    assert h.speed > 12 and not h.is_smash and det.count == 0
    (h,) = feed(det, [(1754, (500, 300), True, 0), (1762, (540, 400), True, 8)], source="physics")
    assert h.is_smash and h.smash_number == 1


def test_new_track_can_count_when_the_requirement_is_off(monkeypatch):
    monkeypatch.setattr(config, "SMASH_REQUIRE_PHYSICS_HIT", False)
    det = make_detector(speed_threshold=12, min_y=50)
    (h,) = feed(det, [(10, (500, 300), True, 0), (11, (500, 400), True, 1)], source="new")
    assert h.is_smash and det.count == 1


def test_source_of_a_hit_is_taken_on_the_start_frame():
    det = make_detector(speed_threshold=12, min_y=50)
    det.update(10, (500, 300), True, 0, (0.0, 0.0), "physics")
    # a later call carrying a different source must not relabel the pending measurement
    h = det.update(12, (500, 400), True, 2, (0.0, 0.0), "new")
    assert h.source == "physics" and h.is_smash


def test_speed_is_measured_to_the_first_detection_at_least_min_dt_frames_later():
    det = SmashDetector(speed_threshold=12, min_y=50, min_dt_frames=4)
    frames = [(100, (500, 300), True, 0), (101, (503, 322), True, 1), (102, (498, 344), True, 2),
              (103, (502, 366), True, 3), (104, (500, 388), True, 4)]       # a detection on every frame, +-3 px noise
    (h,) = feed(det, frames)
    assert h.dt_frames == 4 and abs(h.speed - 22.0) < 1e-6 and h.is_smash    # not 3.0/1 from the noisy first step


def test_min_dt_default_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "SMASH_MIN_DT_FRAMES", 6)
    det = SmashDetector(speed_threshold=12, min_y=50)
    frames = [(0, (500, 300), True, 0), (5, (500, 400), True, 5), (6, (500, 420), True, 6)]
    (h,) = feed(det, frames)
    assert h.dt_frames == 6
