from core import config
from core.smash import SmashDetector


def feed(det, frames):
    """frames: list of (frame_idx, pt, track_active, track_len). Returns non-None results."""
    out = []
    for idx, pt, active, length in frames:
        res = det.update(idx, pt, active, length, track_pos=(0.0, 0.0))
        if res is not None:
            out.append(res)
    return out


def test_fast_downward_hit_is_smash():
    det = SmashDetector(speed_threshold=12, min_y=50, min_angle=10, max_angle=170)
    # new track at frame 100 (pt given), second detection 4 frames later 80 px below: 20 px/frame, 90 deg
    hits = feed(det, [(100, (500, 300), True, 0), (101, None, True, 1), (104, (500, 380), True, 4)])
    assert len(hits) == 1
    h = hits[0]
    assert h.is_smash and h.smash_number == 1
    assert h.frame_idx == 100 and h.pt == (500, 300)
    assert abs(h.speed - 20.0) < 1e-6 and abs(h.angle_deg - 90.0) < 1e-6
    assert det.count == 1


def test_slow_hit_is_measured_but_not_smash():
    det = SmashDetector(speed_threshold=12)
    hits = feed(det, [(10, (500, 300), True, 0), (12, (500, 310), True, 2)])
    assert len(hits) == 1 and not hits[0].is_smash and hits[0].smash_number == 0
    assert det.count == 0


def test_y_gate_rejects_hit_above_play_area():
    det = SmashDetector(speed_threshold=12, min_y=200)
    hits = feed(det, [(10, (500, 100), True, 0), (11, (500, 180), True, 1)])
    assert len(hits) == 1 and not hits[0].is_smash


def test_angle_gate_rejects_upward_and_horizontal():
    det = SmashDetector(speed_threshold=12, min_angle=10, max_angle=170)
    up = feed(det, [(10, (500, 300), True, 0), (11, (500, 250), True, 1)])         # -90 deg: upward
    assert not up[0].is_smash
    flat = feed(det, [(20, (500, 300), True, 0), (21, (540, 300), True, 1)])       # 0 deg: horizontal
    assert not flat[0].is_smash


def test_no_detection_on_start_frame_uses_track_pos():
    det = SmashDetector(speed_threshold=12, min_y=0)
    det.update(50, None, True, 0, track_pos=(400.0, 200.0))
    hit = det.update(52, (400, 260), True, 2, track_pos=(0.0, 0.0))
    assert hit is not None and hit.pt == (400, 200)
    assert abs(hit.speed - 30.0) < 1e-6


def test_track_lost_discards_pending_hit():
    det = SmashDetector(speed_threshold=12)
    feed(det, [(10, (500, 300), True, 0), (11, None, False, 0)])
    # a later detection must not be paired with the discarded start
    hits = feed(det, [(20, (500, 400), True, 3)])
    assert hits == []


def test_no_measurement_on_start_frame_itself():
    det = SmashDetector()
    assert det.update(10, (500, 300), True, 0, track_pos=(0.0, 0.0)) is None


def test_defaults_read_config_at_call_time(monkeypatch):
    det = SmashDetector()
    monkeypatch.setattr(config, "SMASH_SPEED_THRESHOLD", 5.0)
    assert det.speed_threshold == 5.0
    monkeypatch.setattr(config, "SMASH_SPEED_THRESHOLD", 50.0)
    assert det.speed_threshold == 50.0
    assert SmashDetector(speed_threshold=7.0).speed_threshold == 7.0
