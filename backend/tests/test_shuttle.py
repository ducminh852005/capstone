import numpy as np
import cv2

import pytest

from core.shuttle_tracker import ShuttleDetector, ShuttleTrajectoryProcessor

H, W = 1080, 1920
ROI = (100, 50, 1800, 1080)


def truth(t):
    """Parabolic flight in image pixels (gravity pulls y down by 1 px/frame^2)."""
    return 400.0 + 12.0 * t, 700.0 - 25.0 * t + 0.5 * t * t


def render(points, bg=40):
    frame = np.full((H, W, 3), bg, np.uint8)
    for x, y in points:
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)
    return frame


@pytest.mark.slow
def test_tracks_parabola_through_distractors_and_dropouts():
    rng = np.random.default_rng(1)
    det = ShuttleDetector(backend="cv")
    dropped = set(range(30, 34))
    errors, false_hits = [], 0
    for t in range(50):
        shuttle = truth(t)
        blobs = [] if t in dropped else [shuttle]
        if t % 2 == 0:
            blobs.append((1500, 300))                   # flickering static light
        for _ in range(3):                              # random clutter far from the shuttle
            while True:
                p = (rng.uniform(150, 1750), rng.uniform(100, 1050))
                if np.hypot(p[0] - shuttle[0], p[1] - shuttle[1]) > 150:
                    break
            blobs.append(p)
        pt, mask = det.detect(render(blobs), roi=ROI)
        assert mask.shape == (ROI[3] - ROI[1], ROI[2] - ROI[0])
        if pt is None:
            continue
        err = np.hypot(pt[0] - shuttle[0], pt[1] - shuttle[1])
        if t in dropped or err > 5:
            false_hits += 1
        else:
            errors.append(err)
    assert false_hits == 0
    assert len(errors) >= 40           # tracked on almost every visible frame, including after the dropout
    assert np.mean(errors) < 3.0
    # trajectory is stored in full-frame coordinates (not ROI-local)
    last = det.trajectory[49]
    assert np.hypot(last[0] - truth(49)[0], last[1] - truth(49)[1]) < 5


@pytest.mark.slow
def test_static_flicker_never_starts_a_track():
    det = ShuttleDetector(backend="cv")
    for t in range(40):
        blobs = [(900, 400)] if t % 2 == 0 else []
        blobs += [(300, 800)] if t % 3 == 0 else []
        pt, _ = det.detect(render(blobs))
        assert pt is None
    assert not det.track_active


@pytest.mark.slow
def test_player_body_blobs_are_ignored():
    det = ShuttleDetector(backend="cv")
    box = (800, 300, 1000, 800)
    for t in range(20):
        # a limb moving consistently inside the lower body region of the player
        pt, _ = det.detect(render([(900, 600 - 5 * t)]), player_boxes=[box])
        assert pt is None


@pytest.mark.slow
def test_slow_track_does_not_slide_onto_player_legs():
    det = ShuttleDetector(backend="cv")
    box = (800, 300, 1000, 800)   # lower 2/3 starts at y = 466
    for t in range(40):
        y = 250 + 6 * t           # racket-like blob moving slowly down into the body
        pt, _ = det.detect(render([(900, y)]), player_boxes=[box])
        if y > 480:
            assert pt is None


@pytest.mark.slow
def test_spectator_boxes_are_ignored():
    det = ShuttleDetector(backend="cv")
    box = (100, 100, 400, 900)
    for t in range(20):
        pt, _ = det.detect(render([(250, 200 + 8 * t)]), exclude_boxes=[box])
        assert pt is None


@pytest.mark.slow
def test_overlong_track_is_ended():
    det = ShuttleDetector(backend="cv", max_track_len=30)
    longest = 0
    for t in range(60):
        det.detect(render([(300 + 10 * t, 500)]))
        longest = max(longest, det.track_len)
    assert det.track_active and longest <= 31


def test_fill_missing_parabola_gap():
    proc = ShuttleTrajectoryProcessor()
    traj = [truth(t) for t in range(30)]
    gap = range(12, 17)
    holey = [None if t in gap else (round(p[0]), round(p[1])) for t, p in enumerate(traj)]
    filled = proc.fill_missing_trajectory(holey)
    for t in gap:
        assert np.hypot(filled[t][0] - traj[t][0], filled[t][1] - traj[t][1]) < 2.0


def test_fill_missing_gap_with_hit_falls_back_to_linear():
    proc = ShuttleTrajectoryProcessor()
    # moving right, gap, then moving left (hit inside the gap)
    traj = [(100 + 20 * t, 500 - 10 * t) for t in range(10)] + [None] * 3 + [(260 - 20 * k, 380 + 5 * k) for k in range(10)]
    filled = proc.fill_missing_trajectory(traj)
    a, b = np.array(traj[9], float), np.array(traj[13], float)
    for k, t in enumerate(range(10, 13), start=1):
        expected = a + (b - a) * k / 4
        assert np.hypot(*(np.array(filled[t]) - expected)) <= 1.0


def test_fill_missing_gap_with_vertical_reversal_falls_back_to_linear():
    proc = ShuttleTrajectoryProcessor()
    # x keeps drifting the same way on both sides (no horizontal flip), but the shuttle
    # was going up before the gap and comes out diving steeply after it -- a straight
    # net kill hit inside the occlusion, not inertia carrying it through.
    traj = [(100 + 20 * t, 700 - 30 * t) for t in range(10)]
    y13 = traj[9][1] - 30
    traj += [None] * 3
    traj += [(100 + 20 * t, y13 + 40 * (t - 13)) for t in range(13, 23)]
    filled = proc.fill_missing_trajectory(traj)
    a, b = np.array(traj[9], float), np.array(traj[13], float)
    for k, t in enumerate(range(10, 13), start=1):
        expected = a + (b - a) * k / 4
        assert np.hypot(*(np.array(filled[t]) - expected)) <= 1.0


def test_fill_missing_leaves_long_gaps():
    proc = ShuttleTrajectoryProcessor(max_gap_frames=15)
    traj = [truth(t) for t in range(5)] + [None] * 20 + [truth(t) for t in range(25, 30)]
    filled = proc.fill_missing_trajectory(traj)
    assert all(p is None for p in filled[5:25])


# --- physics filter (racket-hit detection) ---------------------------------------------

def _tracking_detector(speed=10.0):
    """CV detector with a running track moving right at `speed` px/frame, last seen at (120, 100)."""
    det = ShuttleDetector(backend="cv")
    det.kf.init((120 - 2 * speed, 100), (120 - speed, 100), (120, 100))
    det.track_active = True
    det.track_len = 3
    det.trajectory = [(int(120 - speed), 100), (120, 100)]
    return det


def test_physics_accepts_continued_flight_and_rejects_hits():
    det = _tracking_detector()
    cands = np.array([
        [130.0, 100.0],     # keeps going: fine
        [110.0, 100.0],     # reversed direction: a racket hit
        [200.0, 100.0],     # 8x speed jump: a racket hit
        [120.5, 100.0],     # 20x deceleration: not the same shuttle
    ])
    assert det._physics_reject(cands).tolist() == [False, True, True, True]


def test_physics_does_nothing_for_slow_or_new_tracks():
    slow = _tracking_detector(speed=1.0)
    assert not slow._physics_reject(np.array([[100.0, 100.0]])).any()   # reversal ignored: too slow to read
    new = _tracking_detector()
    new.track_len = 0       # only the start point of the track has been seen: no direction yet
    assert not new._physics_reject(np.array([[100.0, 100.0]])).any()


def test_hit_inside_gate_ends_the_track(monkeypatch):
    det = _tracking_detector()
    # a reversal candidate 10 px behind the shuttle: inside the Euclidean gate, but unphysical
    monkeypatch.setattr(det._source, "generate",
                        lambda frame, roi: (np.array([[110.0, 100.0]]), np.ones(1), np.zeros((4, 4), np.uint8)))
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert pt is None
    assert not det.track_active


def test_cv_coast_comes_from_config():
    from core import config
    assert ShuttleDetector(backend="cv").max_coast == config.MAX_COAST_CV
    assert ShuttleDetector(backend="cv", max_coast=3).max_coast == 3


# --- flight bookkeeping: flight_id / start_source ---------------------------------------

def _gen(cands, confs):
    return lambda frame, roi: (np.array(cands, dtype=float), np.array(confs, dtype=float),
                               np.zeros((4, 4), np.uint8))


def _tracknet_like_detector(speed=10.0):
    det = _tracking_detector(speed)
    det.simple_init = True          # single-point restart, as for the tracknet backends
    det.flight_id = 1
    return det


def test_hit_restarts_track_in_place_as_a_new_flight(monkeypatch):
    det = _tracknet_like_detector()
    monkeypatch.setattr(det._source, "generate", _gen([[110.0, 100.0]], [0.9]))   # reversal = hit
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert det.track_active and det.track_len == 0          # no edge in track_active...
    assert det.flight_id == 2 and det.start_source == "physics"   # ...but a new flight
    assert pt == (110, 100)


def test_low_confidence_hit_candidate_still_restarts(monkeypatch):
    det = _tracknet_like_detector()
    conf = det.init_min_confidence - 0.05
    assert 0.5 <= conf < det.init_min_confidence
    monkeypatch.setattr(det._source, "generate", _gen([[110.0, 100.0]], [conf]))
    det.detect(np.zeros((10, 10, 3), np.uint8))
    assert det.flight_id == 2 and det.start_source == "physics"


def test_restart_uses_the_candidate_that_broke_physics_not_the_most_confident(monkeypatch):
    det = _tracknet_like_detector()
    # (400, 400) is far outside the gate and most confident; (110, 100) is the hit
    monkeypatch.setattr(det._source, "generate", _gen([[400.0, 400.0], [110.0, 100.0]], [0.99, 0.7]))
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert pt == (110, 100)


def test_hit_candidate_on_player_body_still_restarts(monkeypatch):
    det = _tracknet_like_detector()
    monkeypatch.setattr(det._source, "generate", _gen([[110.0, 100.0]], [0.9]))
    body = [(90, 0, 130, 130)]      # the candidate lies in the lower 2/3 of this box
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8), player_boxes=body)
    assert pt == (110, 100) and det.start_source == "physics"


def test_fresh_track_is_a_new_flight_not_a_hit(monkeypatch):
    det = ShuttleDetector(backend="cv", simple_init=True)
    assert det.flight_id == 0 and det.start_source is None
    monkeypatch.setattr(det._source, "generate", _gen([[300.0, 300.0]], [0.9]))
    det.detect(np.zeros((10, 10, 3), np.uint8))
    assert det.track_active and det.flight_id == 1 and det.start_source == "new"


def test_continued_flight_keeps_its_flight_id(monkeypatch):
    det = _tracknet_like_detector()
    monkeypatch.setattr(det._source, "generate", _gen([[130.0, 100.0]], [0.9]))   # keeps going
    det.detect(np.zeros((10, 10, 3), np.uint8))
    assert det.flight_id == 1 and det.track_len == 4


def test_unphysical_distractor_does_not_break_a_track_that_has_a_good_candidate(monkeypatch):
    det = _tracknet_like_detector()
    # (130, 100) continues the flight; (110, 100) is inside the gate too but reverses direction
    monkeypatch.setattr(det._source, "generate", _gen([[130.0, 100.0], [110.0, 100.0]], [0.6, 0.99]))
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert pt == (130, 100)
    assert det.flight_id == 1 and det.track_len == 4       # same flight, no hit


def test_shuttle_coming_to_rest_restarts_the_track_but_is_not_a_hit(monkeypatch):
    det = _tracknet_like_detector()
    # 20x deceleration to 0.5 px/frame: unphysical for the running track, but it is a stop
    monkeypatch.setattr(det._source, "generate", _gen([[120.5, 100.0]], [0.9]))
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert pt == (120, 100) and det.track_active and det.track_len == 0
    assert det.start_source == "stop"
    assert det.flight_id == 1           # same flight: nothing new started


# --- physics uses the measured velocity, not the Kalman one ---------------------------------

def _falling_track_detector():
    """
    Track that measured a shuttle falling right/down, (100,100) at call 0 and (145,142) at call 8,
    and has coasted 7 calls since: the next detection arrives 8 calls after the last one.
    """
    det = ShuttleDetector(backend="cv")
    det.simple_init = True
    det.kf.init((100.0, 100.0), (100.0, 100.0), (145.0, 142.0))
    det.track_active, det.track_len, det.flight_id = True, 15, 1
    det.trajectory = [(100, 100)] + [None] * 7 + [(145, 142)] + [None] * 7
    return det


def test_wrong_kalman_velocity_does_not_break_a_landing(monkeypatch):
    det = _falling_track_detector()
    det.kf.x[2:4] = (-14.0, -2.4)      # what the Kalman filter really reported on the footage: the wrong way
    monkeypatch.setattr(det._source, "generate", _gen([[173.0, 164.0]], [0.9]))   # slower, same direction
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert pt == (173, 164) and det.flight_id == 1 and det.track_len == 16


def test_direction_is_judged_against_the_last_two_detections(monkeypatch):
    det = _falling_track_detector()
    det.kf.x[2:4] = (5.0, 5.0)         # a Kalman velocity that would agree with the bad candidate
    det.min_gate_px = 200.0            # wide enough (stride-scaled in real use) for the candidate to be in the gate
    monkeypatch.setattr(det._source, "generate", _gen([[118.0, 118.0]], [0.9]))   # doubles back up-left
    pt, _ = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert det.flight_id == 2 and det.start_source == "physics"


def test_chord_velocity_ignores_points_of_earlier_tracks():
    det = _falling_track_detector()
    det.track_len = 0                  # the current track has only its start point in the trajectory
    assert det._chord_velocity() is None
    det.track_len = 15
    v = det._chord_velocity()
    assert v is not None and np.allclose(v, np.array([45.0, 42.0]) / 8.0)


def test_draw_trajectory_can_draw_in_place():
    det = ShuttleDetector(backend="cv")
    det.trajectory = [(10, 10), (20, 20), (30, 25), (40, 25)]
    frame = np.zeros((60, 60, 3), np.uint8)
    out = det.draw_trajectory(frame)
    assert out is not frame and frame.sum() == 0 and out.sum() > 0
    same = det.draw_trajectory(frame, copy=False)
    assert same is frame and frame.sum() > 0
    assert np.array_equal(same, out)               # same picture either way


# --- lagged (all_heatmaps) sources: the tracker steps once per replayed frame ---------------------

class LaggedSource:
    """Stands in for TrackNetCandidateSource(all_heatmaps=True): replays scripted frames `lag` calls late."""

    lagged = True
    frames_per_detection = 1
    max_frames_between_detections = 1
    batch_stride = 8
    idle_stride = 8
    seq_len = 8

    def __init__(self, script, lag=7):
        self.script, self.lag, self.calls = script, lag, 0     # script: {frame: [(x, y, conf), ...]}
        self.emitted_call_idx = None

    def generate(self, frame, roi):
        call, self.calls = self.calls, self.calls + 1
        frame_of = call - self.lag
        mask = np.zeros((4, 4), np.uint8)
        if frame_of < 0:
            self.emitted_call_idx = None
            return np.empty((0, 2)), np.empty((0,)), mask
        self.emitted_call_idx = frame_of
        rows = self.script.get(frame_of, [])
        cands = np.array([r[:2] for r in rows], float).reshape(-1, 2)
        return cands, np.array([r[2] for r in rows], float), mask


def lagged_detector(script, lag=7):
    det = ShuttleDetector(backend="cv", simple_init=True)
    det._source = LaggedSource(script, lag)
    det._lagged = True
    det._chord_min_frames = 4
    det._stop_baseline_frames = 2
    det.min_gate_px = 15.0
    det.max_coast = 12
    return det


BLANK = np.zeros((10, 10, 3), np.uint8)


def test_lagged_detector_does_not_move_until_the_first_replay_and_reports_the_lag():
    det = lagged_detector({0: [(100.0, 100.0, 0.9)]})
    for _ in range(7):
        pt, _ = det.detect(BLANK)
        assert pt is None and det.trajectory == [] and det.frame_lag == 0 and not det.track_active
    pt, _ = det.detect(BLANK)                               # call 7 replays frame 0
    assert pt == (100, 100) and det.track_active
    assert det.frame_lag == 7 and det.trajectory_first_call == 0 and len(det.trajectory) == 1


def test_lagged_detector_follows_a_flight_one_step_per_replayed_frame():
    script = {f: [(100.0 + 10 * f, 300.0 - 4 * f, 0.9)] for f in range(30)}
    det = lagged_detector(script)
    pts = [det.detect(BLANK)[0] for _ in range(37)]         # 30 replayed frames + 7 warm-up calls
    assert pts[:7] == [None] * 7
    assert pts[7:] == [(100 + 10 * f, 300 - 4 * f) for f in range(30)]
    assert len(det.trajectory) == 30 and det.flight_id == 1 and det.track_len == 29


def test_lagged_detector_uses_the_player_boxes_of_the_replayed_frame():
    script = {0: [(500.0, 500.0, 0.9)]}
    body = [(450, 300, 550, 600)]           # the candidate sits in the legs/torso zone of this box
    on_frame_0 = lagged_detector(script)
    on_frame_0.detect(BLANK, player_boxes=body)             # call 0: the player is there when frame 0 is shot
    for _ in range(6):
        on_frame_0.detect(BLANK, player_boxes=None)
    pt, _ = on_frame_0.detect(BLANK, player_boxes=None)     # call 7 replays frame 0: it was on the body
    assert pt is None and not on_frame_0.track_active

    only_now = lagged_detector(script)
    for _ in range(7):
        only_now.detect(BLANK, player_boxes=None)
    pt, _ = only_now.detect(BLANK, player_boxes=body)       # a box that appeared only at call 7 is irrelevant
    assert pt == (500, 500) and only_now.track_active


def test_lagged_detector_still_recognises_a_hit_from_per_frame_detections():
    right = {f: [(100.0 + 10 * f, 300.0, 0.9)] for f in range(12)}
    left = {f: [(210.0 - 10 * (f - 11), 300.0, 0.9)] for f in range(12, 24)}      # reverses after frame 11
    det = lagged_detector({**right, **left})
    for _ in range(31):
        det.detect(BLANK)
    assert det.flight_id == 2 and det.start_source == "physics"
    assert len(det.trajectory) == 24


def test_lag_defaults_to_zero_for_a_normal_source():
    det = ShuttleDetector(backend="cv")
    assert det.frame_lag == 0 and det.trajectory_first_call == 0 and not det._lagged


def test_detector_strides_property():
    assert ShuttleDetector(backend="cv").strides is None
    det = ShuttleDetector(backend="cv")
    det._source = LaggedSource({})
    assert det.strides == (8, 8)
