from types import SimpleNamespace

import numpy as np
import pytest

from core import gap_fill
from core.clip_pipeline import ClipRun, PoseSchedule, iter_frame_records, run_clip
from core.player_tracker import PlayerObs
from core.smash import HitMeasurement
from core.umpire import Call

FRAME = np.zeros((4, 4, 3), np.uint8)


def frames(n):
    return [(i, FRAME) for i in range(n)]


class FakeTracker:
    def __init__(self):
        self.last_boxes = np.asarray([(10.0, 20.0, 50.0, 120.0)], np.float32)
        self.last_ids = np.asarray([7], np.int64)
        self.calls = []
        self.pose_every_log = []                     # (frame processed next, cadence) of every set_pose_every call

    def set_pose_every(self, every):
        self.pose_every_log.append((len(self.calls), every))

    def process(self, frame, frame_idx, H, H_inv):
        self.calls.append((frame_idx, H, H_inv))
        return {1: PlayerObs(1, 7, (10.0, 20.0, 50.0, 120.0), (30.0, 120.0), (2.0, 3.0), "foot")}

    def non_player_boxes(self, players):
        return []


class FakeDetector:
    """Replays a scripted (point, track_active, track_len, flight_id, start_source) per call."""

    def __init__(self, script, frame_lags=None):
        self.script = script
        self.frame_lags = frame_lags or [0] * len(script)
        self.trajectory = []
        self.detect_kwargs = []
        self.kf = SimpleNamespace(x=np.zeros(6))
        self.filled_with = None
        self._i = -1
        self.frame_lag = 0
        self.track_active = False
        self.track_len = 0
        self.flight_id = 0
        self.start_source = None

    def detect(self, frame, roi=None, exclude_boxes=None, player_boxes=None):
        self._i += 1
        pt, self.track_active, self.track_len, self.flight_id, self.start_source = self.script[self._i]
        self.frame_lag = self.frame_lags[self._i]
        self.trajectory.append(pt)
        self.detect_kwargs.append((roi, exclude_boxes, player_boxes))
        return pt, None

    def fill_gaps(self, allowed=None):
        self.filled_with = allowed
        return [gap_fill.FilledPoint(1, (5.0, 6.0), "stride")]


class FakeSmash:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def update(self, frame_idx, pt, track_active, track_len, track_pos, start_source):
        self.calls.append((frame_idx, pt, track_active, track_len, start_source))
        return self.results.get(frame_idx)


class FakeUmpire:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def update(self, frame_idx, pt, track_active, people_boxes=None, flight_id=None):
        self.calls.append((frame_idx, pt, track_active, flight_id, len(people_boxes)))
        return self.results.get(frame_idx)


def call_at(frame_idx):
    return Call(frame_idx, "IN", "near", (1, 2), (3.0, 4.0), 0.5, False)


SCRIPT = [
    (None, False, 0, 0, None),
    ((100, 200), True, 0, 1, "new"),
    ((110, 210), True, 1, 1, "new"),
    ((120, 220), True, 0, 2, "physics"),
]


def test_one_record_per_frame_with_player_and_shuttle_state():
    tracker, detector = FakeTracker(), FakeDetector(SCRIPT)
    records = list(iter_frame_records(frames(4), tracker, detector, FakeSmash(), FakeUmpire(), roi=(0, 0, 4, 4)))
    assert [r.frame_idx for r in records] == [0, 1, 2, 3]
    r = records[3]
    assert r.shuttle_pt_px == (120, 220) and r.shuttle_frame_idx == 3
    assert (r.track_active, r.track_len, r.flight_id, r.start_source) == (True, 0, 2, "physics")
    assert r.players[1].track_id == 7
    assert records[0].shuttle_pt_px is None and records[0].track_active is False


def test_detector_gets_the_roi_and_player_boxes_and_tracker_gets_the_homographies():
    tracker, detector = FakeTracker(), FakeDetector(SCRIPT)
    H, H_inv = np.eye(3), 2 * np.eye(3)
    list(iter_frame_records(frames(1), tracker, detector, FakeSmash(), FakeUmpire(), roi=(1, 2, 3, 4), H=H, H_inv=H_inv))
    roi, exclude, player_boxes = detector.detect_kwargs[0]
    assert roi == (1, 2, 3, 4) and exclude == [] and player_boxes == [(10.0, 20.0, 50.0, 120.0)]
    assert tracker.calls[0][1] is H and tracker.calls[0][2] is H_inv


def test_umpire_gets_the_flight_id_and_smash_the_start_source_of_the_same_call():
    detector, smash, umpire = FakeDetector(SCRIPT), FakeSmash(), FakeUmpire()
    list(iter_frame_records(frames(4), FakeTracker(), detector, smash, umpire, roi=None))
    # a hit restarts the track within one detect() call: track_active never drops, flight_id changes
    assert [c[3] for c in umpire.calls] == [0, 1, 1, 2]
    assert [c[2] for c in umpire.calls] == [False, True, True, True]
    assert [(c[3], c[4]) for c in smash.calls] == [(0, None), (0, "new"), (1, "new"), (0, "physics")]
    assert all(c[4] == 1 for c in umpire.calls)         # the person boxes of the tracker


def test_measurement_and_call_are_attached_to_the_frame_that_returned_them():
    m = HitMeasurement(1, (100, 200), 15.0, 80.0, True, 1, "physics", 5)
    detector = FakeDetector(SCRIPT)
    records = list(iter_frame_records(frames(4), FakeTracker(), detector, FakeSmash({2: m}), FakeUmpire({3: call_at(1)}),
                                      roi=None))
    assert records[2].measurement is m and records[2].call is None
    assert records[3].call.frame_idx == 1 and records[3].measurement is None


def test_a_lagged_detector_is_rejected():
    detector = FakeDetector(SCRIPT, frame_lags=[0, 0, 7, 7])
    with pytest.raises(NotImplementedError, match="frame_lag=7"):
        list(iter_frame_records(frames(4), FakeTracker(), detector, FakeSmash(), FakeUmpire(), roi=None))


def test_run_collects_trajectory_and_trail_and_orders_calls_by_contact_frame():
    detector = FakeDetector(SCRIPT)
    # a Call is emitted several frames after its contact: arrival order is not contact order
    umpire = FakeUmpire({2: call_at(1), 3: call_at(0)})
    m_late = HitMeasurement(3, (1, 1), 1.0, 1.0, False, 0, "physics", 4)
    m_early = HitMeasurement(1, (1, 1), 1.0, 1.0, False, 0, "new", 4)
    run = run_clip(frames(4), FakeTracker(), detector, FakeSmash({1: m_late, 2: m_early}), umpire, roi=None)
    assert isinstance(run, ClipRun) and run.n_frames == 4
    assert run.shuttle_px == [None, (100, 200), (110, 210), (120, 220)]
    assert run.filled == [gap_fill.FilledPoint(1, (5.0, 6.0), "stride")]
    assert callable(detector.filled_with)                # plausibility check handed to fill_gaps
    assert [c.frame_idx for c in run.calls] == [0, 1]
    assert [m.frame_idx for m in run.measurements] == [1, 3]


def test_the_pose_cadence_is_sparse_in_the_preroll_and_changes_only_at_the_boundary():
    tracker = FakeTracker()
    schedule = PoseSchedule(preroll_frames=2, preroll_every=3, every=1)
    list(iter_frame_records(frames(4), tracker, FakeDetector(SCRIPT), FakeSmash(), FakeUmpire(), roi=None,
                            pose_schedule=schedule))
    assert tracker.pose_every_log == [(0, 3), (2, 1)]


def test_without_a_schedule_the_tracker_cadence_is_left_alone():
    tracker = FakeTracker()
    list(iter_frame_records(frames(4), tracker, FakeDetector(SCRIPT), FakeSmash(), FakeUmpire(), roi=None))
    assert tracker.pose_every_log == []


def test_pose_schedule_picks_the_cadence_by_frame():
    s = PoseSchedule(preroll_frames=120, preroll_every=3, every=1)
    assert [s.every_at(f) for f in (0, 119, 120, 400)] == [3, 3, 1, 1]


def test_the_analysis_modules_do_not_pull_in_the_model_libraries():
    """Importing the analysis must not load torch, MediaPipe or YOLO: it costs seconds and is not needed."""
    import subprocess
    import sys
    from pathlib import Path
    backend = Path(__file__).resolve().parent.parent
    code = ("import sys; sys.path.insert(0, %r); import core.clip_analysis, core.clip_pipeline, core.clip_report, "
            "core.hit_events, core.shots, core.movement; "
            "print([m for m in ('torch', 'ultralytics', 'mediapipe') if m in sys.modules])" % str(backend))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
