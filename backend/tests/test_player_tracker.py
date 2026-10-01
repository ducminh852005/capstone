import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core import config
from core.player_tracker import PlayerObs, PlayerTracker


def bare_tracker(boxes, ids):
    """A PlayerTracker without models (only the bookkeeping methods are used)."""
    t = object.__new__(PlayerTracker)
    t.last_boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    t.last_ids = np.asarray(ids, np.int64)
    return t


def obs(pid, tid, bbox):
    return PlayerObs(pid, tid, tuple(bbox), (0.0, 0.0), (0.0, 0.0), "bbox")


def test_non_player_boxes_are_the_boxes_of_the_other_tracks_in_order():
    boxes = [(10, 10, 50, 100), (200, 20, 260, 140), (400, 30, 450, 120)]
    t = bare_tracker(boxes, [3, 7, 9])
    players = {1: obs(1, 7, boxes[1])}
    assert t.non_player_boxes(players) == [tuple(np.float32(v) for v in boxes[0]),
                                           tuple(np.float32(v) for v in boxes[2])]
    assert t.non_player_boxes({}) == [tuple(np.float32(v) for v in b) for b in boxes]
    assert bare_tracker([], []).non_player_boxes(players) == []


def test_non_player_boxes_survive_two_tracks_with_identical_boxes():
    same = (10, 10, 50, 100)
    t = bare_tracker([same, same], [1, 2])
    # only the selected track is excluded; the twin is still reported as a non-player
    assert len(t.non_player_boxes({1: obs(1, 1, same)})) == 1


def test_draw_tracking_copies_by_default_and_can_draw_in_place():
    t = bare_tracker([], [])
    frame = np.zeros((200, 300, 3), np.uint8)
    players = {1: obs(1, 5, (20, 40, 120, 180))}
    out = t.draw_tracking(frame, players)
    assert out is not frame and frame.sum() == 0 and out.sum() > 0
    same = t.draw_tracking(frame, players, copy=False)
    assert same is frame and frame.sum() > 0


def fake_result(rows):
    data = torch.tensor(rows, dtype=torch.float32).reshape(-1, 7)
    return SimpleNamespace(boxes=SimpleNamespace(data=data, id=data[:, 4] if len(rows) else None))


def test_detect_reads_boxes_and_ids_from_one_tensor_and_applies_the_roi_offset():
    t = bare_tracker([], [])
    rows = [[10, 20, 60, 120, 4, 0.9, 0], [200, 30, 260, 150, 8, 0.8, 0]]
    t.track_frame = lambda frame, persist=True: fake_result(rows)
    frame = np.zeros((300, 500, 3), np.uint8)
    boxes, ids = t.detect(frame, roi=(100, 50, 400, 250))
    assert ids.tolist() == [4, 8] and ids.dtype == np.int64
    assert boxes.tolist() == [[110, 70, 160, 170], [300, 80, 360, 200]]
    # the network's own tensor must not be modified by the offset
    boxes_full, _ = t.detect(frame, roi=None)
    assert boxes_full.tolist() == [[10, 20, 60, 120], [200, 30, 260, 150]]


def test_detect_without_tracks_returns_empty_arrays():
    t = bare_tracker([], [])
    t.track_frame = lambda frame, persist=True: SimpleNamespace(
        boxes=SimpleNamespace(data=torch.zeros((0, 6)), id=None))
    boxes, ids = t.detect(np.zeros((100, 100, 3), np.uint8))
    assert boxes.shape == (0, 4) and ids.shape == (0,)


def test_yolo_runs_in_fp32_by_default():
    assert config.PLAYER_YOLO_HALF is False
    weights = config.PLAYER_YOLO_MODEL_PATH
    pose = config.MODELS_DIR / "pose_landmarker_lite.task"
    if not (os.path.exists(weights) and os.path.exists(pose)):
        pytest.skip("YOLO / MediaPipe weights not downloaded")
    assert PlayerTracker().half is False
    assert PlayerTracker(half=True).half is True
