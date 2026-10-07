import numpy as np
import pytest

from core import config
from core.pose_features import (LEFT, LEFT_ELBOW, LEFT_HIP, LEFT_INDEX, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT,
                                RIGHT_ELBOW, RIGHT_HIP, RIGHT_INDEX, RIGHT_SHOULDER, RIGHT_WRIST, angle_deg,
                                body_height_px, contact_pose, hand_points, infer_racket_hand)


def skeleton(joints):
    """(33, 3) landmarks: all invisible except the given {index: (x, y, score)}."""
    lm = np.zeros((33, 3), np.float32)
    for index, value in joints.items():
        lm[index] = value
    return lm


def test_hand_points_are_the_wrist_and_index_finger_of_the_side():
    lm = skeleton({LEFT_WRIST: (10, 20, 0.9), LEFT_INDEX: (12, 22, 0.8),
                   RIGHT_WRIST: (100, 200, 0.9), RIGHT_INDEX: (102, 202, 0.9)})
    assert hand_points(lm, LEFT, min_score=0.5) == [(10.0, 20.0), (12.0, 22.0)]
    assert hand_points(lm, RIGHT, min_score=0.5) == [(100.0, 200.0), (102.0, 202.0)]


def test_hand_points_drop_joints_below_the_score_and_missing_skeletons():
    lm = skeleton({LEFT_WRIST: (10, 20, 0.9), LEFT_INDEX: (12, 22, 0.3)})
    assert hand_points(lm, LEFT, min_score=0.5) == [(10.0, 20.0)]
    assert hand_points(lm, RIGHT, min_score=0.5) == []
    assert hand_points(None, LEFT, min_score=0.5) == []


def test_hand_points_default_score_is_read_from_config_at_call_time(monkeypatch):
    lm = skeleton({LEFT_WRIST: (10, 20, 0.6)})
    monkeypatch.setattr(config, "POSE_MIN_SCORE", 0.5)
    assert len(hand_points(lm, LEFT)) == 1
    monkeypatch.setattr(config, "POSE_MIN_SCORE", 0.7)
    assert hand_points(lm, LEFT) == []


def test_body_height_is_the_box_height():
    assert body_height_px((10, 20, 60, 220)) == 200.0


def test_racket_hand_is_the_majority_with_its_share():
    assert infer_racket_hand([RIGHT, RIGHT, LEFT]) == (RIGHT, pytest.approx(2 / 3))
    assert infer_racket_hand([LEFT]) == (LEFT, 1.0)


def test_racket_hand_is_unknown_without_votes_or_on_a_tie():
    assert infer_racket_hand([]) == (None, 0.0)
    assert infer_racket_hand([LEFT, RIGHT]) == (None, 0.5)


# ---- contact pose -----------------------------------------------------------------------------

BOX = (0.0, 0.0, 100.0, 200.0)        # a player 200 px tall


def arm_up():
    """Right arm straight up above the head, hips at y = 120, nose at y = 40."""
    return skeleton({NOSE: (50, 40, 0.9), RIGHT_HIP: (45, 120, 0.9), LEFT_HIP: (55, 120, 0.9),
                     RIGHT_SHOULDER: (40, 70, 0.9), RIGHT_ELBOW: (40, 45, 0.9), RIGHT_WRIST: (40, 20, 0.9)})


def test_angle_at_the_middle_point():
    assert angle_deg((0, 0), (1, 0), (2, 0)) == pytest.approx(180.0)
    assert angle_deg((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)
    assert angle_deg((1, 0), (0, 0), (1, 0)) == pytest.approx(0.0)
    assert angle_deg((1, 1), (1, 1), (2, 2)) == 0.0                      # degenerate segment


def test_wrist_above_the_nose_and_a_straight_arm_are_measured_in_body_heights():
    p = contact_pose(arm_up(), BOX, RIGHT, min_score=0.5)
    assert p.hand == RIGHT
    assert p.wrist_above_head_body == pytest.approx((40 - 20) / 200)      # nose y 40, wrist y 20
    assert p.wrist_below_hip_body == pytest.approx((20 - 120) / 200)      # far above the hips: negative
    assert p.elbow_angle_deg == pytest.approx(180.0)


def test_a_wrist_at_the_knees_is_below_the_hips_and_not_above_the_head():
    lm = arm_up()
    lm[RIGHT_WRIST] = (40, 160, 0.9)
    p = contact_pose(lm, BOX, RIGHT, min_score=0.5)
    assert p.wrist_above_head_body == pytest.approx((40 - 160) / 200) and p.wrist_below_hip_body == pytest.approx(0.2)


def test_without_a_given_hand_the_higher_wrist_is_used():
    lm = arm_up()
    lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST] = (60, 70, 0.9), (60, 100, 0.9), (60, 130, 0.9)
    assert contact_pose(lm, BOX, None, min_score=0.5).hand == RIGHT


def test_contact_pose_needs_the_nose_a_hip_and_the_wrist():
    for missing in (NOSE, RIGHT_WRIST):
        lm = arm_up()
        lm[missing, 2] = 0.1
        assert contact_pose(lm, BOX, RIGHT, min_score=0.5) is None
    lm = arm_up()
    lm[RIGHT_HIP, 2] = lm[LEFT_HIP, 2] = 0.1
    assert contact_pose(lm, BOX, RIGHT, min_score=0.5) is None
    assert contact_pose(None, BOX, RIGHT, min_score=0.5) is None
    assert contact_pose(arm_up(), (0, 0, 100, 0), RIGHT, min_score=0.5) is None       # empty box


def test_a_hidden_elbow_leaves_the_angle_unknown_but_keeps_the_heights():
    lm = arm_up()
    lm[RIGHT_ELBOW, 2] = 0.1
    p = contact_pose(lm, BOX, RIGHT, min_score=0.5)
    assert p.elbow_angle_deg is None and p.wrist_above_head_body == pytest.approx(0.1)
