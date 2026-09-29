import numpy as np

from core.pose_estimator import select_foot_point, PoseEstimator


def lm(points):
    arr = np.zeros((33, 3), np.float32)
    for idx, (x, y) in points.items():
        arr[idx] = (x, y, 0.9)
    return arr


def test_both_feet_on_ground_gives_midpoint():
    pt, src = select_foot_point(lm({29: (100, 500), 31: (120, 502), 30: (160, 498), 32: (180, 500)}), bbox_h=300)
    assert src == "foot"
    assert np.allclose(pt, (140, 500), atol=0.5)


def test_raised_foot_uses_lower_foot():
    pt, _ = select_foot_point(lm({29: (100, 500), 31: (120, 500), 30: (160, 420), 32: (180, 430)}), bbox_h=300)
    assert np.allclose(pt, (110, 500))


def test_heel_only_and_ankle_fallback():
    pt, src = select_foot_point(lm({29: (100, 500)}), bbox_h=300)
    assert src == "foot" and np.allclose(pt, (100, 500))
    pt, src = select_foot_point(lm({27: (100, 480)}), bbox_h=300)
    assert src == "ankle" and np.allclose(pt, (100, 480))
    assert select_foot_point(np.zeros((33, 3), np.float32), bbox_h=300) == (None, None)


def test_padded_crop_is_clamped():
    assert PoseEstimator.padded_crop_box((1080, 1920), (-10, 20, 100, 1100), pad=0.1) == (0, 0, 111, 1080)
