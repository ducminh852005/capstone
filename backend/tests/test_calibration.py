import numpy as np
import cv2


from core import court_model
from core.court_calibration import CourtCalibrator

# Corners from data/calibration.json (Bottom-Left, Bottom-Right, Net-Left, Net-Right)
TRUE_CORNERS = [[322, 740], [999, 1029], [1008, 651], [1671, 735]]
FRAME = (1080, 1920)


def render_line_mask(H, thickness=4):
    mask = np.zeros(FRAME, np.uint8)
    for a, b in court_model.near_half_lines():
        pa, pb = court_model.world_to_img([a, b], H)
        cv2.line(mask, tuple(int(round(v)) for v in pa), tuple(int(round(v)) for v in pb), 255, thickness)
    return mask


def mean_reproj_error(H, H_ref):
    pts = court_model.sample_line_points(0.25)
    return float(np.linalg.norm(court_model.world_to_img(pts, H) - court_model.world_to_img(pts, H_ref), axis=1).mean())


def test_homography_roundtrip():
    H = court_model.homography_from_points(TRUE_CORNERS)
    world = np.float32([[0, 0], [3.05, 2.0], [6.7, 6.1], [-1, -0.5]])
    back = court_model.img_to_world(court_model.world_to_img(world, H), np.linalg.inv(H))
    assert np.abs(back - world).max() < 1e-3
    # the 4 calibration corners are reproduced exactly
    assert np.abs(court_model.world_to_img(court_model.CALIB_WORLD_POINTS, H) - np.float32(TRUE_CORNERS)).max() < 0.5


def test_validate_calibration_reports_errors():
    cal = CourtCalibrator()
    H = court_model.homography_from_points(TRUE_CORNERS)
    ok, mean_err, max_err = cal.validate_calibration(H, TRUE_CORNERS, court_model.CALIB_WORLD_POINTS)
    assert ok and mean_err < 0.5 and max_err < 0.5
    shifted = np.float32(TRUE_CORNERS) + 20
    ok, mean_err, _ = cal.validate_calibration(H, shifted, court_model.CALIB_WORLD_POINTS)
    assert not ok and mean_err > 20


def test_refine_homography_recovers_perturbed_corners():
    rng = np.random.default_rng(0)
    H_true = court_model.homography_from_points(TRUE_CORNERS)
    mask = render_line_mask(H_true)
    cal = CourtCalibrator()
    for _ in range(5):
        noisy = np.float32(TRUE_CORNERS) + rng.uniform(-8, 8, size=(4, 2))
        H_rough = court_model.homography_from_points(noisy)
        H_ref, score = cal.refine_homography(mask, H_rough)
        err_rough = mean_reproj_error(H_rough, H_true)
        err_ref = mean_reproj_error(H_ref, H_true)
        assert err_ref < 1.5, (err_rough, err_ref)
        assert err_ref < err_rough
        assert score > 0.95


def test_line_mask_is_limited_to_near_half():
    H = court_model.homography_from_points(TRUE_CORNERS)
    bg = np.zeros((*FRAME, 3), np.uint8)
    bg[:] = (40, 120, 40)
    cv2.line(bg, (0, 60), (1919, 60), (255, 255, 255), 4)  # a line far outside the court
    for a, b in court_model.near_half_lines():
        pa, pb = court_model.world_to_img([a, b], H)
        cv2.line(bg, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (255, 255, 255), 4)
    mask = CourtCalibrator().extract_white_lines(bg, H)
    assert mask[55:66, :].max() == 0
    assert CourtCalibrator().line_overlap_score(mask, H) > 0.9
