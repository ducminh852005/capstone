import numpy as np
import cv2

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


def test_static_flicker_never_starts_a_track():
    det = ShuttleDetector(backend="cv")
    for t in range(40):
        blobs = [(900, 400)] if t % 2 == 0 else []
        blobs += [(300, 800)] if t % 3 == 0 else []
        pt, _ = det.detect(render(blobs))
        assert pt is None
    assert not det.track_active


def test_player_body_blobs_are_ignored():
    det = ShuttleDetector(backend="cv")
    box = (800, 300, 1000, 800)
    for t in range(20):
        # a limb moving consistently inside the lower body region of the player
        pt, _ = det.detect(render([(900, 600 - 5 * t)]), player_boxes=[box])
        assert pt is None


def test_slow_track_does_not_slide_onto_player_legs():
    det = ShuttleDetector(backend="cv")
    box = (800, 300, 1000, 800)   # lower 2/3 starts at y = 466
    for t in range(40):
        y = 250 + 6 * t           # racket-like blob moving slowly down into the body
        pt, _ = det.detect(render([(900, y)]), player_boxes=[box])
        if y > 480:
            assert pt is None


def test_spectator_boxes_are_ignored():
    det = ShuttleDetector(backend="cv")
    box = (100, 100, 400, 900)
    for t in range(20):
        pt, _ = det.detect(render([(250, 200 + 8 * t)]), exclude_boxes=[box])
        assert pt is None


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
