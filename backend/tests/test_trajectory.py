import numpy as np

from core import trajectory

FPS = 60


def test_speed_outlier_removed_and_short_gap_interpolated():
    t = np.arange(120)
    xy = np.stack([1.0 + 0.02 * t, np.full(120, 3.0)], axis=1)   # 1.2 m/s
    xy[50] = (5.0, 5.0)                                           # teleport -> outlier
    keep = np.ones(120, bool)
    keep[80:98] = False                                           # 0.3 s gap
    start, clean = trajectory.clean_player_track(t[keep], xy[keep], FPS)
    assert start == 0 and len(clean) == 120
    assert not np.isnan(clean).any()
    assert abs(clean[50, 0] - (1.0 + 0.02 * 50)) < 0.02
    assert abs(clean[90, 0] - (1.0 + 0.02 * 90)) < 0.02


def test_long_gap_not_counted_in_distance():
    t = np.concatenate([np.arange(0, 60), np.arange(120, 180)])   # 1 s gap
    x = np.where(t < 60, 1.0 + 0.01 * t, 4.0 + 0.01 * (t - 120))
    xy = np.stack([x, np.full(len(t), 3.0)], axis=1)
    _, clean = trajectory.clean_player_track(t, xy, FPS)
    assert np.isnan(clean[60:120]).all()
    assert abs(trajectory.path_length(clean) - 2 * 0.59) < 0.05   # the 2.4 m jump is not counted


def test_heatmap_sums_to_valid_time():
    xy = np.tile([[3.0, 3.0]], (120, 1))
    xy[:30] = np.nan
    grid = trajectory.occupancy_heatmap(xy, FPS, blur_m=0)
    assert abs(grid.sum() - 90 / FPS) < 1e-5


def test_reject_outliers_matches_the_numpy_norm_reference_on_random_tracks():
    """The math.hypot version must accept/reject exactly the same points as the np.linalg.norm one."""
    import numpy as np
    from core import court_model
    from core.trajectory import reject_outliers

    def reference(xy, fps, region=court_model.REGION_BUFFERED, max_speed=7.0, reset_after_s=0.25):
        out = xy.copy()
        x_min, x_max, y_min, y_max = region
        outside = ~((out[:, 0] >= x_min) & (out[:, 0] <= x_max) & (out[:, 1] >= y_min) & (out[:, 1] <= y_max))
        out[outside] = np.nan
        last_i, rejected_since = None, None
        reset_frames = max(int(reset_after_s * fps), 1)
        for i in np.flatnonzero(~np.isnan(out[:, 0])):
            if last_i is not None:
                speed = np.linalg.norm(out[i] - out[last_i]) * fps / (i - last_i)
                if speed > max_speed and (rejected_since is None or i - rejected_since < reset_frames):
                    rejected_since = rejected_since if rejected_since is not None else i
                    out[i] = np.nan
                    continue
            last_i, rejected_since = i, None
        return out

    rng = np.random.default_rng(3)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 400
        walk = np.cumsum(rng.normal(0, 0.05, (n, 2)), axis=0) + [3.0, 3.0]
        walk[rng.random(n) < 0.05] += rng.normal(0, 3.0, 2)       # spikes
        walk[rng.random(n) < 0.05] = np.nan                        # gaps
        assert np.array_equal(reject_outliers(walk, 30.0), reference(walk, 30.0), equal_nan=True)
