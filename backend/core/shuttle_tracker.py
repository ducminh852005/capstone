import bisect
import cv2
import numpy as np
import logging

from .tracknet import TrackNetCandidateSource

logger = logging.getLogger(__name__)

CHI2_2DOF_99 = 9.21


class ShuttleTrajectoryProcessor:
    def __init__(self, max_gap_frames=15, fit_window=8, max_fit_rms_px=4.0):
        """
        max_gap_frames: longest run of missing frames that is interpolated (15 = 0.25 s at 60 fps).
                        Longer gaps are left empty rather than inventing a trajectory.
        fit_window: number of valid points taken on each side of a gap for the parabola fit.
        max_fit_rms_px: if the parabola does not fit the points around the gap this well
                        (typically because the shuttle was hit inside the gap), fall back
                        to linear interpolation between the gap endpoints.
        """
        self.max_gap_frames = max_gap_frames
        self.fit_window = fit_window
        self.max_fit_rms_px = max_fit_rms_px

        # Precompute Gamma LUT table to save CPU (gamma 0.5 -> (i/255)^2 darkens mid-tones)
        gamma = 0.5
        invGamma = 1.0 / gamma
        self.gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # Precompute Morphological kernel
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def preprocess_frame_for_tracking(self, frame):
        """
        Preprocesses a frame to reduce the impact of glare, ceiling lights.
        Returns an enhanced grayscale image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gamma_corrected = cv2.LUT(gray, self.gamma_table)
        # 5x5 top-hat keeps small bright spots such as the shuttle
        top_hat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_TOPHAT, self.tophat_kernel)
        return cv2.addWeighted(gamma_corrected, 0.7, top_hat, 0.5, 0)

    def fill_missing_trajectory(self, trajectory):
        """
        Fills short gaps (None entries) in a list of (x, y) per frame.

        Each gap is filled with a degree-2 polynomial in time fitted to up to `fit_window`
        valid points on each side. If the horizontal direction flips across the gap or the fit
        residual is large (a racket hit happened inside the gap), linear interpolation between
        the two gap endpoints is used instead. Leading/trailing gaps are never extrapolated.
        """
        filled = list(trajectory)
        valid_indices = [i for i, pt in enumerate(trajectory) if pt is not None]
        if len(valid_indices) < 2:
            return filled

        for a, b in zip(valid_indices[:-1], valid_indices[1:]):
            gap_start, gap_end = a + 1, b - 1
            if gap_end < gap_start:
                continue
            gap_length = gap_end - gap_start + 1
            if gap_length > self.max_gap_frames:
                logger.debug(f"Gap of {gap_length} frames is too large, skipping interpolation.")
                continue

            pos = bisect.bisect_left(valid_indices, gap_start)
            before = valid_indices[max(0, pos - self.fit_window):pos]
            after = valid_indices[pos:pos + self.fit_window]
            gap_t = np.arange(gap_start, gap_end + 1)

            values = None
            if len(before) >= 2 and len(after) >= 2 and not self._direction_flips(trajectory, before, after):
                t = np.array(before + after, dtype=np.float64)
                pts = np.array([trajectory[i] for i in before + after], dtype=np.float64)
                coef_x = np.polyfit(t, pts[:, 0], deg=2)
                coef_y = np.polyfit(t, pts[:, 1], deg=2)
                resid = np.hypot(np.polyval(coef_x, t) - pts[:, 0], np.polyval(coef_y, t) - pts[:, 1])
                if np.sqrt(np.mean(resid ** 2)) <= self.max_fit_rms_px:
                    values = np.stack([np.polyval(coef_x, gap_t), np.polyval(coef_y, gap_t)], axis=1)

            if values is None:
                pa, pb = np.array(trajectory[a], float), np.array(trajectory[b], float)
                w = ((gap_t - a) / (b - a))[:, None]
                values = pa + (pb - pa) * w

            for i, (x, y) in zip(gap_t, values):
                filled[i] = (int(round(x)), int(round(y)))
        return filled

    @staticmethod
    def _direction_flips(trajectory, before, after):
        vx_before = (trajectory[before[-1]][0] - trajectory[before[0]][0]) / max(before[-1] - before[0], 1)
        vx_after = (trajectory[after[-1]][0] - trajectory[after[0]][0]) / max(after[-1] - after[0], 1)
        return vx_before * vx_after < 0 and min(abs(vx_before), abs(vx_after)) > 1.0


class ConstantAccelerationKalman:
    """2D constant-acceleration Kalman filter, state (x, y, vx, vy, ax, ay), dt = 1 frame."""

    def __init__(self, meas_std=2.0, accel_noise=3.0):
        self.F = np.eye(6)
        self.F[0, 2] = self.F[1, 3] = 1.0
        self.F[2, 4] = self.F[3, 5] = 1.0
        self.F[0, 4] = self.F[1, 5] = 0.5
        self.Hm = np.zeros((2, 6))
        self.Hm[0, 0] = self.Hm[1, 1] = 1.0
        # piecewise white-jerk process noise
        g = np.array([1 / 6, 1 / 6, 0.5, 0.5, 1.0, 1.0])
        self.Q = np.outer(g, g) * accel_noise ** 2
        self.Q[np.ix_([0, 2, 4], [1, 3, 5])] = 0.0
        self.Q[np.ix_([1, 3, 5], [0, 2, 4])] = 0.0
        self.R = np.eye(2) * meas_std ** 2
        self.x = np.zeros(6)
        self.P = np.eye(6)

    def init(self, p1, p2, p3):
        """Initialise from three consecutive positions (oldest first)."""
        p1, p2, p3 = (np.asarray(p, float) for p in (p1, p2, p3))
        v = p3 - p2
        a = p3 - 2 * p2 + p1
        self.x = np.array([p3[0], p3[1], v[0], v[1], a[0], a[1]])
        self.P = np.diag([4.0, 4.0, 16.0, 16.0, 25.0, 25.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def innovation_cov(self):
        return self.Hm @ self.P @ self.Hm.T + self.R

    def mahalanobis2(self, pts):
        d = np.asarray(pts, float) - self.x[:2]
        S_inv = np.linalg.inv(self.innovation_cov())
        return np.einsum("ni,ij,nj->n", d, S_inv, d)

    def update(self, z):
        S = self.innovation_cov()
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, float) - self.x[:2])
        self.P = (np.eye(6) - K @ self.Hm) @ self.P

    def predict_ahead(self, n):
        x = self.x.copy()
        out = []
        for _ in range(n):
            x = self.F @ x
            out.append((float(x[0]), float(x[1])))
        return out


class CVCandidateSource:
    """
    Classical-CV candidate generator: background subtraction -> blob candidates.
    This is the original ShuttleDetector implementation, unchanged, just factored out
    behind the same generate(frame, roi) -> (candidates, confidences, mask) interface
    that TrackNetCandidateSource (core/tracknet.py) also implements.
    """

    def __init__(self, bg_method="knn", min_area=2, max_area=500, max_merged_area=900, max_elongation=6.0):
        self.processor = ShuttleTrajectoryProcessor()
        if bg_method == "knn":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=400, detectShadows=False)
        elif bg_method == "mog2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
        else:
            raise ValueError(f"Unknown bg_method {bg_method!r}")

        self.min_area, self.max_area = min_area, max_area
        self.max_merged_area = max_merged_area
        # one 9x9 dilation == two 5x5 dilations, in a single pass
        self.merge_kernel = np.ones((9, 9), np.uint8)
        self.max_elongation = max_elongation

    def _extract_candidates(self, fg_mask, offset):
        """
        Small, isolated foreground blobs. The raw mask is dilated so that the many fragments
        of a moving body merge into one large component (rejected by max_merged_area), while
        the shuttle stays a small isolated component. Area and centroid are measured on the
        raw (undilated) pixels of each component.
        """
        merged = cv2.dilate(fg_mask, self.merge_kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        w = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
        h = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        keep = np.flatnonzero((stats[1:, cv2.CC_STAT_AREA] < self.max_merged_area)
                              & (np.maximum(w, h) / np.maximum(np.minimum(w, h), 1.0) <= self.max_elongation)) + 1

        cands = []
        for lbl in keep:
            x, y, bw, bh = stats[lbl, :4]
            raw = fg_mask[y:y + bh, x:x + bw].copy()
            raw[labels[y:y + bh, x:x + bw] != lbl] = 0
            m = cv2.moments(raw, binaryImage=True)
            if self.min_area < m["m00"] < self.max_area:
                cands.append((m["m10"] / m["m00"] + x + offset[0], m["m01"] / m["m00"] + y + offset[1]))
        return np.array(cands, dtype=np.float64).reshape(-1, 2), merged

    def generate(self, frame, roi):
        x0, y0 = 0, 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            frame = frame[y0:y1, x0:x1]

        processed = self.processor.preprocess_frame_for_tracking(frame)
        fg_mask = self.bg_subtractor.apply(processed)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        cands, mask = self._extract_candidates(fg_mask, (x0, y0))
        return cands, np.ones(len(cands)), mask


class ShuttleDetector:
    def __init__(self, backend="tracknet", tracknet_path=None, tracknet_kwargs=None,
                 bg_method="knn", min_area=2, max_area=500, max_merged_area=900, max_elongation=6.0,
                 max_candidates=150, max_coast=None, max_track_len=180, min_gate_px=15.0, min_body_speed=8.0,
                 init_max_residual_px=12.0, init_speed_px=(5.0, 150.0), init_min_confidence=0.6, simple_init=None):
        """
        Shuttle tracker: pluggable candidate generation (backend="cv" classical background
        subtraction, backend="tracknet" TrackNetV3 heatmap regression) feeding a shared
        constant-acceleration Kalman tracker with Mahalanobis gating. See core/tracknet.py
        for why TrackNetV3 uses a different (simpler, confidence-based) track-initiation
        rule than the CV backend's 3-consecutive-frame consistency check.

        A track survives up to max_coast frames without a matching candidate, then ends.
        A single flight never lasts more than a few seconds, so tracks longer than
        max_track_len frames (locked on clutter) are ended as well.

        backend="cv" params: bg_method, min_area, max_area, max_merged_area, max_elongation,
            init_max_residual_px, init_speed_px (3-frame init consistency check).
        backend="tracknet" params: tracknet_path (default backend/TrackNet_best.pt),
            tracknet_kwargs (dict, forwarded to TrackNetCandidateSource: batch_stride,
            conf_threshold, bg_frames, device), init_min_confidence (single-point init).
        """
        if backend == "cv":
            self._source = CVCandidateSource(bg_method=bg_method, min_area=min_area, max_area=max_area,
                                             max_merged_area=max_merged_area, max_elongation=max_elongation)
            self.simple_init = False if simple_init is None else simple_init
            self.max_coast = 8 if max_coast is None else max_coast
        elif backend == "tracknet":
            self._source = TrackNetCandidateSource(tracknet_path, **(tracknet_kwargs or {}))
            self.simple_init = True if simple_init is None else simple_init
            # nonoverlap batching only reports a real detection every batch_stride frames;
            # give coast enough headroom to ride out that structural gap plus a few misses.
            self.max_coast = max(12, self._source.batch_stride + 4) if max_coast is None else max_coast
        else:
            raise ValueError(f"Unknown backend {backend!r}, expected 'cv' or 'tracknet'")
        self.backend = backend

        self.max_candidates = max_candidates
        self.max_track_len = max_track_len
        self.min_gate_px = min_gate_px
        self.min_body_speed = min_body_speed
        self.init_max_residual = init_max_residual_px
        self.init_speed = init_speed_px
        self.init_min_confidence = init_min_confidence

        self.kf = ConstantAccelerationKalman()
        self.track_active = False
        self.misses = 0
        self.track_len = 0
        self._recent_candidates = []   # candidate arrays of the last 2 frames, for track initiation
        self.trajectory = []           # one (x, y) or None per detect() call, full-frame pixels
        self.last_roi = None

    @staticmethod
    def _in_boxes(pts, boxes, top_frac=0.0):
        """True for points inside any box, ignoring the top `top_frac` of each box."""
        inside = np.zeros(len(pts), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            top = y1 + (y2 - y1) * top_frac
            inside |= (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= top) & (pts[:, 1] <= y2)
        return inside

    def _try_init(self, cands):
        """CV backend: find c1, c2, c3 in three consecutive frames with c3 ~ 2*c2 - c1."""
        if len(self._recent_candidates) < 2:
            return None
        A, B = self._recent_candidates
        if len(A) * len(B) * len(cands) > 2_000_000:
            return None
        if not len(A) or not len(B) or not len(cands):
            return None
        # velocity between frame t-1 and t for every (B, C) pair
        v = cands[None, :, :] - B[:, None, :]                       # (nB, nC, 2)
        speed = np.linalg.norm(v, axis=2)
        pred_a = B[:, None, :] - v                                  # where the point must have been at t-2
        resid = np.linalg.norm(pred_a[:, :, None, :] - A[None, None, :, :], axis=3)  # (nB, nC, nA)
        lo, hi = self.init_speed
        resid[~((speed >= lo) & (speed <= hi))] = np.inf
        ib, ic, ia = np.unravel_index(np.argmin(resid), resid.shape)
        if resid[ib, ic, ia] > self.init_max_residual:
            return None
        return A[ia], B[ib], cands[ic]

    def _try_init_simple(self, cands, confs):
        """
        TrackNet backend: start a track from a single confident candidate instead of
        waiting for 3-frame consistency (batched inference only reports a candidate
        every batch_stride frames, so consecutive-frame consistency rarely applies; a
        purpose-trained detector's confident hits are precise enough to trust directly).
        Reuses ConstantAccelerationKalman.init(p1,p2,p3) with all three equal (zero
        initial velocity/acceleration -- the filter converges after a couple of updates).
        """
        if not len(cands):
            return None
        i = int(np.argmax(confs)) if len(confs) else 0
        if len(confs) and confs[i] < self.init_min_confidence:
            return None
        p = cands[i]
        return p, p, p

    def detect(self, frame, roi=None, exclude_boxes=None, player_boxes=None):
        """
        Detects the shuttle in one frame.
        roi: (x0, y0, x1, y1) crop to process; returned points are always full-frame pixels.
        exclude_boxes: boxes of people who are not playing (spectators); candidates inside
                       are ignored.
        player_boxes: boxes of players; candidates on the lower 2/3 (legs/torso) are ignored
                      unless the running track matches them while moving faster than
                      min_body_speed px/frame (keeps the shuttle at the moment of a hit).
        Returns (point or None, a mask sized to the ROI for display -- the foreground mask
        for backend="cv", the most recently resolved heatmap for backend="tracknet").
        """
        self.last_roi = roi
        cands, confs, mask = self._source.generate(frame, roi)

        if len(cands) > self.max_candidates:
            # global change (camera shake, light flicker): no reliable measurement this frame
            cands, confs = cands[:0], confs[:0]

        if exclude_boxes is not None and len(exclude_boxes) and len(cands):
            keep = ~self._in_boxes(cands, exclude_boxes)
            cands, confs = cands[keep], confs[keep]

        on_body = np.zeros(len(cands), dtype=bool)
        if player_boxes is not None and len(player_boxes) and len(cands):
            on_body = self._in_boxes(cands, player_boxes, top_frac=1 / 3)

        best_pt = None
        if self.track_active:
            self.kf.predict()
            if len(cands):
                d2 = self.kf.mahalanobis2(cands)
                eucl = np.linalg.norm(cands - self.kf.x[:2], axis=1)
                in_gate = (d2 <= CHI2_2DOF_99) | (eucl <= self.min_gate_px)
                # a shuttle crossing a player's body is fast (just hit); slow blobs there are limbs
                if np.linalg.norm(self.kf.x[2:4]) < self.min_body_speed:
                    in_gate &= ~on_body
                if in_gate.any():
                    i = int(np.argmin(np.where(in_gate, d2, np.inf)))
                    self.kf.update(cands[i])
                    best_pt = (int(round(cands[i, 0])), int(round(cands[i, 1])))
                    self.misses = 0
            self.track_len += 1
            if best_pt is None:
                self.misses += 1
            if self.misses > self.max_coast or self.track_len > self.max_track_len:
                self.track_active = False

        cands, confs = cands[~on_body], confs[~on_body]
        if not self.track_active:
            triple = self._try_init_simple(cands, confs) if self.simple_init else self._try_init(cands)
            if triple is not None:
                c1, c2, c3 = triple
                self.kf.init(c1, c2, c3)
                self.track_active = True
                self.misses = 0
                self.track_len = 0
                best_pt = (int(round(c3[0])), int(round(c3[1])))
                # the two previous frames belong to the new track as well (only meaningful
                # for the CV backend's 3-consecutive-frame init; a no-op duplicate write
                # for the simple/single-point init, where c1 == c2 == c3)
                for back, c in ((1, c2), (2, c1)):
                    if len(self.trajectory) >= back and self.trajectory[-back] is None:
                        self.trajectory[-back] = (int(round(c[0])), int(round(c[1])))

        self._recent_candidates = (self._recent_candidates + [cands])[-2:]
        self.trajectory.append(best_pt)
        return best_pt, mask

    def predict_ahead(self, n_frames):
        """Kalman extrapolation of the active track for the next n frames ([] if no track)."""
        return self.kf.predict_ahead(n_frames) if self.track_active else []

    def draw_trajectory(self, frame, tail_length=20):
        """Trailing path of the shuttle: older segments thin and dark, newer ones thick and bright."""
        annotated = frame.copy()
        recent_path = self.trajectory[-tail_length:]
        for i in range(1, len(recent_path)):
            pt1, pt2 = recent_path[i - 1], recent_path[i]
            if pt1 is None or pt2 is None:
                continue
            t = i / tail_length
            thickness = int(np.interp(t, [0, 1], [1, 4]))
            color = (0, int(80 + 175 * t), 255)
            cv2.line(annotated, pt1, pt2, color, thickness)

        if self.trajectory and self.trajectory[-1] is not None:
            cv2.circle(annotated, self.trajectory[-1], 6, (0, 0, 255), -1)
        return annotated


if __name__ == "__main__":
    processor = ShuttleTrajectoryProcessor()

    # Simulate a shuttle going up, going missing for 3 frames, then coming down
    mock_path = [
        (100, 500), (110, 400), (120, 310), (130, 230),  # ascending
        None, None, None,                                # out of frame!
        (170, 230), (180, 310), (190, 400), (200, 500)   # descending
    ]

    fixed_path = processor.fill_missing_trajectory(mock_path)
    for i, pt in enumerate(fixed_path):
        print(f"Frame {i}: {pt} {'(Interpolated)' if mock_path[i] is None else ''}")
