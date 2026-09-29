import cv2
import numpy as np
import logging

from . import court_model

logger = logging.getLogger(__name__)

# Line mask is restricted to the near half-court plus a small buffer. The far edge stops
# short of the net so the white net tape does not leak into the mask.
LINE_MASK_REGION = (-0.8, 6.6, -0.4, 6.5)


def _line_crossings(lines, eps=1e-6):
    """{(i, j): (x, y)} for every constant-x line i crossing a constant-y line j within both segments."""
    crossings = {}
    for i, ((xa, ya), (xb, yb)) in enumerate(lines):
        if abs(xa - xb) > eps:
            continue
        for j, ((ca, cy), (cb, cy2)) in enumerate(lines):
            if abs(cy - cy2) > eps:
                continue
            if min(ca, cb) - eps <= xa <= max(ca, cb) + eps and min(ya, yb) - eps <= cy <= max(ya, yb) + eps:
                crossings[(i, j)] = (xa, cy)
    return crossings


class CourtCalibrator:
    def __init__(self, court_model_points=None):
        """
        court_model_points maps corner names to world (x, y) in meters.
        World frame: x along the court length (0 = near baseline, 6.70 = net),
        y along the width (0 .. 6.10). See core/court_model.py.
        """
        self.court_model_points = court_model_points or self._get_default_points()

    def _get_default_points(self):
        names = ["bottom_left", "bottom_right", "net_left", "net_right"]
        return dict(zip(names, court_model.CALIB_WORLD_POINTS))

    def extract_clean_background(self, video_path: str, num_frames=60):
        """
        Step 2: Clean background as the temporal median of N frames spread over the video.
        Reads sequentially with grab() and only decodes the frames it keeps, which is much
        faster than seeking (every H.264 seek re-decodes from the previous keyframe).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = min(num_frames, total_frames)
        if num_frames <= 0:
            cap.release()
            return None
        wanted = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())

        stack = None
        n = 0
        for fid in range(total_frames):
            if not cap.grab():
                break
            if fid not in wanted:
                continue
            ret, frame = cap.retrieve()
            if not ret:
                continue
            if stack is None:
                stack = np.empty((num_frames, *frame.shape), dtype=np.uint8)
            stack[n] = frame
            n += 1
        cap.release()

        if n == 0:
            return None
        return np.median(stack[:n], axis=0).astype(np.uint8)

    def get_rough_homography(self, bg_image, keypoints_image, keypoints_world):
        """
        Step 3: World -> image homography from point correspondences.
        Exactly 4 points use the direct solution (RANSAC is meaningless with the minimum set).
        """
        if len(keypoints_image) < 4 or len(keypoints_world) < 4:
            logger.error("At least 4 points are required for homography.")
            return None
        return court_model.homography_from_points(keypoints_image, keypoints_world)

    def extract_white_lines(self, bg_image, H=None, kernel_size=11):
        """
        Step 4: Top-hat + Otsu to extract bright lines. When H is given, everything outside
        the near half-court (plus buffer) is masked out to drop lines of neighbouring courts.
        The kernel must be wider than the thickest line in pixels or near lines get hollowed.
        """
        gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, thresholded = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if H is not None:
            roi = np.zeros_like(thresholded)
            cv2.fillPoly(roi, [court_model.region_polygon_img(H, LINE_MASK_REGION)], 255)
            thresholded = cv2.bitwise_and(thresholded, roi)
        return thresholded

    @staticmethod
    def _line_centres(line_mask):
        """Thin the mask to 1-px centre lines, since the model describes line centres."""
        if hasattr(cv2, "ximgproc"):
            return cv2.ximgproc.thinning(line_mask)
        return line_mask

    def line_overlap_score(self, line_mask, H, step_m=0.05, dilate_px=2):
        """Fraction of projected model-line samples that land on the white-line mask."""
        h, w = line_mask.shape[:2]
        k = 2 * dilate_px + 1
        mask = cv2.dilate(line_mask, np.ones((k, k), np.uint8))
        pts = np.round(court_model.world_to_img(court_model.sample_line_points(step_m), H)).astype(int)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        if not inside.any():
            return 0.0
        hits = mask[pts[inside, 1], pts[inside, 0]] > 0
        return float(hits.sum()) / len(pts)

    def refine_homography(self, thresholded_image, rough_H, search_radii=(12, 6, 3), step_m=0.05, min_pixels=10):
        """
        Step 5: Refine H by aligning the model lines with the white-line mask.

        Per iteration: project each model line with the current H, collect the line-centre
        pixels within `search_radii[i]` of it, fit a straight image line (Huber), then
        intersect the fitted lines pairwise at every model line crossing (baseline x service
        lines x sidelines x centre line) and re-estimate H from those intersections.
        Plain point-to-point ICP slides along the lines and converges very slowly; using line
        intersections fixes both directions at once.

        A new H is kept only if the line overlap score does not drop. Returns (H, score).
        """
        if rough_H is None:
            return None, 0.0
        centres = self._line_centres(thresholded_image)
        if cv2.countNonZero(centres) < 50:
            logger.warning("Too few line pixels to refine the homography.")
            return rough_H, self.line_overlap_score(thresholded_image, rough_H)

        # distance to the nearest centre pixel + that pixel's coordinates, computed once
        src = np.where(centres > 0, 0, 1).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        zero = src == 0
        lut = np.zeros((labels.max() + 1, 2), dtype=np.float32)
        lut[labels[zero]] = np.argwhere(zero)[:, ::-1]

        lines = court_model.near_half_lines()
        samples = [court_model.sample_line_points(step_m, [line]) for line in lines]
        h, w = thresholded_image.shape[:2]
        H = rough_H
        best = self.line_overlap_score(thresholded_image, H)

        for radius in search_radii:
            fitted = {}
            for i, world_pts in enumerate(samples):
                img = np.round(court_model.world_to_img(world_pts, H)).astype(int)
                img = img[(img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)]
                img = img[dist[img[:, 1], img[:, 0]] <= radius]
                if len(img) < min_pixels:
                    continue
                targets = lut[labels[img[:, 1], img[:, 0]]]
                vx, vy, x0, y0 = cv2.fitLine(targets, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                fitted[i] = np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])

            world_int, img_int = [], []
            for (i, j), world_pt in _line_crossings(lines).items():
                if i in fitted and j in fitted:
                    p = np.cross(fitted[i], fitted[j])
                    if abs(p[2]) > 1e-9:
                        world_int.append(world_pt)
                        img_int.append(p[:2] / p[2])
            if len(world_int) < 4:
                break
            H_new = court_model.homography_from_points(img_int, world_int)
            if H_new is None:
                continue
            score = self.line_overlap_score(thresholded_image, H_new)
            if score >= best - 0.01:
                H, best = H_new, max(score, best)
        return H, best

    def validate_calibration(self, H, test_image_points, test_world_points, threshold=5.0):
        """
        Step 7: Reprojection error on points that were NOT used to compute H.
        Returns (ok, mean_error_px, max_error_px).
        """
        if H is None:
            return False, float("inf"), float("inf")
        projected = court_model.world_to_img(test_world_points, H)
        errors = np.linalg.norm(np.asarray(test_image_points, dtype=np.float32).reshape(-1, 2) - projected, axis=1)
        mean_error, max_error = float(errors.mean()), float(errors.max())
        if mean_error > threshold:
            logger.warning("Calibration warning: Mean error %.2f > threshold %s", mean_error, threshold)
            return False, mean_error, max_error
        return True, mean_error, max_error

    def calculate_intersection_from_lines(self, line1_pts, line2_pts):
        """
        Calculate intersection of two lines using homogeneous coordinates cross product.
        Useful when a corner is out of frame.
        line1_pts: list of points [(x1,y1), (x2,y2)] on line 1
        line2_pts: list of points [(x3,y3), (x4,y4)] on line 2
        """
        p1 = np.array([line1_pts[0][0], line1_pts[0][1], 1.0])
        p2 = np.array([line1_pts[1][0], line1_pts[1][1], 1.0])
        p3 = np.array([line2_pts[0][0], line2_pts[0][1], 1.0])
        p4 = np.array([line2_pts[1][0], line2_pts[1][1], 1.0])

        l1 = np.cross(p1, p2)
        l2 = np.cross(p3, p4)
        pt_homogeneous = np.cross(l1, l2)

        if abs(pt_homogeneous[2]) < 1e-9:
            # Parallel lines
            return None

        x = pt_homogeneous[0] / pt_homogeneous[2]
        y = pt_homogeneous[1] / pt_homogeneous[2]
        return (int(round(x)), int(round(y)))

    def draw_court_frame(self, image, H, full=False):
        """
        Draw the near half-court with all its inner lines. With full=True, also draw the
        full-court outline (the far half is extrapolated, so it is drawn thin and grey).
        """
        if H is None:
            return image

        result = image.copy()
        if full:
            outline = court_model.world_to_img(
                [(0, 0), (court_model.COURT_LENGTH, 0), (court_model.COURT_LENGTH, court_model.COURT_WIDTH), (0, court_model.COURT_WIDTH)], H)
            cv2.polylines(result, [np.int32(outline).reshape(-1, 1, 2)], True, (160, 160, 160), 1)

        for a, b in court_model.near_half_lines():
            pa, pb = np.int32(court_model.world_to_img([a, b], H))
            cv2.line(result, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (0, 255, 0), 2)

        net = np.int32(court_model.world_to_img([(court_model.NET_X, 0), (court_model.NET_X, court_model.COURT_WIDTH)], H))
        cv2.line(result, tuple(int(v) for v in net[0]), tuple(int(v) for v in net[1]), (0, 0, 255), 2)

        for pt in np.int32(court_model.world_to_img(court_model.CALIB_WORLD_POINTS, H)):
            cv2.circle(result, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        return result
