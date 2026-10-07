import cv2
import numpy as np
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

from . import config
from .pose_features import POSE_CONNECTIONS  # noqa: F401  (pure module; re-exported for the demos)

logger = logging.getLogger(__name__)

MODEL_FILES = {
    "lite": "pose_landmarker_lite.task",
    "heavy": "pose_landmarker_heavy.task",
}

# MediaPipe landmark indices per foot: (heel, foot_index/toe, ankle)
LEFT_FOOT = (29, 31, 27)
RIGHT_FOOT = (30, 32, 28)



def _single_foot(landmarks, heel, toe, ankle, min_score):
    """Contact point of one foot: heel/toe midpoint, else heel or toe alone, else the ankle."""
    heel_ok = landmarks[heel, 2] > min_score
    toe_ok = landmarks[toe, 2] > min_score
    if heel_ok and toe_ok:
        return (landmarks[heel, :2] + landmarks[toe, :2]) / 2.0, "foot"
    if heel_ok:
        return landmarks[heel, :2], "foot"
    if toe_ok:
        return landmarks[toe, :2], "foot"
    if landmarks[ankle, 2] > min_score:
        return landmarks[ankle, :2], "ankle"
    return None, None


def select_foot_point(landmarks, bbox_h, min_score=0.5, ground_tol=0.06):
    """
    Pick the player's ground point from pixel landmarks.

    landmarks: array (33, 3) of (x, y, score) in image pixels.
    When both feet are found and their heights differ by less than ground_tol * bbox_h,
    both are assumed on the ground and their midpoint is returned (the labelling
    convention). Otherwise the lower foot is used, because the raised one is off the floor.

    Returns ((x, y), source) with source in {"foot", "ankle"}, or (None, None).
    """
    left, left_src = _single_foot(landmarks, *LEFT_FOOT, min_score)
    right, right_src = _single_foot(landmarks, *RIGHT_FOOT, min_score)

    if left is None and right is None:
        return None, None
    if left is None:
        return (float(right[0]), float(right[1])), right_src
    if right is None:
        return (float(left[0]), float(left[1])), left_src

    source = "foot" if left_src == right_src == "foot" else "ankle"
    if abs(left[1] - right[1]) < ground_tol * bbox_h:
        mid = (left + right) / 2.0
        return (float(mid[0]), float(mid[1])), source
    lower, lower_src = (left, left_src) if left[1] > right[1] else (right, right_src)
    return (float(lower[0]), float(lower[1])), lower_src


class PoseEstimator:
    def __init__(self, variant="lite"):
        """
        MediaPipe Pose Landmarker (Tasks API) run on a crop around one player.
        variant: "lite" or "heavy".
        """
        if variant not in MODEL_FILES:
            raise ValueError(f"Unknown pose variant {variant!r}, expected one of {list(MODEL_FILES)}")
        self.variant = variant

        model_path = config.MODELS_DIR / MODEL_FILES[variant]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model not found at {model_path}; see QUICK_START.md to download it")

        with open(model_path, 'rb') as f:
            model_bytes = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        # Landmarks of the most recent detect_landmarks() call (None when it found nobody). Callers
        # that only get a foot point back from extract_foot_point() read the skeleton from here.
        self.last_landmarks = None

    @staticmethod
    def padded_crop_box(frame_shape, bbox, pad=0.15):
        """Expand bbox by `pad` of its size on every side and clamp to the frame."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        return (max(int(x1 - px), 0), max(int(y1 - py), 0), min(int(x2 + px), w), min(int(y2 + py), h))

    def detect_landmarks(self, frame, bbox, pad=0.15):
        """
        Run MediaPipe on a padded crop around bbox. Padding matters: with a tight crop the
        feet sit on the crop border and the landmarker often misses them.
        Returns an array (33, 3) of (x, y, score) in full-frame pixels, or None.
        The result is also kept in `last_landmarks`.
        """
        self.last_landmarks = None
        cx1, cy1, cx2, cy2 = self.padded_crop_box(frame.shape, bbox, pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        if not result.pose_landmarks:
            return None

        ch, cw = crop.shape[:2]
        out = np.empty((len(result.pose_landmarks[0]), 3), dtype=np.float32)
        for i, lm in enumerate(result.pose_landmarks[0]):
            presence = getattr(lm, 'presence', None)
            out[i] = (lm.x * cw + cx1, lm.y * ch + cy1, min(lm.visibility, presence if presence is not None else 1.0))
        self.last_landmarks = out
        return out

    def extract_foot_point(self, frame, bbox, pad=0.15):
        """
        Foot point of the player inside bbox (x1, y1, x2, y2) of the full frame.
        Returns ((x, y), source) in full-frame pixels, or (None, None).
        """
        landmarks = self.detect_landmarks(frame, bbox, pad)
        if landmarks is None:
            return None, None
        return select_foot_point(landmarks, bbox[3] - bbox[1])

    def draw_landmarks(self, image, landmarks, min_score=0.4):
        """Draw the skeleton from detect_landmarks() output onto image (in place)."""
        if landmarks is None:
            return image

        pts = {i: (int(x), int(y)) for i, (x, y, s) in enumerate(landmarks) if s > min_score}
        for p1, p2 in POSE_CONNECTIONS:
            if p1 in pts and p2 in pts:
                cv2.line(image, pts[p1], pts[p2], (0, 255, 0), 2)
        for pt in pts.values():
            cv2.circle(image, pt, 3, (255, 0, 0), -1)
        return image
