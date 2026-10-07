"""
Measurements on MediaPipe pose landmarks (the (33, 3) arrays of core/pose_estimator.py: x and y in
full-frame pixels, score 0..1). Pure numpy: no model is loaded here.

Image coordinates only (2D pose from one camera): no depth, so nothing here is a real 3D angle.
"""
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import config

LEFT = "left"      # the player's own left, as MediaPipe names its landmarks
RIGHT = "right"

# MediaPipe pose landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_HIP, RIGHT_HIP = 23, 24

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]
"""Pairs of landmark indices drawn as bones."""

ARM_LANDMARKS = {
    LEFT: (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    RIGHT: (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

HAND_LANDMARKS = {
    LEFT: (LEFT_WRIST, LEFT_INDEX),
    RIGHT: (RIGHT_WRIST, RIGHT_INDEX),
}
"""Landmarks that describe where a hand is: the wrist and the index fingertip (the racket extends
beyond them along the forearm)."""


def hand_points(landmarks: Optional[np.ndarray], side: str, min_score: Optional[float] = None
                ) -> List[Tuple[float, float]]:
    """Image points (x, y) px of the wrist and index fingertip of `side` whose score reaches
    `min_score` (default config.POSE_MIN_SCORE). Empty when there is no skeleton or no visible point."""
    if landmarks is None:
        return []
    min_score = config.POSE_MIN_SCORE if min_score is None else min_score
    return [(float(landmarks[i, 0]), float(landmarks[i, 1])) for i in HAND_LANDMARKS[side]
            if landmarks[i, 2] >= min_score]


def body_height_px(bbox: Sequence[float]) -> float:
    """Height (px) of a person's box (x1, y1, x2, y2): the scale that makes pixel distances
    comparable between a player close to the camera and one far from it."""
    return float(bbox[3] - bbox[1])


def infer_racket_hand(votes: Sequence[str]) -> Tuple[Optional[str], float]:
    """
    Racket hand from the hand that was closest to the shuttle at each attributed hit.
    Returns (hand, share of the votes it got); (None, 0.0) without votes, (None, share of the
    leading votes) when two hands tie.
    """
    if not votes:
        return None, 0.0
    ranked = Counter(votes).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, ranked[0][1] / len(votes)
    return ranked[0][0], ranked[0][1] / len(votes)


def angle_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Angle (degrees, 0..180) at b between the segments b-a and b-c; 180 = a straight line."""
    v1 = np.asarray(a, float) - np.asarray(b, float)
    v2 = np.asarray(c, float) - np.asarray(b, float)
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / norm))))


@dataclass
class ContactPose:
    """The racket arm at the moment of a hit, in image terms (no depth: 2D pose)."""
    hand: str                                   # LEFT | RIGHT: the arm that was measured
    wrist_above_head_body: float                # (nose y - wrist y) / box height; positive = wrist above the nose
    wrist_below_hip_body: float                 # (wrist y - hip y) / box height; positive = wrist below the hips
    elbow_angle_deg: Optional[float]            # shoulder-elbow-wrist; None when the arm is not fully visible


def contact_pose(landmarks: Optional[np.ndarray], bbox: Sequence[float], hand: Optional[str],
                 min_score: Optional[float] = None) -> Optional[ContactPose]:
    """
    Racket-arm measurements from one skeleton. hand: the arm that hit (None = the one whose wrist
    is higher in the image). None when the nose, the wrist or both hips are not visible.
    """
    if landmarks is None:
        return None
    min_score = config.POSE_MIN_SCORE if min_score is None else min_score
    height = body_height_px(bbox)

    def visible(i: int) -> bool:
        return bool(landmarks[i, 2] >= min_score)

    hips = [landmarks[i, 1] for i in (LEFT_HIP, RIGHT_HIP) if visible(i)]
    if height <= 0 or not visible(NOSE) or not hips:
        return None
    candidates = (LEFT, RIGHT) if hand is None else (hand,)
    sides = [s for s in candidates if visible(ARM_LANDMARKS[s][2])]      # the wrist must be seen
    if not sides:
        return None
    side = min(sides, key=lambda s: landmarks[ARM_LANDMARKS[s][2], 1])
    shoulder, elbow, wrist = ARM_LANDMARKS[side]
    wrist_y = float(landmarks[wrist, 1])
    elbow_angle = None
    if visible(shoulder) and visible(elbow):
        elbow_angle = angle_deg(landmarks[shoulder, :2], landmarks[elbow, :2], landmarks[wrist, :2])
    return ContactPose(side, (float(landmarks[NOSE, 1]) - wrist_y) / height,
                       (wrist_y - float(np.mean(hips))) / height, elbow_angle)
