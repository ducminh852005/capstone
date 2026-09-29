"""
Smash detection from shuttle tracks.

A hit is recognised when the shuttle tracker starts a new track: the previous trajectory
broke because the shuttle changed direction sharply on contact with the racket (see
ShuttleDetector's physics filter). The hit's speed is the displacement between the new
track's start and its first real detection, divided by the frames between them
(px/frame). A hit counts as a smash when it is fast enough, starts low enough in the image
(not above the play area) and heads in an accepted direction.

Thresholds default to the values in config.py, read at call time so a live tuner that
mutates `config` takes effect immediately; pass explicit values to override.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


@dataclass
class HitMeasurement:
    """Speed and direction of the shuttle right after a racket hit."""
    frame_idx: int          # frame where the new track started
    pt: tuple               # (x, y) image px where the hit happened
    speed: float            # px/frame
    angle_deg: float        # atan2(dy, dx) in image coordinates (y down): 90 = straight down
    is_smash: bool
    smash_number: int = 0   # running smash count when is_smash, else 0


class SmashDetector:
    def __init__(self, speed_threshold: Optional[float] = None, min_y: Optional[float] = None,
                 min_angle: Optional[float] = None, max_angle: Optional[float] = None):
        """
        speed_threshold: px/frame a hit must exceed (default config.SMASH_SPEED_THRESHOLD).
        min_y: hits above this image row are ignored (default config.SMASH_MIN_Y).
        min_angle / max_angle: accepted direction range in degrees (config.SMASH_MIN_ANGLE /
            SMASH_MAX_ANGLE).
        """
        self._speed_threshold = speed_threshold
        self._min_y = min_y
        self._min_angle = min_angle
        self._max_angle = max_angle
        self.events = []            # smash HitMeasurements, in order
        self._pending = None        # (start_frame, start_pt) waiting for a second detection

    @property
    def speed_threshold(self):
        return config.SMASH_SPEED_THRESHOLD if self._speed_threshold is None else self._speed_threshold

    @property
    def min_y(self):
        return config.SMASH_MIN_Y if self._min_y is None else self._min_y

    @property
    def min_angle(self):
        return config.SMASH_MIN_ANGLE if self._min_angle is None else self._min_angle

    @property
    def max_angle(self):
        return config.SMASH_MAX_ANGLE if self._max_angle is None else self._max_angle

    @property
    def count(self):
        return len(self.events)

    def is_smash(self, speed, angle_deg, y):
        """True if a hit with this speed, direction and start row qualifies as a smash."""
        return (speed > self.speed_threshold
                and y >= self.min_y
                and self.min_angle <= angle_deg <= self.max_angle)

    def update(self, frame_idx, pt, track_active, track_len, track_pos) -> Optional[HitMeasurement]:
        """
        Feed one frame after ShuttleDetector.detect().

        pt: detection this frame (x, y) or None.
        track_active / track_len: ShuttleDetector state; track_len == 0 on the frame a
            track starts.
        track_pos: (x, y) estimate of the shuttle on that start frame, used when there is no
            detection on it (e.g. ShuttleDetector.kf.x[:2]).

        Returns a HitMeasurement on the frame the second detection arrives (whether or not
        it is a smash), otherwise None.
        """
        if not track_active:
            self._pending = None    # track lost before a second point: discard
            return None

        if track_len == 0:
            start_pt = pt if pt is not None else track_pos
            self._pending = (frame_idx, start_pt)

        if self._pending is None or pt is None or frame_idx <= self._pending[0]:
            return None

        start_frame, start_pt = self._pending
        self._pending = None
        dt = frame_idx - start_frame
        dx = pt[0] - start_pt[0]
        dy = pt[1] - start_pt[1]
        speed = float(np.hypot(dx, dy) / dt)
        angle_deg = float(np.degrees(np.arctan2(dy / dt, dx / dt)))
        hit = HitMeasurement(start_frame, (int(start_pt[0]), int(start_pt[1])), speed, angle_deg,
                             self.is_smash(speed, angle_deg, start_pt[1]))
        logger.debug("Hit at frame %d: speed=%.1f angle=%.1f y=%.0f", start_frame, speed, angle_deg,
                     start_pt[1])
        if hit.is_smash:
            hit.smash_number = self.count + 1
            self.events.append(hit)
        return hit
