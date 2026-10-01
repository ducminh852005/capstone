"""
Smash detection from shuttle tracks.

A hit is recognised when the shuttle tracker starts a new track because the physics
filter broke the previous one: the shuttle changed direction or speed sharply on contact
with the racket (ShuttleDetector.start_source == "physics"). A track that starts from
nothing ("new": the shuttle was simply not seen before) is measured too, but is not a hit --
it may just be a shuttle first noticed while falling fast -- so it does not count as a smash
unless config.SMASH_REQUIRE_PHYSICS_HIT is turned off.

The hit's speed is the displacement between the new track's start and its first real
detection, divided by the frames between them (px/frame). Note that the start is the first
detection after the hit, up to one TrackNet stride after the actual contact. A hit counts
as a smash when it is fast enough, starts low enough in the image (not above the play area)
and heads in an accepted direction.

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
    source: str = "physics"  # how the track started: "physics" (a hit), "stop" (came to rest) or "new"
    dt_frames: int = 0      # frames between the two detections the speed was measured on


class SmashDetector:
    def __init__(self, speed_threshold: Optional[float] = None, min_y: Optional[float] = None,
                 min_angle: Optional[float] = None, max_angle: Optional[float] = None,
                 min_dt_frames: Optional[int] = None):
        """
        speed_threshold: px/frame a hit must exceed (default config.SMASH_SPEED_THRESHOLD).
        min_y: hits above this image row are ignored (default config.SMASH_MIN_Y).
        min_angle / max_angle: accepted direction range in degrees (config.SMASH_MIN_ANGLE /
            SMASH_MAX_ANGLE).
        min_dt_frames: measure the speed to the first detection at least this many frames after
            the track start (default config.SMASH_MIN_DT_FRAMES).
        """
        self._min_dt = min_dt_frames
        self._speed_threshold = speed_threshold
        self._min_y = min_y
        self._min_angle = min_angle
        self._max_angle = max_angle
        self.events = []            # smash HitMeasurements, in order
        self._pending = None        # (start_frame, start_pt, source) waiting for a second detection

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

    def is_smash(self, speed, angle_deg, y, source="physics"):
        """True if a hit with this speed, direction, start row and origin qualifies as a smash."""
        if config.SMASH_REQUIRE_PHYSICS_HIT and source != "physics":
            return False
        return (speed > self.speed_threshold
                and y >= self.min_y
                and self.min_angle <= angle_deg <= self.max_angle)

    def update(self, frame_idx, pt, track_active, track_len, track_pos,
               start_source) -> Optional[HitMeasurement]:
        """
        Feed one frame after ShuttleDetector.detect().

        pt: detection this frame (x, y) or None.
        track_active / track_len: ShuttleDetector state; track_len == 0 on the frame a
            track starts.
        track_pos: (x, y) estimate of the shuttle on that start frame, used when there is no
            detection on it (e.g. ShuttleDetector.kf.x[:2]).
        start_source: ShuttleDetector.start_source, read on the frame the track starts:
            "physics" (restart at a racket hit), "stop" (the shuttle came to rest) or "new".

        Returns a HitMeasurement on the frame the second detection arrives (whether or not
        it is a smash), otherwise None.
        """
        if not track_active:
            self._pending = None    # track lost before a second point: discard
            return None

        if track_len == 0:
            start_pt = pt if pt is not None else track_pos
            self._pending = (frame_idx, start_pt, start_source)

        min_dt = config.SMASH_MIN_DT_FRAMES if self._min_dt is None else self._min_dt
        if self._pending is None or pt is None or frame_idx - self._pending[0] < max(min_dt, 1):
            return None

        start_frame, start_pt, source = self._pending
        self._pending = None
        dt = frame_idx - start_frame
        dx = pt[0] - start_pt[0]
        dy = pt[1] - start_pt[1]
        speed = float(np.hypot(dx, dy) / dt)
        angle_deg = float(np.degrees(np.arctan2(dy / dt, dx / dt)))
        hit = HitMeasurement(start_frame, (int(start_pt[0]), int(start_pt[1])), speed, angle_deg,
                             self.is_smash(speed, angle_deg, start_pt[1], source),
                             source=source, dt_frames=dt)
        logger.debug("Track start at frame %d (%s): speed=%.1f angle=%.1f y=%.0f", start_frame, source,
                     speed, angle_deg, start_pt[1])
        if hit.is_smash:
            hit.smash_number = self.count + 1
            self.events.append(hit)
        return hit
