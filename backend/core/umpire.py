"""
Line calls for shuttles landing in the near half-court, from shuttle tracks.

A flight = one shuttle track (the tracker starts a new track after every hit). When a
track ends, the flight is judged only if it looks like a landing: it rose above the net
threshold, fell by a clear amount and then came to rest (near-zero image speed for a few
frames) instead of leaving the frame or being hit again. A resting point on a person, or
one that does not map onto the floor around the court (ceiling lights, walls), is not a
landing.
"""
import csv
import os
from dataclasses import dataclass, asdict

import numpy as np

from . import court_model

SINGLES = "singles"
DOUBLES = "doubles"


@dataclass
class Call:
    frame_idx: int
    result: str            # "IN" or "OUT"
    image_pt: tuple
    world_pt: tuple        # meters, near half-court frame
    margin_m: float        # signed distance to the nearest boundary line, > 0 inside
    close_call: bool       # |margin| below the calibration uncertainty

    def to_dict(self):
        d = asdict(self)
        d["world_pt"] = [round(float(v), 3) for v in self.world_pt]
        d["image_pt"] = [int(v) for v in self.image_pt]
        d["margin_m"] = round(float(self.margin_m), 3)
        d["close_call"] = bool(self.close_call)
        return d


def court_bounds(match_type):
    """(x_min, x_max, y_min, y_max) of the near half-court rally area for this match type."""
    y_min, y_max = court_model.SIDELINE_SINGLES_Y if match_type == SINGLES else court_model.SIDELINE_DOUBLES_Y
    return (court_model.BASELINE_X, court_model.NET_X, y_min, y_max)


def judge(world_pt, match_type=SINGLES, close_call_m=0.10):
    """
    ("IN" | "OUT", signed margin in meters). Lines are "in": the model uses line centres,
    so the boundary is pushed out by half a line width.
    """
    x_min, x_max, y_min, y_max = court_bounds(match_type)
    hw = court_model.LINE_HALF_WIDTH
    x, y = world_pt
    margin = min(x - (x_min - hw), (x_max + hw) - x, y - (y_min - hw), (y_max + hw) - y)
    return ("IN" if margin >= 0 else "OUT"), float(margin), bool(abs(margin) < close_call_m)


def match_type_from_metadata(video_name, metadata_path=None):
    """'don' -> singles, 'doi' -> doubles, read from data/metadata.csv (default singles)."""
    metadata_path = metadata_path or os.path.join(os.path.dirname(__file__), "..", "..", "data", "metadata.csv")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("file") == os.path.basename(video_name):
                    return DOUBLES if row.get("loai_tran", "").strip().lower() in ("doi", "đôi") else SINGLES
    return SINGLES


class RallyUmpire:
    def __init__(self, H_inv, match_type=SINGLES, frame_size=None, roi=None, net_top_y=None,
                 edge_margin_px=25, rest_speed_px=2.0, rest_frames=2, close_call_m=0.10, min_flight_len=5,
                 min_descent_px=80, floor_region=(-3.0, court_model.NET_X, -3.0, court_model.COURT_WIDTH + 3.0)):
        """
        frame_size: (h, w) of the frame; with roi, defines the edges a shuttle can exit through.
        net_top_y: image row the flight must rise above (anti-pickup filter); None disables it.
        rest_speed_px / rest_frames: the shuttle is considered on the floor once it moves less
            than rest_speed_px per frame for rest_frames consecutive frames, after falling at
            least min_descent_px from the highest point of the flight.
        floor_region: world (x_min, x_max, y_min, y_max) where a landing can physically be seen.
        """
        self.H_inv = H_inv
        self.match_type = match_type
        self.net_top_y = net_top_y
        self.edge_margin = edge_margin_px
        self.rest_speed = rest_speed_px
        self.rest_frames = rest_frames
        self.close_call_m = close_call_m
        self.min_flight_len = min_flight_len
        self.min_descent = min_descent_px
        self.floor_region = floor_region

        if roi is not None:
            self.bounds = roi
        elif frame_size is not None:
            self.bounds = (0, 0, frame_size[1], frame_size[0])
        else:
            self.bounds = None

        self._flight = []   # [(frame_idx, (x, y), on_person)] detections of the current track
        self._was_active = False

    def update(self, frame_idx, pt, track_active, people_boxes=None):
        """
        Feed one frame. people_boxes: boxes of every person in the frame (players included).
        Returns a Call when a flight just ended with a judgeable landing.
        """
        if pt is not None:
            on_person = any(x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2 for x1, y1, x2, y2 in (people_boxes or []))
            self._flight.append((frame_idx, pt, on_person))
        call = None
        if self._was_active and not track_active:
            call = self._judge_flight(self._flight)
            self._flight = []
        elif not track_active and pt is None:
            self._flight = []
        self._was_active = track_active
        return call

    def _near_edge(self, pt):
        if self.bounds is None:
            return False
        x0, y0, x1, y1 = self.bounds
        m = self.edge_margin
        return pt[0] < x0 + m or pt[0] > x1 - m or pt[1] < y0 + m or pt[1] > y1 - m

    def landing_point(self, flight):
        """
        Index of the first point where the shuttle stays (almost) still for rest_frames frames
        after having fallen at least min_descent px from the top of the flight.
        None if it never rests (it was hit again or left the view).
        """
        if len(flight) < self.min_flight_len:
            return None
        frames = np.array([f[0] for f in flight], dtype=float)
        pts = np.array([f[1] for f in flight], dtype=float)
        dt = np.maximum(np.diff(frames), 1.0)
        speed = np.linalg.norm(np.diff(pts, axis=0) / dt[:, None], axis=1)
        top_y = np.minimum.accumulate(pts[:, 1])
        still = 0
        for i in range(len(speed)):
            if speed[i] < self.rest_speed and pts[i + 1, 1] - top_y[i + 1] >= self.min_descent:
                still += 1
                if still >= self.rest_frames:
                    return i + 1 - self.rest_frames
            else:
                still = 0
        return None

    def _judge_flight(self, flight):
        if not flight or self.H_inv is None:
            return None
        if self.net_top_y is not None and min(f[1][1] for f in flight) > self.net_top_y:
            return None  # never rose above the net: pickup / rolling on the floor
        i = self.landing_point(flight)
        if i is None:
            return None
        frame_idx, img_pt, on_person = flight[i]
        if on_person or self._near_edge(img_pt):
            return None
        world = court_model.img_to_world([img_pt], self.H_inv)[0]
        if not court_model.in_region(world, self.floor_region):
            return None  # not on the visible floor of the near half (far half, walls, ceiling)
        result, margin, close = judge(world, self.match_type, self.close_call_m)
        return Call(frame_idx, result, tuple(img_pt), (float(world[0]), float(world[1])), margin, close)
