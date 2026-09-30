"""
Line calls for shuttles landing on either half-court, from shuttle tracks.

A flight = one shuttle track (the tracker starts a new track after every hit). When a
flight ends -- the track dies, or, if the caller passes `flight_id`, a hit starts a new
track -- it is judged only if it looks like a landing: it rose above the net threshold,
fell by a clear amount and then hit the floor -- a sharp break from the smooth free-fall
curve, not necessarily the point where it finally stops moving, since a real landing can
bounce or skid a short distance afterwards. The line call is decided by that
first-contact point (see `landing_point`), and confirmed by checking the shuttle calms
down afterwards instead of flying off on another energetic path. A resting point on a
person, or one that does not map onto the floor around the court (ceiling lights, walls),
is not a landing. The landing's x position (vs. NET_X) decides which half it fell in, and
that half is judged against its own boundary lines.

Every Call says how it was obtained (`Call.method`) and how precise the position can be
(`Call.uncertainty_m`): far from the camera one image pixel spans more than 10 cm of court.
"""
import csv
import logging
import os
from dataclasses import dataclass, asdict
from typing import NamedTuple, Optional

import numpy as np

from . import court_model
from . import config

logger = logging.getLogger(__name__)

SINGLES = "singles"
DOUBLES = "doubles"

METHOD_CONTACT = "contact"                        # impact found, then the shuttle calmed down
METHOD_UNCONFIRMED = "contact_unconfirmed"        # impact found, but the track ended right after it
METHOD_LOST = "lost"                              # track lost mid-fall; position extrapolated
METHOD_RESTING = "resting"                        # fall not seen through; called at the first resting position


@dataclass
class Call:
    frame_idx: int
    result: str            # "IN" or "OUT"
    half: str              # "near" or "far": which half-court the shuttle landed in
    image_pt: tuple
    world_pt: tuple        # meters, full-court frame
    margin_m: float        # signed distance to the nearest boundary line, > 0 inside
    close_call: bool       # |margin_m| below max(close_call_m, uncertainty_m)
    method: str = METHOD_CONTACT
    uncertainty_m: float = 0.0   # court length (m) spanned by one image pixel at the landing

    def to_dict(self):
        d = asdict(self)
        d["world_pt"] = [round(float(v), 3) for v in self.world_pt]
        d["image_pt"] = [int(v) for v in self.image_pt]
        d["margin_m"] = round(float(self.margin_m), 3)
        d["close_call"] = bool(self.close_call)
        d["uncertainty_m"] = round(float(self.uncertainty_m), 3)
        return d


class FlightPoint(NamedTuple):
    """One detection of the current flight."""
    frame_idx: int
    pt: tuple              # (x, y) image pixels
    on_person: bool        # the point lies inside some person's box


def default_floor_region():
    """World (x_min, x_max, y_min, y_max): the full court plus a buffer on every side."""
    b = config.UMPIRE_FLOOR_BUFFER_M
    return (-b, court_model.COURT_LENGTH + b, -b, court_model.COURT_WIDTH + b)


def court_bounds(match_type, half="near"):
    """(x_min, x_max, y_min, y_max) of the given half-court's rally area for this match type."""
    y_min, y_max = court_model.SIDELINE_SINGLES_Y if match_type == SINGLES else court_model.SIDELINE_DOUBLES_Y
    if half == "near":
        return (court_model.BASELINE_X, court_model.NET_X, y_min, y_max)
    return (court_model.NET_X, court_model.COURT_LENGTH, y_min, y_max)


def judge(world_pt, match_type=SINGLES, close_call_m=config.UMPIRE_CLOSE_CALL_M, half="near",
          uncertainty_m=0.0):
    """
    ("IN" | "OUT", signed margin in meters, close-call flag). Lines are "in": the model uses
    line centres, so the boundary is pushed out by half a line width. The call is "close"
    when the margin is below close_call_m or below the position uncertainty (uncertainty_m).
    """
    x_min, x_max, y_min, y_max = court_bounds(match_type, half)
    hw = court_model.LINE_HALF_WIDTH
    x, y = world_pt
    margin = min(x - (x_min - hw), (x_max + hw) - x, y - (y_min - hw), (y_max + hw) - y)
    return ("IN" if margin >= 0 else "OUT"), float(margin), bool(abs(margin) < max(close_call_m, uncertainty_m))


def match_type_from_metadata(video_name, metadata_path=None):
    """
    'don' -> singles, 'doi' -> doubles, read from data/metadata.csv. Falls back to singles
    (with a warning: it changes which lines are in play) if the file or the row is missing.
    """
    metadata_path = metadata_path or config.METADATA_PATH
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("file") == os.path.basename(video_name):
                    return DOUBLES if row.get("loai_tran", "").strip().lower() in ("doi", "đôi") else SINGLES
        logger.warning("No metadata row for %s in %s; assuming %s", video_name, metadata_path, SINGLES)
    else:
        logger.warning("Metadata file %s not found; assuming %s", metadata_path, SINGLES)
    return SINGLES


class RallyUmpire:
    def __init__(self, H_inv, match_type=SINGLES, frame_size=None, roi=None, net_top_y=None,
                 edge_margin_px=config.UMPIRE_EDGE_MARGIN_PX,
                 rest_speed_px=config.UMPIRE_REST_SPEED_PX,
                 rest_frames=config.UMPIRE_REST_FRAMES,
                 close_call_m=config.UMPIRE_CLOSE_CALL_M,
                 min_flight_len=config.UMPIRE_MIN_FLIGHT_LEN,
                 min_descent_px=config.UMPIRE_MIN_DESCENT_PX,
                 settle_search_frames=config.UMPIRE_SETTLE_SEARCH_FRAMES,
                 floor_region=None,
                 allow_extrapolation=config.UMPIRE_ALLOW_EXTRAPOLATION,
                 allow_resting=config.UMPIRE_ALLOW_RESTING):
        """
        frame_size: (h, w) of the frame; with roi, defines the edges a shuttle can exit through.
        net_top_y: image row the flight must rise above (anti-pickup filter); None disables it.
        rest_speed_px / rest_frames: the impact is where the vertical speed falls to
            rest_speed_px per frame or less after the shuttle has been falling faster; the
            shuttle must then calm down to less than rest_speed_px per frame for rest_frames
            consecutive frames within settle_search_frames steps, confirming it was a landing
            and not a mid-flight blip. rest_speed_px is given at the net; nearer to the camera
            it is scaled up to account for perspective (see _local_rest_speed).
        min_descent_px: an impact only counts once the shuttle has fallen at least this many
            pixels below the highest point of the flight (rules out gentle tosses and
            wobbles near the top of an arc).
        settle_search_frames: how many frames after a candidate impact to look for it calming
            down (see rest_speed_px/rest_frames) before accepting it as a real landing.
        floor_region: world (x_min, x_max, y_min, y_max) where a landing can physically be
            seen; default_floor_region() if None.
        allow_extrapolation: when a flight ends while the shuttle is still falling fast,
            extrapolate where it would reach the floor (Call.method == "lost"). Off by
            default: the shuttle is in the air, so its pixel does not map to a floor point,
            and the extrapolation just runs to the edge of floor_region.
        allow_resting: a flight that ended while the shuttle was falling, followed soon after
            by detections of it lying still on the floor (UMPIRE_REST_RUN_POINTS in a row),
            is called at the first of those resting positions (Call.method == "resting").
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
        self.settle_search_frames = settle_search_frames
        self.floor_region = default_floor_region() if floor_region is None else floor_region
        self.allow_extrapolation = allow_extrapolation
        self.allow_resting = allow_resting
        self._fall = None       # (frame_idx, pt) where the last flight ended while still falling

        self._H = None                  # world -> image, inverse of H_inv
        self._ref_scale = None          # metres per image pixel at the net
        if H_inv is not None:
            self._init_perspective()

        if roi is not None:
            self.bounds = roi
        elif frame_size is not None:
            self.bounds = (0, 0, frame_size[1], frame_size[0])
        else:
            self.bounds = None

        self._flight = []   # [FlightPoint] detections of the current flight
        self._pending = None    # a flight that ended at a hit or bounce, waiting to see what the shuttle does next
        self._was_active = False
        self._flight_id = None

    def _init_perspective(self):
        """Cache H and the reference (net) scale used by _local_rest_speed."""
        try:
            self._H = np.linalg.inv(self.H_inv)
        except np.linalg.LinAlgError:
            logger.warning("H_inv is singular; rest speed will not be perspective-corrected")
            return
        net_pt = court_model.world_to_img([[court_model.NET_X, court_model.COURT_WIDTH / 2]], self._H)[0]
        self._ref_scale = float(self._metres_per_px(net_pt[None, :])[0])

    def _metres_per_px(self, img_pts):
        """Local ground scale (m/px) along the image vertical at each (N, 2) image point."""
        probe = config.UMPIRE_PERSPECTIVE_PROBE_PX
        w1 = court_model.img_to_world(img_pts, self.H_inv)
        w2 = court_model.img_to_world(img_pts + np.array([0, -probe]), self.H_inv)
        return np.maximum(np.linalg.norm(w1 - w2, axis=1) / probe, config.UMPIRE_MIN_METERS_PER_PX)

    def _local_rest_speed(self, pts):
        """
        Per-point rest-speed threshold (px/frame) for the (N, 2) image points `pts`.
        The same physical speed covers more pixels near the camera than near the net, so
        the threshold grows with the ratio of the net scale to the local scale.
        """
        floor = min(self.rest_speed, config.UMPIRE_REST_SPEED_FLOOR_PX)
        if self._ref_scale is None:
            return np.full(len(pts), float(self.rest_speed))
        return np.maximum(self.rest_speed * (self._ref_scale / self._metres_per_px(pts)), floor)

    def update(self, frame_idx, pt, track_active, people_boxes=None, flight_id=None):
        """
        Feed one frame. people_boxes: boxes of every person in the frame (players included).
        flight_id: ShuttleDetector.flight_id. A racket hit (or a bounce) restarts the track
            within a single detect() call, so `track_active` never drops; when flight_id
            changes, this frame's point starts the next flight. The ended flight is not
            discarded: a shuttle that touches the floor reverses its vertical velocity, which the
            tracker cannot tell from a racket hit, so the flight is judged once the next
            rest_frames + 1 points have shown whether the shuttle calmed down (a landing) or
            flew on (a hit). Without flight_id, only the track ending separates flights.
        Returns a Call when a flight just ended with a judgeable landing.
        """
        call = None
        if flight_id is not None:
            if self._flight_id is not None and flight_id != self._flight_id and self._flight:
                call = self._resolve_pending()      # an earlier boundary; this flight is its look-ahead
                self._pending = self._flight
                self._flight = []
            self._flight_id = flight_id
        if pt is not None:
            on_person = any(x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2 for x1, y1, x2, y2 in (people_boxes or []))
            self._flight.append(FlightPoint(frame_idx, pt, on_person))
        if self._pending is not None and len(self._flight) >= self.rest_frames + 1:
            call = call or self._resolve_pending()
        if self._was_active and not track_active:
            call = call or self._resolve_pending()
            ended = self._judge_flight(self._flight)
            call = call or ended
            self._flight = []
        elif not track_active and pt is None:
            call = call or self._resolve_pending()
            self._flight = []
        self._was_active = track_active
        return call

    def _resolve_pending(self) -> Optional[Call]:
        """Judge the flight that ended at a hit/bounce, using the points seen since as look-ahead.
        Only a contact at (or before) the boundary counts; later ones belong to the next flight."""
        pending, self._pending = self._pending, None
        if not pending:
            return None
        series = pending + self._flight[:self.rest_frames + 1]
        return self._landing_call(series, max_idx=len(pending))

    def _near_edge(self, pt):
        if self.bounds is None:
            return False
        x0, y0, x1, y1 = self.bounds
        m = self.edge_margin
        return pt[0] < x0 + m or pt[0] > x1 - m or pt[1] < y0 + m or pt[1] > y1 - m

    def landing_point(self, flight):
        """
        Index of the shuttle's first ground contact, i.e. the first sharp break from the
        smooth free-fall curve after it has fallen at least min_descent px from the top of
        the flight: the vertical speed, after exceeding the rest speed while falling, drops
        to the rest speed or less -- a bounce or a skid loses its downward speed on contact,
        unlike a shuttle still falling.

        That break is only accepted once the motion afterwards calms down to near-rest
        within settle_search_frames (see _settle_evidence) -- confirming a real landing
        that then bounced/skidded to a stop, as opposed to a soft return that only looked
        like a deceleration. The reported point is the contact itself, not wherever it
        finally comes to rest afterwards.

        None if no such landing is found (still in flight, hit again, or left the view).
        """
        found = self._find_landing(flight)
        return None if found is None else found[0]

    def _find_landing(self, flight, max_idx=None):
        """(index, confirmed) of the first ground contact (see landing_point), or None.
        confirmed is False when the track ended before the shuttle could be seen calming down.
        max_idx: ignore a contact later than this index in `flight`."""
        if len(flight) < self.min_flight_len:
            return None
        frames = np.array([f[0] for f in flight], dtype=float)
        pts = np.array([f[1] for f in flight], dtype=float)
        dt = np.maximum(np.diff(frames), 1.0)
        vy = np.diff(pts[:, 1]) / dt
        speed = np.linalg.norm(np.diff(pts, axis=0) / dt[:, None], axis=1)
        local_rest_speed = self._local_rest_speed(pts[:-1])
        top_y = np.minimum.accumulate(pts[:, 1])      # highest point of the flight so far

        peak_vy = 0.0
        for i in range(len(vy)):
            peak_vy = max(peak_vy, vy[i])
            impact = peak_vy > local_rest_speed[i] and vy[i] <= local_rest_speed[i]
            if not impact or max(pts[i, 1], pts[i + 1, 1]) - top_y[i] < self.min_descent:
                continue
            evidence = self._settle_evidence(speed, i + 1, local_rest_speed)
            if evidence is None:
                continue
            # Pick the lowest-on-screen contact frame: if still moving down (vy > 0, skidding
            # along the floor) the next point is lower; if bouncing up (vy <= 0) this one is.
            idx = i + 1 if vy[i] > 0 else i
            img_pt = flight[idx][1]
            # Reject invalid contacts right away: on a person, or off the court floor.
            if flight[idx][2] or self._near_edge(img_pt):
                continue
            if self.H_inv is not None and not self._on_floor(img_pt):
                continue
            if max_idx is not None and idx > max_idx:
                return None
            return idx, evidence == "confirmed"
        return None

    def _settle_evidence(self, speed, start, local_rest_speed=None):
        """
        Did the shuttle calm down after an impact? "confirmed" if speed drops under the rest
        speed for rest_frames consecutive steps starting somewhere in
        [start, start + settle_search_frames); "unconfirmed" if the track ended before there
        were enough steps to tell (nothing contradicts the landing, nothing supports it);
        None if it kept moving.
        """
        if start >= len(speed):
            return "unconfirmed"    # track lost right after the impact (blind spot, out of view)

        still = 0
        for j in range(start, min(start + self.settle_search_frames, len(speed))):
            thresh = local_rest_speed[j] if local_rest_speed is not None else self.rest_speed
            if speed[j] < thresh:
                still += 1
                if still >= self.rest_frames:
                    return "confirmed"
            else:
                still = 0

        # Track ended before rest_frames steps were available, but every remaining step is still.
        if len(speed) - start < self.rest_frames and still == len(speed) - start:
            return "unconfirmed"

        return None

    def _settles_within(self, speed, start, local_rest_speed=None):
        """True if the shuttle calms down (or the track ends before that can be told) after `start`."""
        return self._settle_evidence(speed, start, local_rest_speed) is not None

    def _extrapolated_landing(self, flight):
        """
        (frame_idx, image_pt) where a shuttle lost while still falling fast would reach the
        floor, extrapolating its last velocity; None if it is not falling or never reaches
        the floor region within UMPIRE_EXTRAPOLATE_MAX_FRAMES.
        """
        if len(flight) < config.UMPIRE_EXTRAPOLATE_MIN_POINTS:
            return None
        f_start = flight[-min(config.UMPIRE_EXTRAPOLATE_WINDOW, len(flight))]
        f_end = flight[-1]
        dt = max(1, f_end[0] - f_start[0])
        vx = (f_end[1][0] - f_start[1][0]) / dt
        vy = (f_end[1][1] - f_start[1][1]) / dt
        if vy <= self.rest_speed:
            return None     # not falling fast: nothing to extrapolate

        cur_pt = np.array(f_end[1], dtype=float)
        if self._on_floor(cur_pt):
            return f_end[0], [int(cur_pt[0]), int(cur_pt[1])]
        for step in range(1, config.UMPIRE_EXTRAPOLATE_MAX_FRAMES + 1):
            cur_pt[0] += vx
            cur_pt[1] += vy
            if self._on_floor(cur_pt):
                return int(f_end[0] + step), [int(cur_pt[0]), int(cur_pt[1])]
        return None

    def _on_floor(self, img_pt):
        world = court_model.img_to_world([img_pt], self.H_inv)[0]
        return court_model.in_region(world, self.floor_region)

    def _judge_flight(self, flight) -> Optional[Call]:
        return self._landing_call(flight)

    def _landing_call(self, flight, max_idx=None) -> Optional[Call]:
        """
        Call for the flight, or None. max_idx is set when the flight ended at a hit/bounce
        boundary and is judged with look-ahead points appended (see _resolve_pending); only
        whole flights (max_idx None) take part in the resting-landing bookkeeping.
        """
        if not flight or self.H_inv is None:
            return None
        rose = self.net_top_y is None or min(f[1][1] for f in flight) <= self.net_top_y

        call = None
        if rose:
            found = self._find_landing(flight, max_idx)
            if found is not None:
                idx, confirmed = found
                call = self._make_call(flight[idx][0], flight[idx][1],
                                       METHOD_CONTACT if confirmed else METHOD_UNCONFIRMED)
            elif self.allow_extrapolation:
                landing = self._extrapolated_landing(flight)
                if landing is not None:
                    call = self._make_call(landing[0], landing[1], METHOD_LOST)
        if call is not None:
            self._fall = None
            return call

        if max_idx is None:
            call = self._resting_call(flight)
            if call is not None:
                self._fall = None
                return call
            if rose:
                fall = self._fall_end(flight)
                if fall is not None:
                    self._fall = fall
        return None

    def _make_call(self, frame_idx, img_pt, method) -> Call:
        world = court_model.img_to_world([img_pt], self.H_inv)[0]
        uncertainty = float(self._metres_per_px(np.array([img_pt], dtype=float))[0])
        half = "near" if world[0] < court_model.NET_X else "far"
        result, margin, close = judge(world, self.match_type, self.close_call_m, half=half,
                                      uncertainty_m=uncertainty)
        return Call(frame_idx, result, half, tuple(img_pt), (float(world[0]), float(world[1])), margin, close,
                    method=method, uncertainty_m=uncertainty)

    def _fall_end(self, flight):
        """(frame_idx, pt) of the last detection if the flight ended while still falling after a
        real descent, else None: the shuttle is probably on its way to the floor, unseen."""
        if len(flight) < 2:
            return None
        pts = np.array([f.pt for f in flight], dtype=float)
        if pts[:, 1].max() - pts[:, 1].min() < self.min_descent:
            return None
        dt = max(flight[-1].frame_idx - flight[-2].frame_idx, 1)
        vy = (pts[-1, 1] - pts[-2, 1]) / dt
        if vy <= self._local_rest_speed(pts[-2:-1])[0]:
            return None
        return flight[-1].frame_idx, tuple(flight[-1].pt)

    def _first_still_run(self, flight, n_points):
        """Index of the first detection that starts a run of n_points detections lying still."""
        if len(flight) < n_points:
            return None
        frames = np.array([f.frame_idx for f in flight], dtype=float)
        pts = np.array([f.pt for f in flight], dtype=float)
        dt = np.maximum(np.diff(frames), 1.0)
        speed = np.linalg.norm(np.diff(pts, axis=0) / dt[:, None], axis=1)
        still = speed < self._local_rest_speed(pts[:-1])
        for k in range(len(flight) - n_points + 1):
            if still[k:k + n_points - 1].all():
                return k
        return None

    def _resting_call(self, flight) -> Optional[Call]:
        """
        Landing for a shuttle first seen lying still after a fall that ended unseen: the first
        resting position (Call.method == "resting"). Needs a preceding flight that ended falling
        (self._fall) not longer than UMPIRE_REST_MAX_GAP_FRAMES before, and the resting spot not
        higher in the image than where the fall was last seen.
        """
        if not self.allow_resting or self._fall is None:
            return None
        k = self._first_still_run(flight, config.UMPIRE_REST_RUN_POINTS)
        if k is None:
            return None
        fall_frame, fall_pt = self._fall
        f = flight[k]
        gap = f.frame_idx - fall_frame
        if gap < 0 or gap > config.UMPIRE_REST_MAX_GAP_FRAMES:
            return None
        if f.pt[1] < fall_pt[1] - config.UMPIRE_REST_Y_SLACK_PX:
            return None
        if f.on_person or self._near_edge(f.pt) or not self._on_floor(f.pt):
            return None
        return self._make_call(f.frame_idx, f.pt, METHOD_RESTING)
