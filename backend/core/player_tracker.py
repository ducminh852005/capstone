import cv2
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from . import court_model
from . import config
from .pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)


@dataclass
class PlayerObs:
    player_id: int
    track_id: int
    bbox: tuple                # (x1, y1, x2, y2) full-frame pixels
    foot_px: tuple             # (x, y) full-frame pixels
    foot_world: tuple          # (x, y) meters, None without calibration
    source: str                # "foot" / "ankle" (MediaPipe), "cached" (offset), "bbox" (fallback, low confidence)


class PlayerSelector:
    """
    Chooses at most `max_players` players on the near half-court from all tracks.

    - Score of a track = fraction of the last `window_s` seconds its foot was inside the
      buffered half-court (short-lived tracks are normalised by `min_track_s`, so a new
      track needs ~min_track_s * min_score seconds inside before it can be selected).
    - Hysteresis: a selected player keeps the slot while its track is alive; a challenger
      only takes over if its score beats the player's by `switch_margin`.
    - Re-identification: when the player's track disappears, a NEW track that appears inside
      the court within `reid_window_s` and within `reid_dist_m` of the last position inherits
      the player_id. After `reid_window_s` without a match the slot is released.
    """

    def __init__(self, fps=60.0, window_s=config.SELECTOR_WINDOW_S,
                 min_track_s=config.SELECTOR_MIN_TRACK_S,
                 min_score=config.SELECTOR_MIN_SCORE,
                 switch_margin=config.SELECTOR_SWITCH_MARGIN,
                 reid_window_s=config.SELECTOR_REID_WINDOW_S,
                 reid_dist_m=config.SELECTOR_REID_DIST_M,
                 max_players=1):
        self.window = max(int(window_s * fps), 1)
        self.min_track = max(int(min_track_s * fps), 1)
        self.min_score = min_score
        self.switch_margin = switch_margin
        self.reid_window = int(reid_window_s * fps)
        self.reid_dist = reid_dist_m
        self.max_players = max_players

        self.history = {}      # track_id -> deque[bool] in-court flags
        self.first_seen = {}   # track_id -> frame_idx
        self.last_seen = {}    # track_id -> frame_idx
        self.players = {}      # player_id -> {"track": id, "last_seen": idx, "last_world": xy}
        self._next_player_id = 1

    def score(self, track_id):
        hist = self.history.get(track_id)
        if not hist:
            return 0.0
        return sum(hist) / max(len(hist), self.min_track)

    def update(self, frame_idx, observations):
        """
        observations: {track_id: (foot_world_xy or None, in_court: bool)} for tracks seen this frame.
        Returns {player_id: track_id} for players visible in this frame.
        """
        for tid, (_, inside) in observations.items():
            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.window)
                self.first_seen[tid] = frame_idx
            self.history[tid].append(bool(inside))
            self.last_seen[tid] = frame_idx

        assigned = {p["track"] for p in self.players.values()}

        for pid in list(self.players):
            p = self.players[pid]
            tid = p["track"]
            if tid in observations:
                p["last_seen"] = frame_idx
                if observations[tid][0] is not None:
                    p["last_world"] = observations[tid][0]
                # challenger with a clearly better in-court score takes over
                challenger = self._best_candidate(observations, assigned)
                if challenger is not None and self.score(challenger) > self.score(tid) + self.switch_margin:
                    del self.players[pid]
                    assigned.discard(tid)
                continue

            match = self._reid(p, observations, assigned)
            if match is not None:
                p["track"] = match
                p["last_seen"] = frame_idx
                if observations[match][0] is not None:
                    p["last_world"] = observations[match][0]
                assigned.add(match)
            elif frame_idx - p["last_seen"] > self.reid_window:
                del self.players[pid]
                assigned.discard(tid)

        while len(self.players) < self.max_players:
            cand = self._best_candidate(observations, assigned)
            if cand is None:
                break
            self.players[self._next_player_id] = {
                "track": cand, "last_seen": frame_idx, "last_world": observations[cand][0]}
            self._next_player_id += 1
            assigned.add(cand)

        self._forget_stale(frame_idx)
        return {pid: p["track"] for pid, p in self.players.items() if p["track"] in observations}

    def _best_candidate(self, observations, assigned):
        best, best_score = None, self.min_score
        for tid, (_, inside) in observations.items():
            if tid in assigned or not inside:
                continue
            s = self.score(tid)
            if s >= best_score:
                best, best_score = tid, s
        return best

    def _reid(self, player, observations, assigned):
        if player["last_world"] is None:
            return None
        best, best_d = None, self.reid_dist
        for tid, (xy, inside) in observations.items():
            if tid in assigned or not inside or xy is None:
                continue
            # only tracks born after the player was lost can be a fragment of that player
            if self.first_seen[tid] < player["last_seen"] - 2:
                continue
            d = float(np.hypot(xy[0] - player["last_world"][0], xy[1] - player["last_world"][1]))
            if d < best_d:
                best, best_d = tid, d
        return best

    def _forget_stale(self, frame_idx):
        active = {p["track"] for p in self.players.values()}
        for tid in [t for t, last in self.last_seen.items() if frame_idx - last > self.window and t not in active]:
            self.history.pop(tid, None)
            self.first_seen.pop(tid, None)
            self.last_seen.pop(tid, None)


class PlayerTracker:
    def __init__(self, model_path=None, conf_thresh=config.PLAYER_YOLO_CONF, fps=60.0, pose_variant="lite",
                 pose_every=config.PLAYER_POSE_EVERY, yolo_every=config.PLAYER_YOLO_EVERY,
                 imgsz=config.PLAYER_YOLO_IMGSZ, device=None, half=None, selector_kwargs=None):
        """
        YOLOv8 + ByteTrack person tracking, MediaPipe foot points and player selection.

        fps: rate at which process() is called (after any frame skipping); it converts the
             selector's time windows to frame counts.
        pose_every: MediaPipe runs on the selected player's track every N frames and on other
             tracks near the court every 3N frames (staggered by track id); in between, the
             cached foot offset relative to the bbox is used.
        yolo_every: run YOLO+ByteTrack every N calls to process(); on skipped calls the boxes/ids
             from the last real detection are reused as-is (frozen, not re-predicted) -- cheap
             because a player's position barely moves across 1-2 frames, and the shuttle
             detector's exclude/player boxes and the pose/selection logic downstream only need
             an approximate box, not a fresh one every frame.
        """
        if model_path is None:
            model_path = config.PLAYER_YOLO_MODEL_PATH
        logger.info("Loading YOLO model from %s...", model_path)
        self.model = YOLO(str(model_path))
        self.conf_thresh = conf_thresh
        self.tracker_config = "bytetrack.yaml"
        self.imgsz = imgsz

        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        self.device = device if device is not None else (0 if cuda else "cpu")
        self.half = half if half is not None else cuda

        self.pose_estimator = PoseEstimator(variant=pose_variant)
        self.pose_every = max(int(pose_every), 1)
        self.yolo_every = max(int(yolo_every), 1)
        self.selector = PlayerSelector(fps=fps, **(selector_kwargs or {}))

        # {track_id: (dx / bbox_h, dy / bbox_h)} foot offset from the bbox bottom-centre
        self.foot_offsets = {}
        self._pose_tried = set()  # tracks that already had a first pose attempt
        self._n_updates = 0  # pose schedule counter (frame_idx may skip frames)
        self._detect_calls = 0  # yolo schedule counter, counts process() calls
        self._roi_cache = (None, None, None)  # (H id, frame shape, roi)
        self.last_boxes = np.empty((0, 4), np.float32)  # every person box of the last frame
        self.last_ids = np.empty((0,), np.int64)         # matching track ids of last_boxes

    def track_frame(self, frame, persist=True):
        """Raw YOLO + ByteTrack result for one frame (class 0 = person)."""
        results = self.model.track(
            frame,
            persist=persist,
            classes=[0],
            conf=self.conf_thresh,
            tracker=self.tracker_config,
            imgsz=self.imgsz,
            device=self.device,
            quantize=16 if self.half else None,
            verbose=False
        )
        return results[0]

    def detect(self, frame, roi=None):
        """Tracks people inside roi=(x0, y0, x1, y1). Returns (boxes Nx4, ids N) in full-frame pixels."""
        x0, y0 = 0, 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            frame = frame[y0:y1, x0:x1]
        res = self.track_frame(frame, persist=True)
        if res.boxes.id is None:
            return np.empty((0, 4), np.float32), np.empty((0,), np.int64)
        boxes = res.boxes.xyxy.cpu().numpy()
        boxes[:, [0, 2]] += x0
        boxes[:, [1, 3]] += y0
        return boxes, res.boxes.id.cpu().numpy().astype(np.int64)

    def roi_for(self, H, frame_shape):
        if H is None:
            return None
        key = (id(H), frame_shape[:2])
        if self._roi_cache[:2] != key:
            self._roi_cache = (*key, court_model.player_roi(H, frame_shape))
        return self._roi_cache[2]

    def _foot_point(self, frame, track_id, bbox, is_player):
        x1, y1, x2, y2 = bbox
        bh = max(y2 - y1, 1)
        bottom = ((x1 + x2) / 2.0, y2)

        every = self.pose_every if is_player else 3 * self.pose_every
        if track_id not in self._pose_tried or (self._n_updates + track_id) % every == 0:
            self._pose_tried.add(track_id)
            pt, source = self.pose_estimator.extract_foot_point(frame, bbox)
            if pt is not None:
                self.foot_offsets[track_id] = ((pt[0] - bottom[0]) / bh, (pt[1] - bottom[1]) / bh)
                return pt, source

        if track_id in self.foot_offsets:
            dx, dy = self.foot_offsets[track_id]
            return (bottom[0] + dx * bh, bottom[1] + dy * bh), "cached"
        return bottom, "bbox"

    def non_player_boxes(self, players):
        """Boxes of every person tracked in the last frame who is not a selected player."""
        return [tuple(b) for b in self.last_boxes
                if not any(np.allclose(b, p.bbox) for p in players.values())]

    def update(self, frame, frame_idx, boxes, ids, H_inv):
        """
        Computes foot points, gates tracks to the buffered near half-court (in meters) and
        selects the player. Returns {player_id: PlayerObs}.
        """
        self._n_updates += 1
        player_tracks = {p["track"] for p in self.selector.players.values()}
        observations, details = {}, {}
        for box, tid in zip(boxes, ids):
            tid = int(tid)
            bbox = tuple(float(v) for v in box)
            bottom = ((bbox[0] + bbox[2]) / 2.0, bbox[3])

            if H_inv is not None:
                # cheap pre-filter: skip pose for people far from the court (audience, next court)
                if not court_model.in_region(court_model.img_to_world([bottom], H_inv)[0], court_model.REGION_PREFILTER):
                    observations[tid] = (None, False)
                    continue

            foot_px, source = self._foot_point(frame, tid, bbox, tid in player_tracks)
            if H_inv is not None:
                foot_world = tuple(float(v) for v in court_model.img_to_world([foot_px], H_inv)[0])
                inside = court_model.in_region(foot_world)
            else:
                foot_world, inside = None, True
            observations[tid] = (foot_world, inside)
            details[tid] = (bbox, foot_px, foot_world, source)

        selected = self.selector.update(frame_idx, observations)
        live = set(observations)
        for tid in [t for t in self._pose_tried if t not in live and t not in self.selector.history]:
            self._pose_tried.discard(tid)
            self.foot_offsets.pop(tid, None)

        players = {}
        for pid, tid in selected.items():
            if tid in details:
                bbox, foot_px, foot_world, source = details[tid]
                players[pid] = PlayerObs(pid, tid, bbox, foot_px, foot_world, source)
        return players

    def process(self, frame, frame_idx, H=None, H_inv=None):
        """
        Full per-frame step: ROI from calibration -> YOLO/ByteTrack -> foot points -> selection.
        YOLO/ByteTrack itself only runs every `yolo_every` calls (see __init__); other calls
        reuse the last boxes/ids untouched.
        """
        if self._detect_calls % self.yolo_every == 0:
            self.last_boxes, self.last_ids = self.detect(frame, self.roi_for(H, frame.shape))
        self._detect_calls += 1
        return self.update(frame, frame_idx, self.last_boxes, self.last_ids, H_inv)

    def draw_tracking(self, frame, players):
        """Bounding box, player/track id and foot point for each selected player."""
        annotated_frame = frame.copy()
        for pid, p in players.items():
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"P{pid} (T{p.track_id})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            color = (0, 165, 255) if p.source == "bbox" else (0, 0, 255)
            cv2.circle(annotated_frame, (int(p.foot_px[0]), int(p.foot_px[1])), 6, color, -1)
        return annotated_frame
