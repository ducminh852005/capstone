"""
Analysis of one clip, in the form the HTML viewer (web/telestrator.html) reads: the per-frame
skeletons, player positions, shuttle trail and the events found by the pipeline, plus everything the
page needs to place a drawing on the court (the homography).

build_clip_analysis() turns a ClipRun (core/clip_pipeline.py) into a ClipAnalysis;
ClipAnalysis.to_jsonable() gives the JSON-ready dict that is written as analysis.json and inlined
into index.html (core/clip_report.py).

Conventions of the exported JSON (SCHEMA_VERSION 2):
  - frame indices are clip-local (frame 0 = first frame of clip.mp4, pre-roll included);
  - `_px` values are full-frame image pixels, `_m` values are metres in the court frame of
    court_model.py (x along the court, 0 at the near baseline, NET_X at the net; y across);
  - per-frame lists have one entry per clip frame, null where there is no value;
  - a skeleton is a flat list of 33 x (x_px, y_px) integers (66 numbers), HIDDEN_LANDMARK_PX where
    the joint was not visible (score below config.CLIP_SKELETON_MIN_SCORE).
"""
import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config, court_model, labels_vi
from .clip_pipeline import ClipRun
from .clip_stats import ShotSummary, summarize_shots
from .hit_events import NEAR, HitterParams, ShuttleHit, apply_alternation, attribute_hits, racket_hands
from .movement import MovementParams, MovementStats, compute_movement
from .pose_features import POSE_CONNECTIONS
from .rally import Rally, RallyParams, segment_rallies
from .shots import Shot, ShotThresholds, build_shots
from .umpire import SINGLES

SCHEMA_VERSION = 2

HIDDEN_LANDMARK_PX = -1
"""Exported coordinate of a skeleton joint that was not visible (visible ones are clipped to >= 0)."""

SELFTEST_WORLD_POINTS_M = [
    (court_model.BASELINE_X, court_model.SIDELINE_DOUBLES_Y[0]),
    (court_model.BASELINE_X, court_model.SIDELINE_DOUBLES_Y[1]),
    (court_model.NET_X, court_model.SIDELINE_DOUBLES_Y[0]),
    (court_model.NET_X, court_model.SIDELINE_DOUBLES_Y[1]),
    (court_model.NET_X / 2, court_model.COURT_WIDTH / 2),
    (court_model.COURT_LENGTH * 3 / 4, court_model.COURT_WIDTH / 2),
]
"""Court points whose image position and round-trip world position are exported so the page can
check that its own homography arithmetic agrees with Python's (see court_block)."""


@dataclass
class ClipSource:
    video: str                  # file name of the original video
    start_frame: int            # frame of the original video that is clip frame 0 (pre-roll included)
    preroll_frames: int         # leading frames analysed but not counted in the statistics
    analysis_start_frame: int   # first clip frame that counts (== preroll_frames)


@dataclass
class VideoMeta:
    file: str                   # clip file name, relative to analysis.json / index.html
    fps: float                  # frames per second of the clip
    width: int                  # px
    height: int                 # px
    n_frames: int


@dataclass
class RacketHand:
    """Which hand holds the racket, from the hand that was closest to the shuttle at the player's hits."""
    hand: Optional[str]         # "left" | "right" | None when unknown or tied
    share: float                # fraction of the votes the hand got
    votes: int                  # attributed hits that voted
    source: str                 # "auto" (voted) | "manual" (--hand)


@dataclass
class PlayerTrack:
    """One tracked near-half player over the clip. Lists have one entry per clip frame."""
    player_id: int
    track_ids: List[int]                                    # ByteTrack ids this player was seen under
    n_frames: int                                           # frames the player was tracked
    n_pose_frames: int                                      # frames with a skeleton
    bbox_px: List[Optional[List[float]]]                    # [x1, y1, x2, y2]
    foot_px: List[Optional[List[float]]]                    # [x, y]
    foot_world_m: List[Optional[List[float]]]               # [x, y]
    skeleton: List[Optional[np.ndarray]]                    # int32 (66,): 33 x (x_px, y_px), HIDDEN_LANDMARK_PX if hidden
    racket_hand: Optional[RacketHand] = None
    shot_summary: Optional[ShotSummary] = None
    movement: Optional[MovementStats] = None                # over the analysed stretch (without the pre-roll)


@dataclass
class ClipAnalysis:
    schema_version: int
    generator: Dict[str, Any]
    source: ClipSource
    video: VideoMeta
    court: Dict[str, Any]
    pose: Dict[str, Any]
    shuttle: Dict[str, Any]                                 # px: per-frame [x, y] | null; kind: "d" detected | "e" estimated; trail_frames
    players: List[PlayerTrack]
    default_player: Optional[int]
    events: Dict[str, Any]                                  # hits: [ShuttleHit]; landings: [Call.to_dict()]; rallies: [Rally]
    labels: Dict[str, Dict[str, str]]
    warnings: List[Dict[str, str]] = field(default_factory=list)

    def to_jsonable(self) -> Dict[str, Any]:
        """Plain dict/list/str/number/bool/None structure that json.dumps accepts."""
        return to_jsonable(self)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, numpy values and tuples to JSON types; NaN/inf become None."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "iub":
            return obj.tolist()                 # already plain ints/bools: no per-element walk
        return to_jsonable(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def _round_list(values: Sequence[float], decimals: int) -> List[float]:
    return [round(float(v), decimals) for v in values]


def court_block(H: np.ndarray, H_inv: np.ndarray, match_type: str) -> Dict[str, Any]:
    """Court geometry the page needs: homographies, painted lines, dimensions and a self-test."""
    world = np.asarray(SELFTEST_WORLD_POINTS_M, dtype=np.float64)
    img = court_model.world_to_img(world, H)
    world_back = court_model.img_to_world(img, H_inv)
    return {
        "H": np.asarray(H, dtype=np.float64).tolist(),          # world (m) -> image (px)
        "H_inv": np.asarray(H_inv, dtype=np.float64).tolist(),  # image (px) -> world (m)
        "lines_m": [[list(a), list(b)] for a, b in court_model.full_court_lines()],
        "net_x_m": court_model.NET_X,
        "length_m": court_model.COURT_LENGTH,
        "width_m": court_model.COURT_WIDTH,
        "singles_y_m": list(court_model.SIDELINE_SINGLES_Y),
        "match_type": match_type,
        "calibration_covers": "near_half",
        # full precision on purpose: the page compares its own H_inv result against these
        "selftest": [{"img_px": i.tolist(), "world_m": w.tolist()} for i, w in zip(img, world_back)],
    }


def pose_block() -> Dict[str, Any]:
    return {"connections": [list(c) for c in POSE_CONNECTIONS], "hidden": HIDDEN_LANDMARK_PX}


def shuttle_block(run: ClipRun) -> Dict[str, Any]:
    """Per-frame shuttle position: detections ("d") and the estimates between them ("e")."""
    n = run.n_frames
    decimals = config.CLIP_EXPORT_PX_DECIMALS
    px: List[Optional[List[float]]] = [None] * n
    kind: List[Optional[str]] = [None] * n
    for i, pt in enumerate(run.shuttle_px[:n]):
        if pt is not None:
            px[i], kind[i] = _round_list(pt, decimals), "d"
    for fp in run.filled:
        if 0 <= fp.frame_idx < n and px[fp.frame_idx] is None:
            px[fp.frame_idx], kind[fp.frame_idx] = _round_list(fp.pt, decimals), "e"
    return {"px": px, "kind": kind, "trail_frames": config.CLIP_TRAIL_FRAMES}


def skeleton_row(landmarks: np.ndarray) -> np.ndarray:
    """(33, 3) landmarks (x, y, score) -> int32 (66,): integer pixels clipped to >= 0, joints below
    config.CLIP_SKELETON_MIN_SCORE set to HIDDEN_LANDMARK_PX."""
    xy = np.maximum(np.rint(landmarks[:, :2]), 0).astype(np.int32)
    xy[landmarks[:, 2] < config.CLIP_SKELETON_MIN_SCORE] = HIDDEN_LANDMARK_PX
    return xy.reshape(-1)


def players_block(run: ClipRun, min_frames: int) -> List[PlayerTrack]:
    """One PlayerTrack per selected player seen in at least `min_frames` frames, most frames first."""
    n = run.n_frames
    px_d, world_d = config.CLIP_EXPORT_PX_DECIMALS, config.CLIP_EXPORT_WORLD_DECIMALS
    tracks: Dict[int, PlayerTrack] = {}
    for i, record in enumerate(run.records):
        for pid, obs in record.players.items():
            t = tracks.get(pid)
            if t is None:
                t = tracks[pid] = PlayerTrack(pid, [], 0, 0, [None] * n, [None] * n, [None] * n, [None] * n)
            if obs.track_id not in t.track_ids:
                t.track_ids.append(obs.track_id)
            t.n_frames += 1
            t.bbox_px[i] = _round_list(obs.bbox, px_d)
            t.foot_px[i] = _round_list(obs.foot_px, px_d)
            if obs.foot_world is not None:
                t.foot_world_m[i] = _round_list(obs.foot_world, world_d)
            if obs.landmarks is not None:
                t.skeleton[i] = skeleton_row(obs.landmarks)
                t.n_pose_frames += 1
    kept = [t for t in tracks.values() if t.n_frames >= min_frames]
    return sorted(kept, key=lambda t: t.n_frames, reverse=True)


def _r(value: Optional[float], decimals: int) -> Optional[float]:
    return None if value is None else round(value, decimals)


def _export_hit(hit: ShuttleHit) -> ShuttleHit:
    """A copy with the numbers rounded for the JSON (px as for the other pixel values)."""
    px_d = config.CLIP_EXPORT_PX_DECIMALS
    return replace(hit, shuttle_pt_px=tuple(_round_list(hit.shuttle_pt_px, px_d)),
                   prev_pt_px=None if hit.prev_pt_px is None else tuple(_round_list(hit.prev_pt_px, px_d)),
                   speed_px_per_frame=round(hit.speed_px_per_frame, config.CLIP_EXPORT_SPEED_DECIMALS),
                   angle_deg=round(hit.angle_deg, config.CLIP_EXPORT_ANGLE_DECIMALS),
                   wrist_dist_body=_r(hit.wrist_dist_body, config.CLIP_EXPORT_RATIO_DECIMALS),
                   confidence=round(hit.confidence, config.CLIP_EXPORT_CONFIDENCE_DECIMALS))


def _export_shot(shot: Shot) -> Shot:
    """A copy with the confidence and the measured features rounded for the JSON."""
    f = shot.features
    if f is not None:
        speed_d, angle_d = config.CLIP_EXPORT_SPEED_DECIMALS, config.CLIP_EXPORT_ANGLE_DECIMALS
        ratio_d = config.CLIP_EXPORT_RATIO_DECIMALS
        f = replace(f, wrist_above_head_body=_r(f.wrist_above_head_body, ratio_d),
                    wrist_below_hip_body=_r(f.wrist_below_hip_body, ratio_d),
                    elbow_angle_deg=_r(f.elbow_angle_deg, angle_d), elevation_deg=round(f.elevation_deg, angle_d),
                    speed_px_per_frame=round(f.speed_px_per_frame, speed_d),
                    speed_body_per_s=_r(f.speed_body_per_s, speed_d), flight_s=_r(f.flight_s, ratio_d),
                    landing_from_net_m=_r(f.landing_from_net_m, speed_d), ground_dist_m=_r(f.ground_dist_m, speed_d),
                    ground_speed_mps=_r(f.ground_speed_mps, speed_d))
    return replace(shot, confidence=round(shot.confidence, config.CLIP_EXPORT_CONFIDENCE_DECIMALS), features=f)


def _export_rally(rally: Rally) -> Rally:
    return replace(rally, winner_confidence=round(rally.winner_confidence, config.CLIP_EXPORT_CONFIDENCE_DECIMALS))


def _attach_racket_hands(players: List[PlayerTrack], hits: Sequence[ShuttleHit], forced: Optional[str]) -> None:
    voted = racket_hands(hits)
    for track in players:
        if forced is not None:
            track.racket_hand = RacketHand(forced, 1.0, voted.get(track.player_id, (None, 0.0, 0))[2], "manual")
        elif track.player_id in voted:
            hand, share, votes = voted[track.player_id]
            track.racket_hand = RacketHand(hand, round(share, config.CLIP_EXPORT_CONFIDENCE_DECIMALS), votes, "auto")


def _attach_movement(players: List[PlayerTrack], hits: Sequence[ShuttleHit], rallies: Sequence[Rally], fps: float,
                     first_frame_idx: int, last_frame_idx: int, params: MovementParams) -> None:
    spans = [(r.start_frame_idx, r.end_frame_idx) for r in rallies]
    for track in players:
        own_hits = [h.contact_frame_idx for h in hits if h.valid and h.hitter == NEAR and h.player_id == track.player_id]
        track.movement = compute_movement(track.foot_world_m, fps, own_hits, spans, first_frame_idx, last_frame_idx,
                                          params)


@dataclass
class CourtCalibration:
    """The court calibration of the video and its match type."""
    H: np.ndarray                   # world (m) -> image (px)
    H_inv: np.ndarray               # image (px) -> world (m)
    match_type: str                 # umpire.SINGLES | umpire.DOUBLES


@dataclass
class ClipAnalysisParams:
    """Thresholds of the analysis; each defaults to the configured values, read when this is created."""
    hitter: HitterParams = field(default_factory=HitterParams.from_config)
    rally: RallyParams = field(default_factory=RallyParams.from_config)
    movement: MovementParams = field(default_factory=MovementParams.from_config)
    shots: ShotThresholds = field(default_factory=ShotThresholds.from_config)
    extra_warnings: List[Dict[str, str]] = field(default_factory=list)      # added to the warnings found here


def _analyse_events(run: ClipRun, players_by_frame, match_type: str, fps: float, first_frame_idx: int,
                    params: ClipAnalysisParams) -> Tuple[list, list, list]:
    """(hits, rallies, shots) of the clip."""
    calls = run.calls
    hits = attribute_hits(run.measurements, run.shuttle_px, players_by_frame, calls, params.hitter)
    apply_alternation(hits, calls, match_type, params.hitter)
    rallies = segment_rallies(hits, calls, fps, params.rally, first_frame_idx, run.n_frames - 1)
    shots = build_shots(hits, rallies, calls, players_by_frame, fps, params.shots)
    return hits, rallies, shots


def _collect_warnings(fps: float, match_type: str, has_players: bool, hits: Sequence[ShuttleHit],
                      extra: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    warnings = [labels_vi.warning("far_half_extrapolated")]
    if abs(fps - config.CFR_FPS) > 0.01:
        warnings.append(labels_vi.warning("fps_not_60", fps=fps))
    if not has_players:
        warnings.append(labels_vi.warning("no_player"))
    n_breaks = sum(h.alternation_break for h in hits if h.valid)
    if n_breaks:
        warnings.append(labels_vi.warning("alternation_break", count=n_breaks))
    if match_type != SINGLES and any(h.valid and h.evidence != "wrist" for h in hits):
        warnings.append(labels_vi.warning("doubles_unsupported"))
    return warnings + list(extra)


def build_clip_analysis(run: ClipRun, *, video: VideoMeta, source: ClipSource, court: CourtCalibration,
                        generator: Dict[str, Any], params: Optional[ClipAnalysisParams] = None) -> ClipAnalysis:
    """
    Assemble the ClipAnalysis of a processed clip.

    run.records must be consecutive clip frames starting at 0 (what run_clip produces from the
    cut clip); video.n_frames is set from them. params.hitter.hand forces the racket hand.
    """
    for i, record in enumerate(run.records):
        if record.frame_idx != i:
            raise ValueError(f"records must be consecutive clip frames from 0 (record {i} is frame {record.frame_idx})")
    video.n_frames = run.n_frames
    params = params or ClipAnalysisParams()
    first = source.analysis_start_frame

    players = players_block(run, config.CLIP_MIN_PLAYER_FRAMES)
    # hits are attributed among the players the viewer offers, not to a passer-by seen for a moment
    offered = {p.player_id for p in players}
    players_by_frame = [{pid: obs for pid, obs in r.players.items() if pid in offered} for r in run.records]
    hits, rallies, shots = _analyse_events(run, players_by_frame, court.match_type, video.fps, first, params)

    _attach_racket_hands(players, hits, params.hitter.hand)
    for track in players:
        track.shot_summary = summarize_shots(shots, rallies, track.player_id, first, params.shots.depth_edges_m)
    _attach_movement(players, hits, rallies, video.fps, first, run.n_frames - 1, params.movement)

    return ClipAnalysis(
        schema_version=SCHEMA_VERSION,
        generator=generator,
        source=source,
        video=video,
        court=court_block(court.H, court.H_inv, court.match_type),
        pose=pose_block(),
        shuttle=shuttle_block(run),
        players=players,
        default_player=players[0].player_id if players else None,
        events={"hits": [_export_hit(h) for h in hits], "landings": [c.to_dict() for c in run.calls],
                "rallies": [_export_rally(r) for r in rallies], "shots": [_export_shot(x) for x in shots]},
        labels=labels_vi.labels_for_export(),
        warnings=_collect_warnings(video.fps, court.match_type, bool(players), hits, params.extra_warnings),
    )
