"""
Per-frame pipeline of the clip analysis: players (YOLO + ByteTrack + pose), shuttle (TrackNet +
Kalman), hit measurements and landing calls, merged into one FrameRecord per video frame.

This is the same loop the demo scripts each re-implement (scripts/demo_auto_umpire.py,
generate_highlights.py, eval_events.py, benchmark_pipeline.py); new code builds on this one. The
components are duck-typed so tests can feed fakes and a list of arrays instead of a video.

Frame indices are clip-local: the first frame of the video that is passed in is frame 0.
"""
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from . import config, gap_fill
from .smash import HitMeasurement
from .umpire import Call

if TYPE_CHECKING:       # only an annotation: importing it would load YOLO and MediaPipe for nothing
    from .player_tracker import PlayerObs

logger = logging.getLogger(__name__)


@dataclass
class ClipPipelineParams:
    """How the clip is processed. Defaults come from config.CLIP_*."""
    preroll_s: float = config.CLIP_PREROLL_S
    max_players: int = config.CLIP_MAX_PLAYERS
    yolo_every: int = config.CLIP_YOLO_EVERY
    pose_every: int = config.CLIP_POSE_EVERY
    pose_variant: str = config.CLIP_POSE_VARIANT
    preroll_pose_every: int = config.CLIP_PREROLL_POSE_EVERY


@dataclass
class PoseSchedule:
    """Pose cadence over the clip: sparser while the pre-roll warms the trackers up, whose skeletons
    the statistics do not use, then `every` frames on the selected players."""
    preroll_frames: int
    preroll_every: int
    every: int

    def every_at(self, frame_idx: int) -> int:
        return self.preroll_every if frame_idx < self.preroll_frames else self.every


@dataclass
class FrameRecord:
    """Everything the pipeline knows after processing one video frame."""
    frame_idx: int                                      # clip-local video frame
    players: Dict[int, "PlayerObs"]                     # selected near-half players, by player_id
    shuttle_pt_px: Optional[Tuple[int, int]]            # detection of frame `shuttle_frame_idx`
    shuttle_frame_idx: int                              # frame the shuttle point belongs to (frame_idx - frame_lag)
    track_active: bool
    track_len: int                                      # 0 on the frame a shuttle track starts
    flight_id: int
    start_source: Optional[str]                         # "physics" | "stop" | "new" while track_len == 0
    measurement: Optional[HitMeasurement] = None        # speed/direction after a track start (see smash.py)
    call: Optional[Call] = None                         # landing call; Call.frame_idx is the CONTACT frame


@dataclass
class ClipRun:
    """Result of run_clip()."""
    records: List[FrameRecord] = field(default_factory=list)
    shuttle_px: List[Optional[Tuple[int, int]]] = field(default_factory=list)   # per frame, ShuttleDetector.trajectory
    filled: List[gap_fill.FilledPoint] = field(default_factory=list)            # estimated points between detections

    @property
    def n_frames(self) -> int:
        return len(self.records)

    @property
    def measurements(self) -> List[HitMeasurement]:
        """Track-start measurements ordered by the frame the track started on."""
        return sorted((r.measurement for r in self.records if r.measurement is not None),
                      key=lambda m: m.frame_idx)

    @property
    def calls(self) -> List[Call]:
        """Landing calls ordered by contact frame. A Call is emitted several frames after the
        contact (look-ahead), so the order of arrival is not the order of Call.frame_idx."""
        return sorted((r.call for r in self.records if r.call is not None), key=lambda c: c.frame_idx)


def iter_frame_records(frames: Iterable[Tuple[int, np.ndarray]], tracker, detector, smash, umpire,
                       roi: Optional[tuple], H=None, H_inv=None,
                       pose_schedule: Optional[PoseSchedule] = None) -> Iterator[FrameRecord]:
    """
    Run the pipeline over `frames` ((frame_idx, BGR image) pairs, e.g. a ThreadedVideoReader) and
    yield one FrameRecord per frame.

    roi: shuttle detection ROI (x0, y0, x1, y1) px.  H / H_inv: world <-> image homographies.
    pose_schedule: changes the tracker's pose cadence (tracker.set_pose_every) at the pre-roll boundary.
    Raises NotImplementedError when the detector reports a frame lag (TRACKNET_ALL_HEATMAPS): the
    delayed shuttle stream is not validated for clip analysis.
    """
    current_every = None
    for frame_idx, frame in frames:
        if pose_schedule is not None and pose_schedule.every_at(frame_idx) != current_every:
            current_every = pose_schedule.every_at(frame_idx)
            tracker.set_pose_every(current_every)
        players = tracker.process(frame, frame_idx, H, H_inv)
        pt, _ = detector.detect(frame, roi=roi,
                                exclude_boxes=tracker.non_player_boxes(players),
                                player_boxes=[p.bbox for p in players.values()])
        lag = detector.frame_lag
        if lag:
            raise NotImplementedError(
                f"Clip analysis does not support a lagged shuttle detector (frame_lag={lag}); "
                "set config.TRACKNET_ALL_HEATMAPS = False")
        shuttle_frame_idx = frame_idx - lag

        # The speed after a hit is measured on the first two detections of the new track: the
        # Kalman velocity is still zero right after the restart.
        measurement = smash.update(shuttle_frame_idx, pt, detector.track_active, detector.track_len,
                                   detector.kf.x[:2], detector.start_source)
        call = umpire.update(shuttle_frame_idx, pt, detector.track_active,
                             people_boxes=list(tracker.last_boxes), flight_id=detector.flight_id)
        yield FrameRecord(
            frame_idx=frame_idx,
            players=players,
            shuttle_pt_px=pt,
            shuttle_frame_idx=shuttle_frame_idx,
            track_active=detector.track_active,
            track_len=detector.track_len,
            flight_id=detector.flight_id,
            start_source=detector.start_source,
            measurement=measurement,
            call=call,
        )


def run_clip(frames: Iterable[Tuple[int, np.ndarray]], tracker, detector, smash, umpire,
             roi: Optional[tuple], H=None, H_inv=None, log_every_frames: int = 0,
             pose_schedule: Optional[PoseSchedule] = None) -> ClipRun:
    """
    Process every frame (see iter_frame_records) and collect the per-frame shuttle trajectory and
    the gap-filled trail. The trail is only meaningful because the detector saw every frame of the
    clip from frame 0 (its trajectory index is then the clip frame).
    """
    run = ClipRun()
    for record in iter_frame_records(frames, tracker, detector, smash, umpire, roi, H, H_inv, pose_schedule):
        run.records.append(record)
        if log_every_frames and record.frame_idx % log_every_frames == 0:
            logger.info("Processed frame %d", record.frame_idx)
    run.shuttle_px = list(detector.trajectory)
    run.filled = detector.fill_gaps(allowed=gap_fill.court_constraint(H_inv, roi))
    return run
