"""
Analyse a short clip of a match and write a telestrator-style viewer for it.

Cuts `--frames` frames from `--start` (plus a pre-roll the trackers need to warm up) out of the
video, runs players (YOLO + ByteTrack + pose) and the shuttle (TrackNet + Kalman + umpire) over it,
and writes data/clips/<name>/{clip.mp4, analysis.json, index.html}. Open index.html in Chrome or
Edge: pick the player, step through the frames and draw arrows on the video.

    python scripts/analyze_clip.py ../data/cfr/tran04_cam1.mp4 --start 250 --frames 300

Pose dominates the run time; --profile prints ms per call of every stage, --pose-variant lite and
--yolo-every 2 trade accuracy for speed (see backend/README.md).
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import StageTimer, create_detector, git_revision, setup_logging
from core import clip_analysis, clip_report, config, court_model, labels_vi
from core.clip_pipeline import ClipPipelineParams, PoseSchedule, run_clip
from core.hit_events import HitterParams
from core.player_tracker import PlayerTracker
from core.pose_estimator import MODEL_FILES
from core.pose_features import LEFT, RIGHT
from core.smash import SmashDetector
from core.umpire import RallyUmpire, match_type_from_metadata
from core.video_io import ThreadedVideoReader, probe_video
from core.video_processor import cut_clip

CLIP_FILE = "clip.mp4"
LOG_EVERY_FRAMES = 60
BACKENDS = ("cv", "tracknet", "tracknet-onnx")


def parse_args():
    ap = argparse.ArgumentParser(description="Analyse a short clip and write an HTML viewer for it.")
    ap.add_argument("video", nargs="?", default=config.DEFAULT_VIDEO, help="source video (60 fps CFR, see process_all_videos.py)")
    ap.add_argument("--start", type=int, default=0, help="first frame of the clip in the source video")
    ap.add_argument("--frames", type=int, default=config.CLIP_DEFAULT_FRAMES, help="clip length in frames")
    ap.add_argument("--name", help="output folder name under data/clips/ (default <video>_f<start>)")
    ap.add_argument("--backend", default=config.DEFAULT_BACKEND, choices=BACKENDS)
    ap.add_argument("--max-players", type=int, default=config.CLIP_MAX_PLAYERS,
                    help="near-half players tracked and offered in the viewer")
    ap.add_argument("--pose-variant", default=config.CLIP_POSE_VARIANT, choices=sorted(MODEL_FILES),
                    help="MediaPipe model: heavy keeps the racket arm better, lite is about 2x faster")
    ap.add_argument("--yolo-every", type=int, default=config.CLIP_YOLO_EVERY,
                    help="run person detection every N frames (boxes of the frames in between are frozen)")
    ap.add_argument("--preroll-pose-every", type=int, default=config.CLIP_PREROLL_POSE_EVERY,
                    help="pose cadence (frames) during the pre-roll, whose skeletons the statistics do not use")
    ap.add_argument("--hand", choices=[LEFT, RIGHT],
                    help="racket hand of the near player (default: inferred from the hits)")
    ap.add_argument("--profile", action="store_true", help="print ms per call of pose, YOLO, shuttle and umpire")
    ap.add_argument("--no-cut", action="store_true", help="reuse an existing clip.mp4 in the output folder")
    return ap.parse_args()


def validate_args(args):
    """Reject what cannot work before any video or model is opened."""
    if args.start < 0 or args.frames <= 0:
        raise ValueError("--start must be >= 0 and --frames > 0")
    if args.yolo_every < 1 or args.preroll_pose_every < 1:
        raise ValueError("--yolo-every and --preroll-pose-every must be >= 1")
    if config.TRACKNET_ALL_HEATMAPS:
        raise ValueError("Clip analysis does not support the lagged shuttle stream of "
                         "config.TRACKNET_ALL_HEATMAPS; set it to False")


def needed_preroll_frames(detector, fps):
    """Frames before the shuttle detector and the player selector both produce output."""
    selector_frames = round(config.SELECTOR_MIN_TRACK_S * config.SELECTOR_MIN_SCORE * fps)
    return max(detector.warmup_frames, selector_frames)


def resolve_range(args, source, params):
    """(n_frames, preroll_frames) of the clip to cut: the request clipped to the video, plus the pre-roll."""
    if source.frame_count and args.start >= source.frame_count:
        raise ValueError(f"--start {args.start} is past the end of {args.video} ({source.frame_count} frames)")
    n_frames = min(args.frames, source.frame_count - args.start) if source.frame_count else args.frames
    return n_frames, min(round(params.preroll_s * source.fps), args.start)


def ensure_clip(args, source, preroll, n_frames, clip_path):
    """Cut the clip (or check the existing one with --no-cut); returns warnings about its length."""
    if args.no_cut:
        if not clip_path.exists():
            raise FileNotFoundError(f"--no-cut but {clip_path} does not exist")
    else:
        cut_clip(args.video, str(clip_path), args.start - preroll, preroll + n_frames, source.fps)
    clip = probe_video(clip_path)
    if clip.frame_count and clip.frame_count != preroll + n_frames:
        return clip, [labels_vi.warning("frame_count_mismatch", actual=clip.frame_count, expected=preroll + n_frames)]
    return clip, []


def build_components(args, params, clip, H, H_inv):
    """(tracker, detector, umpire, roi) for the clip."""
    w, h = clip.size
    roi = court_model.shuttle_roi(H, (h, w))
    detector = create_detector(args.backend)
    tracker = PlayerTracker(conf_thresh=config.PLAYER_DEMO_YOLO_CONF, fps=clip.fps,
                            pose_variant=params.pose_variant, pose_every=params.pose_every,
                            yolo_every=params.yolo_every, selector_kwargs={"max_players": params.max_players})
    umpire = RallyUmpire(H_inv, match_type=match_type_from_metadata(args.video), frame_size=(h, w), roi=roi,
                         net_top_y=court_model.net_top_threshold_y(H))
    return tracker, detector, umpire, roi


def instrument(timer, tracker, detector, umpire):
    """Time the stages that cost something: pose, person detection, shuttle detection, the umpire."""
    timer.enabled = True
    tracker.pose_estimator.extract_foot_point = timer.wrap("pose", tracker.pose_estimator.extract_foot_point)
    tracker.track_frame = timer.wrap("yolo", tracker.track_frame)
    detector.detect = timer.wrap("shuttle", detector.detect)
    umpire.update = timer.wrap("umpire", umpire.update)


def analyze(args):
    validate_args(args)
    H, H_inv = court_model.load_calibration()
    if H is None:
        raise FileNotFoundError(f"No court calibration at {config.CALIBRATION_PATH}; "
                                "run scripts/calibrate_court.py first")

    params = ClipPipelineParams(max_players=args.max_players, yolo_every=args.yolo_every,
                                pose_variant=args.pose_variant,
                                preroll_pose_every=args.preroll_pose_every)
    source = probe_video(args.video)
    n_frames, preroll = resolve_range(args, source, params)
    name = args.name or f"{Path(args.video).stem}_f{args.start}"
    out_dir = config.CLIPS_DIR / name
    clip, warnings = ensure_clip(args, source, preroll, n_frames, out_dir / CLIP_FILE)

    tracker, detector, umpire, roi = build_components(args, params, clip, H, H_inv)
    needed = needed_preroll_frames(detector, clip.fps)
    if preroll < needed:
        warnings.append(labels_vi.warning("preroll_short", preroll_frames=preroll, needed_frames=needed))
    timer = StageTimer()
    if args.profile:
        instrument(timer, tracker, detector, umpire)

    reader = ThreadedVideoReader(str(out_dir / CLIP_FILE))
    started = time.time()
    try:
        run = run_clip(reader, tracker, detector, SmashDetector(), umpire, roi, H, H_inv,
                       log_every_frames=LOG_EVERY_FRAMES,
                       pose_schedule=PoseSchedule(preroll, params.preroll_pose_every, params.pose_every))
    finally:
        reader.release()
    print(f"Processed {run.n_frames} frames in {time.time() - started:.1f} s")
    if args.profile:
        print("\n".join(timer.report(run.n_frames)))

    w, h = clip.size
    analysis = clip_analysis.build_clip_analysis(
        run,
        video=clip_analysis.VideoMeta(CLIP_FILE, clip.fps, w, h, run.n_frames),
        source=clip_analysis.ClipSource(Path(args.video).name, args.start - preroll, preroll, preroll),
        court=clip_analysis.CourtCalibration(H, H_inv, umpire.match_type),
        generator={"git": git_revision(), "backend": args.backend,
                   "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        params=clip_analysis.ClipAnalysisParams(hitter=HitterParams.from_config(hand=args.hand),
                                                extra_warnings=warnings))
    json_path, html_path = clip_report.write_clip_files(out_dir, analysis.to_jsonable(), CLIP_FILE,
                                                        title=f"{name} - CầuLôngStats")
    print(f"Wrote {json_path}\nOpen {html_path.as_uri()} in Chrome or Edge")
    for warning in analysis.warnings:
        print(f"WARNING: {warning['message']}")
    return html_path


if __name__ == "__main__":
    setup_logging()
    try:
        analyze(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
        sys.exit(f"ERROR: {e}")
