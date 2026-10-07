"""
Headless benchmark for the player + shuttle pipeline.

Measures per-stage latency (decode, YOLO, pose, shuttle) and a few quality
proxies (ID stability, track continuity; ground-truth scoring is scripts/eval_events.py).
Timing starts after the warm-up of every stage, including the shuttle detector's background
building and first (cudnn-autotuned) forward pass. Laptop GPUs throttle: to compare two
configurations, run them back to back or use scripts/ab_bench.py, never across sessions.

Usage:
    python scripts/benchmark_pipeline.py <video> --frames 1800 --out data/benchmarks/after.json
"""
import argparse
import json
import os
import time

import numpy as np

import _common
from core import config, court_model
from core.player_tracker import PlayerTracker
from core.shuttle_tracker import ShuttleDetector
from core.umpire import RallyUmpire, match_type_from_metadata
from core.video_io import ThreadedVideoReader

MIN_WARMUP_FRAMES = 10


StageTimer = _common.StageTimer


def runs_of_detections(points):
    runs, cur = [], 0
    for p in points:
        if p is not None:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def second_diff_px(points):
    """|p[i+1] - 2 p[i] + p[i-1]| over consecutive detections: small for physical motion, large for jumps between blobs."""
    out = []
    for a, b, c in zip(points, points[1:], points[2:]):
        if a is not None and b is not None and c is not None:
            out.append(np.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stages", default="player,shuttle")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pose-variant", default=None, help="lite|heavy")
    ap.add_argument("--bg-method", default=None, help="cv shuttle backend only: knn|mog2")
    ap.add_argument("--shuttle-backend", default=config.DEFAULT_BACKEND, choices=["cv", "tracknet", "tracknet-onnx"],
                     help="candidate generator for ShuttleDetector (default: %(default)s)")
    ap.add_argument("--yolo-every", type=int, default=config.PLAYER_YOLO_EVERY,
                     help="run YOLO every N frames (default: config.PLAYER_YOLO_EVERY = %(default)s)")
    ap.add_argument("--tracknet-stride", type=int, default=None,
                     help="tracknet backend only: frames between forward passes (default: config.TRACKNET_BATCH_STRIDE)")
    ap.add_argument("--all-heatmaps", action="store_true",
                    help="tracknet backend only: use every heatmap of a pass (detection per frame, 7 frames late)")
    ap.add_argument("--tracknet-conf", type=float, default=None,
                     help="tracknet backend only: heatmap confidence threshold (default: 0.5)")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    H, H_inv = court_model.load_calibration()
    if H is None:
        raise SystemExit("No calibration found: run scripts/calibrate_court.py first.")
    match_type = match_type_from_metadata(args.video)
    timer = StageTimer()
    reader = ThreadedVideoReader(args.video, start_frame=args.start)
    frames = iter(reader)

    tracker = detector = umpire = None
    if "player" in stages:
        kwargs = {"conf_thresh": config.PLAYER_DEMO_YOLO_CONF, "fps": reader.fps,
                  "yolo_every": args.yolo_every}
        if args.pose_variant:
            kwargs["pose_variant"] = args.pose_variant
        tracker = PlayerTracker(**kwargs)
        tracker.track_frame = timer.wrap("yolo", tracker.track_frame)
        tracker.pose_estimator.extract_foot_point = timer.wrap("pose", tracker.pose_estimator.extract_foot_point)
    if "shuttle" in stages:
        kwargs = {"backend": args.shuttle_backend}
        if args.bg_method and args.shuttle_backend == "cv":
            kwargs["bg_method"] = args.bg_method
        if args.shuttle_backend == "tracknet" and (args.tracknet_stride or args.tracknet_conf or args.all_heatmaps):
            tnk = {"all_heatmaps": True} if args.all_heatmaps else {}
            if args.tracknet_stride:
                tnk["batch_stride"] = args.tracknet_stride
            if args.tracknet_conf:
                tnk["conf_threshold"] = args.tracknet_conf
                kwargs["init_min_confidence"] = args.tracknet_conf
            kwargs["tracknet_kwargs"] = tnk
        detector = ShuttleDetector(**kwargs)

    warmup = max(MIN_WARMUP_FRAMES, detector.warmup_frames if detector is not None else 0)
    roi = None
    player_ids_per_frame = []
    shuttle_pts = []
    calls = []
    decode_t = 0.0
    n = 0
    t_start = None

    while n < args.frames:
        t0 = time.perf_counter()
        try:
            idx, frame = next(frames)
        except StopIteration:
            break
        if n == warmup:
            timer.enabled = True
            t_start = time.perf_counter()
            decode_t = 0.0
        decode_t += time.perf_counter() - t0

        if tracker is not None:
            players = tracker.process(frame, idx, H, H_inv)
            player_ids_per_frame.append(sorted(int(k) for k in players.keys()))

        if detector is not None:
            t1 = time.perf_counter()
            pboxes = [p.bbox for p in players.values()] if tracker is not None else None
            others = tracker.non_player_boxes(players) if tracker is not None else None
            if roi is None:
                roi = court_model.shuttle_roi(H, frame.shape)
            pt = detector.detect(frame, roi=roi, exclude_boxes=others, player_boxes=pboxes)[0]
            if umpire is None:
                umpire = RallyUmpire(H_inv, match_type=match_type, frame_size=frame.shape[:2], roi=roi,
                                     net_top_y=court_model.net_top_threshold_y(H))
            c = umpire.update(idx - detector.frame_lag, pt, detector.track_active,
                              people_boxes=list(tracker.last_boxes) if tracker is not None else None,
                              flight_id=detector.flight_id)
            if c is not None:
                calls.append(c.to_dict())
            if timer.enabled:
                timer.total["shuttle"] += time.perf_counter() - t1
                timer.calls["shuttle"] += 1
            shuttle_pts.append(pt)
        n += 1

    wall = time.perf_counter() - t_start if t_start else 0.0
    timed = max(n - warmup, 1)

    result = {
        "video": os.path.basename(args.video),
        "git": _common.git_revision(),
        "config": {
            "shuttle_backend": args.shuttle_backend if detector is not None else None,
            "tracknet_strides_active_idle": detector.strides if detector is not None else None,
            "all_heatmaps": args.all_heatmaps if detector is not None else None,
            "yolo_every": args.yolo_every if tracker is not None else None,
            "yolo_half": tracker.half if tracker is not None else None,
        },
        "frames": n,
        "warmup_frames": warmup,
        "wall_fps": round(timed / wall, 2) if wall else None,
        "ms_per_frame": {
            "decode_wait": round(1000 * decode_t / timed, 2),
            **{k: round(1000 * v / timed, 2) for k, v in timer.total.items()},
        },
        "calls": dict(timer.calls),
    }

    if player_ids_per_frame:
        all_ids = [i for ids in player_ids_per_frame for i in ids]
        counts = [len(ids) for ids in player_ids_per_frame]
        result["player"] = {
            "pct_frames_with_player": round(100 * np.mean([c > 0 for c in counts]), 1),
            "pct_frames_multi_player": round(100 * np.mean([c > 1 for c in counts]), 1),
            "distinct_player_ids": len(set(all_ids)),
            "pose_calls": timer.calls.get("pose", 0),
        }
    if shuttle_pts:
        runs = runs_of_detections(shuttle_pts)
        acc = second_diff_px(shuttle_pts)
        result["shuttle"] = {
            "accel_px_median": round(float(np.median(acc)), 2) if len(acc) else None,
            "accel_px_p90": round(float(np.percentile(acc, 90)), 2) if len(acc) else None,
            "pct_steps_jump_gt_30px": round(100 * float(np.mean(acc > 30)), 1) if len(acc) else None,
            "pct_frames_detected": round(100 * np.mean([p is not None for p in shuttle_pts]), 1),
            "n_runs": len(runs),
            "mean_run_len": round(float(np.mean(runs)), 2) if runs else 0,
            "n_runs_ge_10": sum(r >= 10 for r in runs),
            "umpire_calls": len(calls),
        }
        if calls:
            result["shuttle"]["calls"] = calls

    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    _common.setup_logging()
    main()
