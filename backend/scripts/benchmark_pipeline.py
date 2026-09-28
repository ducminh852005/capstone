"""
Headless benchmark for the player + shuttle pipeline.

Measures per-stage latency (decode, YOLO, pose, shuttle) and a few quality
proxies. There are no hand labels yet, so the quality numbers are PROXIES
(ID stability, track continuity), not ground-truth accuracy.

Works with both the original core API and the refactored one, so the same
script produces the "before" and "after" numbers.

Usage:
    python scripts/benchmark_pipeline.py ../data/cfr/tran04_cam1.mp4 --frames 1800 --out ../data/benchmarks/after.json
"""
import argparse
import inspect
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.player_tracker import PlayerTracker
from core.shuttle_tracker import ShuttleDetector

try:
    from core import court_model
    from core.umpire import RallyUmpire
    from core.video_io import ThreadedVideoReader
    NEW_API = True
except ImportError:
    NEW_API = False

WARMUP_FRAMES = 10
CALIB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calibration.json")


class StageTimer:
    def __init__(self):
        self.total = defaultdict(float)
        self.calls = defaultdict(int)
        self.enabled = False

    def wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.total[name] += time.perf_counter() - t0
                self.calls[name] += 1
        return wrapped


def load_h():
    if NEW_API:
        return court_model.load_calibration(CALIB_PATH)
    with open(CALIB_PATH) as f:
        pts = json.load(f)["image_points"]
    world = [(0, 0), (0, 6.10), (6.70, 0), (6.70, 6.10)]
    H, _ = cv2.findHomography(np.float32(world), np.float32(pts), 0)
    return H, np.linalg.inv(H)


def fixed_roi(w, h):
    return int(w * 0.15), int(h * 0.10), int(w * 0.85), h


def run_players_old(tracker, frame, H):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = fixed_roi(w, h)
    res = tracker.track_frame(frame[y0:y1, x0:x1], persist=True)
    if res.boxes.id is None:
        return {}
    boxes = res.boxes.xyxy.cpu().numpy()
    boxes[:, [0, 2]] += x0
    boxes[:, [1, 3]] += y0
    ids = res.boxes.id.cpu().numpy()
    return tracker.filter_players_by_roi(frame, boxes, ids, (300, 1080), H)


def run_shuttle_old(detector, frame):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = fixed_roi(w, h)
    pt, _ = detector.detect(frame[y0:y1, x0:x1])
    if pt is not None:
        pt = (pt[0] + x0, pt[1] + y0)
        detector.trajectory[-1] = pt  # replicates test_auto_umpire.py behaviour
    return pt


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
    ap.add_argument("--pose-variant", default=None, help="new API only: lite|heavy")
    ap.add_argument("--bg-method", default=None, help="cv shuttle backend only: knn|mog2")
    ap.add_argument("--shuttle-backend", default="tracknet", choices=["cv", "tracknet"],
                     help="new API only: candidate generator for ShuttleDetector (default: tracknet)")
    ap.add_argument("--tracknet-stride", type=int, default=None,
                     help="tracknet backend only: frames between forward passes (default: seq_len, nonoverlap)")
    ap.add_argument("--tracknet-conf", type=float, default=None,
                     help="tracknet backend only: heatmap confidence threshold (default: 0.5)")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    H, H_inv = load_h()
    timer = StageTimer()

    tracker = detector = umpire = None
    if "player" in stages:
        kwargs = {"model_path": "yolov8n.pt", "conf_thresh": 0.5}
        if NEW_API:
            kwargs["fps"] = 60.0
        if NEW_API and args.pose_variant:
            kwargs["pose_variant"] = args.pose_variant
        tracker = PlayerTracker(**kwargs)
        tracker.track_frame = timer.wrap("yolo", tracker.track_frame)
        tracker.pose_estimator.extract_foot_point = timer.wrap("pose", tracker.pose_estimator.extract_foot_point)
    if "shuttle" in stages:
        kwargs = {}
        if NEW_API:
            kwargs["backend"] = args.shuttle_backend
            if args.bg_method and args.shuttle_backend == "cv":
                kwargs["bg_method"] = args.bg_method
            if args.shuttle_backend == "tracknet" and (args.tracknet_stride or args.tracknet_conf):
                tnk = {}
                if args.tracknet_stride:
                    tnk["batch_stride"] = args.tracknet_stride
                if args.tracknet_conf:
                    tnk["conf_threshold"] = args.tracknet_conf
                    kwargs["init_min_confidence"] = args.tracknet_conf
                kwargs["tracknet_kwargs"] = tnk
        detector = ShuttleDetector(**kwargs)
        new_shuttle = "roi" in inspect.signature(detector.detect).parameters

    if NEW_API:
        reader = ThreadedVideoReader(args.video, start_frame=args.start)
        frames = iter(reader)
    else:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

        def gen():
            i = args.start
            while True:
                ok, fr = cap.read()
                if not ok:
                    return
                yield i, fr
                i += 1
        frames = gen()

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
        if n == WARMUP_FRAMES:
            timer.enabled = True
            t_start = time.perf_counter()
            decode_t = 0.0
        decode_t += time.perf_counter() - t0

        if tracker is not None:
            if NEW_API and hasattr(tracker, "process"):
                players = tracker.process(frame, idx, H, H_inv)
            else:
                players = run_players_old(tracker, frame, H)
            player_ids_per_frame.append(sorted(int(k) for k in players.keys()))

        if detector is not None:
            t1 = time.perf_counter()
            if new_shuttle:
                pboxes = [p.bbox for p in players.values()] if tracker is not None else None
                others = tracker.non_player_boxes(players) if tracker is not None else None
                roi = court_model.shuttle_roi(H, frame.shape)
                pt = detector.detect(frame, roi=roi, exclude_boxes=others, player_boxes=pboxes)[0]
                if umpire is None and NEW_API:
                    umpire = RallyUmpire(H_inv, match_type="singles", frame_size=frame.shape[:2], roi=roi,
                                         net_top_y=court_model.net_top_threshold_y(H))
                if umpire is not None:
                    c = umpire.update(idx, pt, detector.track_active,
                                      people_boxes=list(tracker.last_boxes) if tracker is not None else None)
                    if c is not None:
                        calls.append(c.to_dict())
            else:
                pt = run_shuttle_old(detector, frame)
            if timer.enabled:
                timer.total["shuttle"] += time.perf_counter() - t1
                timer.calls["shuttle"] += 1
            shuttle_pts.append(pt)
        n += 1

    wall = time.perf_counter() - t_start if t_start else 0.0
    timed = max(n - WARMUP_FRAMES, 1)

    result = {
        "video": os.path.basename(args.video),
        "api": "new" if NEW_API else "old",
        "frames": n,
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
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
