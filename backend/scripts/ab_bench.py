"""
A/B benchmark: two pipeline configurations run on the SAME frames, interleaved frame by frame.

A laptop GPU throttles and shares its clock with everything else on the machine, so timings
of two separate runs can differ by 10-50% for no reason. Here both variants process every
frame back to back and each is timed separately, so slow drift hits both equally.

A variant is a comma separated list of key=value settings (all optional):
    half=0|1         PlayerTracker YOLO half precision (default: config.PLAYER_YOLO_HALF)
    yolo_every=N     PlayerTracker YOLO cadence (default: config.PLAYER_YOLO_EVERY)
    stride=N         TrackNet batch_stride while a track is active (default: config.TRACKNET_BATCH_STRIDE)
    all=0|1          use every heatmap of a TrackNet pass (detection per frame, 7 frames late)
    idle=N           TrackNet stride while no track is active (0 = same as stride; default: config.TRACKNET_IDLE_STRIDE)
    conf=F           TrackNet heatmap threshold
It also compares the RESULTS: an optimisation that must not change behaviour should print
"identical umpire calls: True".

Usage:
    python scripts/ab_bench.py ../data/cfr/tran04_cam1.mp4 --frames 600 --start 300 \
        --a "half=1" --b "half=0"
"""
import argparse
import time
from collections import defaultdict

import _common
from core import config, court_model
from core.player_tracker import PlayerTracker
from core.umpire import RallyUmpire, match_type_from_metadata
from core.video_io import ThreadedVideoReader


def parse_variant(text):
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, _, value = item.partition("=")
        if key not in ("half", "yolo_every", "stride", "idle", "conf", "all"):
            raise SystemExit(f"Unknown variant key {key!r} in {text!r}")
        out[key] = float(value) if key == "conf" else int(value)
    return out


class Variant:
    """One complete pipeline (players, shuttle, umpire) with its own state and timers."""

    def __init__(self, name, settings, args, fps, H, H_inv, match_type, frame_size):
        self.name, self.settings = name, settings
        self.H, self.H_inv = H, H_inv
        self.tracker = self.detector = self.umpire = None
        if "player" in args.stages:
            kwargs = {"conf_thresh": config.PLAYER_DEMO_YOLO_CONF, "fps": fps,
                      "yolo_every": settings.get("yolo_every", config.PLAYER_YOLO_EVERY)}
            if "half" in settings:
                kwargs["half"] = bool(settings["half"])
            self.tracker = PlayerTracker(**kwargs)
        if "shuttle" in args.stages:
            tnk = {}
            if "stride" in settings:
                tnk["batch_stride"] = settings["stride"]
            if "idle" in settings:
                tnk["idle_stride"] = settings["idle"]
            if "all" in settings:
                tnk["all_heatmaps"] = bool(settings["all"])
            if "conf" in settings:
                tnk["conf_threshold"] = settings["conf"]
            self.detector = _common.create_detector(args.backend, tracknet_kwargs=tnk or None)
            self.roi = court_model.shuttle_roi(H, frame_size)
            self.umpire = RallyUmpire(H_inv, match_type=match_type, frame_size=frame_size, roi=self.roi,
                                      net_top_y=court_model.net_top_threshold_y(H))
        self.time = defaultdict(float)
        self.calls, self.detections = [], 0

    @property
    def warmup_frames(self):
        return self.detector.warmup_frames if self.detector is not None else 0

    def step(self, idx, frame, timed):
        players = {}
        if self.tracker is not None:
            t = time.perf_counter()
            players = self.tracker.process(frame, idx, self.H, self.H_inv)
            if timed:
                self.time["players"] += time.perf_counter() - t
        if self.detector is not None:
            t = time.perf_counter()
            pt, _ = self.detector.detect(
                frame, roi=self.roi,
                exclude_boxes=self.tracker.non_player_boxes(players) if self.tracker else None,
                player_boxes=[p.bbox for p in players.values()] if self.tracker else None)
            call = self.umpire.update(idx - self.detector.frame_lag, pt, self.detector.track_active,
                                      people_boxes=list(self.tracker.last_boxes) if self.tracker else None,
                                      flight_id=self.detector.flight_id)
            if timed:
                self.time["shuttle"] += time.perf_counter() - t
            self.detections += pt is not None
            if call is not None:
                self.calls.append(call.to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stages", default="player,shuttle")
    ap.add_argument("--backend", default=config.DEFAULT_BACKEND)
    ap.add_argument("--a", default="", help="settings of variant A (see the module docstring)")
    ap.add_argument("--b", default="", help="settings of variant B")
    args = ap.parse_args()

    H, H_inv = court_model.load_calibration()
    if H is None:
        raise SystemExit("No calibration found: run scripts/calibrate_court.py first.")
    reader = ThreadedVideoReader(args.video, start_frame=args.start)
    match_type = match_type_from_metadata(args.video)
    frame_size = (reader.size[1], reader.size[0])
    variants = [Variant(n, parse_variant(s), args, reader.fps, H, H_inv, match_type, frame_size)
                for n, s in (("A", args.a), ("B", args.b))]
    warmup = max(10, *(v.warmup_frames for v in variants))

    n = 0
    for idx, frame in reader:
        if n >= args.frames + warmup:
            break
        # alternate which variant goes first, so neither always pays for the other's leftover GPU work
        for v in (variants if n % 2 == 0 else variants[::-1]):
            v.step(idx, frame, timed=n >= warmup)
        n += 1
    reader.release()

    timed = max(n - warmup, 1)
    print(f"\n{timed} timed frames after {warmup} warm-up frames, {args.video}")
    for v in variants:
        parts = "  ".join(f"{k} {1000 * t / timed:6.2f} ms" for k, t in v.time.items())
        total = 1000 * sum(v.time.values()) / timed
        print(f"  {v.name} {str(v.settings):38s} {parts}  | total {total:6.2f} ms/frame = {1000 / total:5.1f} fps")
    a, b = variants
    ta, tb = sum(a.time.values()), sum(b.time.values())
    if ta and tb:
        print(f"  B is {ta / tb:.2f}x the speed of A ({100 * (1 - tb / ta):+.0f}% time)")
    print(f"  detections: A {a.detections}, B {b.detections}")
    print(f"  umpire calls: A {len(a.calls)}, B {len(b.calls)}; identical umpire calls: {a.calls == b.calls}")


if __name__ == "__main__":
    _common.setup_logging()
    main()
