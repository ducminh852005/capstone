"""
Score hit / smash / landing detection against hand-labelled events.

Runs the shuttle pipeline (detector + smash detector + umpire; add --players for the
person boxes the umpire and detector use in the demos) up to a little past the last
labelled frame, then matches predictions to labels within +-tol frames.

Labels: data/events/<video stem>.events.json, see core/event_eval.py for the format
(data/events/tran04_cam1.events.json is a template to fill in).

Usage:
    python scripts/eval_events.py ../data/cfr/tran04_cam1.mp4 [--tol 12] [--players] [--out NAME]

Predictions compared:
    hits      frames where the physics filter restarted the track (start_source == "physics")
    smashes   frames of SmashDetector smashes
    landings  frames of umpire Calls, reported per Call.method; the IN/OUT agreement is shown
              for labelled landings that carry a "call"
"""
import argparse
import json
import os
from pathlib import Path

import _common
from core import config, court_model
from core.event_eval import EVENT_KINDS, call_agreement, load_events, match_events
from core.smash import SmashDetector
from core.umpire import RallyUmpire, match_type_from_metadata
from core.video_io import ThreadedVideoReader

TAIL_FRAMES = 60    # keep running this long after the last label so the last flight can end


def run_pipeline(video, end_frame, backend, use_players, all_heatmaps=False):
    """Returns {"hits": [frame], "smashes": [frame], "landings": [Call dict], "starts": [...]}."""
    H, H_inv = court_model.load_calibration()
    if H is None:
        raise SystemExit("No calibration found: run scripts/calibrate_court.py first.")
    reader = ThreadedVideoReader(video)
    w, h = reader.size
    roi = court_model.shuttle_roi(H, (h, w))
    detector = _common.create_detector(backend, tracknet_kwargs={"all_heatmaps": True} if all_heatmaps else None)
    umpire = RallyUmpire(H_inv, match_type=match_type_from_metadata(video), frame_size=(h, w), roi=roi,
                         net_top_y=court_model.net_top_threshold_y(H))
    smash = SmashDetector()
    tracker = None
    if use_players:
        from core.player_tracker import PlayerTracker
        tracker = PlayerTracker(conf_thresh=config.PLAYER_DEMO_YOLO_CONF, fps=reader.fps)

    out = {"hits": [], "smashes": [], "landings": [], "starts": []}
    for frame_idx, frame in reader:
        if frame_idx > end_frame:
            break
        players = tracker.process(frame, frame_idx, H, H_inv) if tracker else {}
        pt, _ = detector.detect(
            frame, roi=roi,
            exclude_boxes=tracker.non_player_boxes(players) if tracker else None,
            player_boxes=[p.bbox for p in players.values()] if tracker else None)
        shuttle_frame = frame_idx - detector.frame_lag      # the point belongs to an earlier frame in all-heatmaps mode
        if detector.track_active and detector.track_len == 0:
            out["starts"].append({"frame": shuttle_frame, "source": detector.start_source})
            if detector.start_source == "physics":
                out["hits"].append(shuttle_frame)
        measured = smash.update(shuttle_frame, pt, detector.track_active, detector.track_len,
                                detector.kf.x[:2], detector.start_source)
        if measured is not None and measured.is_smash:
            out["smashes"].append(measured.frame_idx)
        call = umpire.update(shuttle_frame, pt, detector.track_active,
                             people_boxes=list(tracker.last_boxes) if tracker else None,
                             flight_id=detector.flight_id)
        if call is not None:
            out["landings"].append(call.to_dict())
    reader.release()
    return out


def fmt(x):
    return "  n/a" if x is None else f"{x:5.2f}"


def report(name, r):
    print(f"{name:34s} P={fmt(r['precision'])} R={fmt(r['recall'])} F1={fmt(r['f1'])}"
          f"  tp={len(r['tp'])} fp={len(r['fp'])} fn={len(r['fn'])}")
    if r["fp"]:
        print(f"    false positives (frames): {r['fp']}")
    if r["fn"]:
        print(f"    missed (frames):          {r['fn']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--events", default=None, help="label file (default data/events/<stem>.events.json)")
    ap.add_argument("--tol", type=int, default=12, help="match tolerance in frames (default 12)")
    ap.add_argument("--backend", default=config.DEFAULT_BACKEND)
    ap.add_argument("--players", action="store_true", help="also run player tracking (slower)")
    ap.add_argument("--all-heatmaps", action="store_true",
                    help="use every heatmap of a TrackNet pass (detections for every frame, 7 frames late)")
    ap.add_argument("--out", default=None, help="save the report as data/benchmarks/events_<OUT>.json")
    args = ap.parse_args()

    events_path = Path(args.events) if args.events else config.DATA_DIR / "events" / f"{Path(args.video).stem}.events.json"
    if not events_path.exists():
        raise SystemExit(f"Label file not found: {events_path}")
    labels = load_events(events_path)
    all_frames = [e["frame"] for kind in EVENT_KINDS for e in labels[kind]]
    if not all_frames:
        raise SystemExit(f"{events_path} has no labelled events yet; fill in hits/smashes/landings first.")

    pred = run_pipeline(args.video, max(all_frames) + TAIL_FRAMES + (7 if args.all_heatmaps else 0),
                        args.backend, args.players, args.all_heatmaps)

    print(f"\nvideo={os.path.basename(args.video)} backend={args.backend} tol=+-{args.tol} frames\n")
    results = {}
    for kind in ("hits", "smashes"):
        results[kind] = match_events(pred[kind], [e["frame"] for e in labels[kind]], args.tol)
        report(kind, results[kind])

    true_landings = [e["frame"] for e in labels["landings"]]
    calls = pred["landings"]
    results["landings"] = match_events([c["frame_idx"] for c in calls], true_landings, args.tol)
    report("landings (all methods)", results["landings"])
    for method in sorted({c["method"] for c in calls}):
        sub = [c for c in calls if c["method"] == method]
        r = match_events([c["frame_idx"] for c in sub], true_landings, args.tol)
        print(f"  method={method:20s} calls={len(sub):3d} matched={len(r['tp']):3d} "
              f"precision={fmt(r['precision'])}")
    agreement = call_agreement(calls, labels["landings"], args.tol)
    print(f"  IN/OUT agreement on matched labelled landings: {fmt(agreement)}")

    if args.out:
        config.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        path = config.BENCHMARK_DIR / f"events_{args.out}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"video": os.path.basename(args.video), "backend": args.backend, "tol": args.tol,
                       "all_heatmaps": args.all_heatmaps,
                       "results": results, "call_agreement": agreement, "predictions": pred}, f, indent=2)
        print(f"\nSaved -> {path}")


if __name__ == "__main__":
    _common.setup_logging()
    main()
