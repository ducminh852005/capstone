"""
Score the shot types and hitters found by analyze_clip.py against hand labels.

    python scripts/eval_shots.py ../data/clips/tran04_cam1_f250/analysis.json [more analysis.json ...]
        [--labels ../data/events/tran04_cam1.events.json] [--tol 6] [--out NAME]

Labels come from the clip viewer ("Nhãn" tab) merged with scripts/merge_labels.py. Predictions of
several clips of the same video are pooled; only the frames inside the labels' reviewed `ranges` count.
Prints hit precision / recall, the hitter agreement, the shot-type accuracy, the confusion matrix and
every wrong type with the reasons the rules gave, which is where to look when tuning config SHOT_*.
"""
import argparse
import json
import sys
from pathlib import Path

from _common import setup_logging
from core import config
from core.event_eval import evaluate_shots, load_labels

REPORT_PREFIX = "shots_"


def predictions(analysis_paths):
    """Predicted shots of the analysis files as {"frame" (absolute), "type", "hitter", "reasons"} and the video name."""
    preds, video = [], None
    for path in analysis_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        name = data["source"]["video"]
        if video not in (None, name):
            raise ValueError(f"{path} is for {name}, the others for {video}")
        video = name
        offset = data["source"]["start_frame"]
        first = data["source"]["analysis_start_frame"]
        preds += [{"frame": offset + s["frame_idx"], "type": s["type"], "hitter": s["hitter"], "reasons": s["reasons"]}
                  for s in data["events"]["shots"] if s["frame_idx"] >= first]
    return preds, video


def pct(value):
    return "n/a" if value is None else f"{100 * value:.0f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", nargs="+", help="analysis.json written by analyze_clip.py")
    ap.add_argument("--labels", help="label file (default data/events/<video stem>.events.json)")
    ap.add_argument("--tol", type=int, default=config.SHOT_EVAL_TOL_FRAMES, help="frames within which a hit matches a label")
    ap.add_argument("--out", help=f"write the report to data/benchmarks/{REPORT_PREFIX}<OUT>.json")
    args = ap.parse_args()

    preds, video = predictions(args.analysis)
    path = Path(args.labels) if args.labels else config.DATA_DIR / "events" / f"{Path(video).stem}.events.json"
    labels = load_labels(path)
    if labels["video"] not in (None, video):
        raise ValueError(f"{path} is for {labels['video']}, the analysis for {video}")
    if not labels["shots"]:
        raise ValueError(f"{path} has no labelled shots: review some in the viewer and run merge_labels.py")
    if not labels["ranges"]:
        print("WARNING: the labels carry no reviewed range: only recall can be scored, precision is not meaningful")

    result = evaluate_shots(preds, labels["shots"], args.tol, labels["ranges"])
    hits = result["hits"]
    print(f"Hits: precision {pct(hits['precision'])}, recall {pct(hits['recall'])} "
          f"({len(hits['tp'])} matched, {len(hits['fp'])} extra, {len(hits['fn'])} missed; tol +-{args.tol} frames)")
    print(f"Hitter agreement {pct(result['hitter_agreement'])}; type accuracy {pct(result['type_accuracy'])} "
          f"over {result['n_typed']} labelled near-player hits")
    for kind, row in sorted(result["confusion"].items()):
        print(f"  labelled {kind:8s} -> " + ", ".join(f"{p} x{n}" for p, n in sorted(row.items())))
    for kind, c in sorted(result["by_true_type"].items()):
        print(f"  {kind:8s} labelled {c['n']}, missed {c['missed']}, type right {c['correct']}")

    pred_at = {p["frame"]: p for p in preds}
    label_at = {x["frame"]: x for x in labels["shots"]}
    for p_frame, t_frame in hits["tp"]:
        t = label_at[t_frame]
        p = pred_at[p_frame]
        if t["hitter"] == "near" and t["type"] not in ("unknown", p["type"]):
            print(f"  WRONG frame {t_frame}: labelled {t['type']}, predicted {p['type']} ({', '.join(p['reasons'])})")

    if args.out:
        report = {"analysis": list(args.analysis), "labels": str(path), "tol": args.tol,
                  "ranges": [list(r) for r in labels["ranges"]], "result": result}
        config.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        out = config.BENCHMARK_DIR / f"{REPORT_PREFIX}{args.out}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError) as e:
        sys.exit(f"ERROR: {e}")
