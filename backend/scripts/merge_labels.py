"""
Merge the labels exported by the clip viewer ("Nhãn" tab) into the events file of the video.

    python scripts/merge_labels.py ../data/clips/tran04_cam1_f250/nhan-tran04_cam1_f250.json

The viewer's file holds absolute frames of the source video, so it merges into
data/events/<video stem>.events.json (created when missing) or the file given with --into: per kind,
entries on the same frame are replaced by the new ones, the others kept, reviewed ranges are united.
The result is what scripts/eval_events.py and scripts/eval_shots.py read.
"""
import argparse
import json
import sys
from pathlib import Path

from _common import setup_logging
from core import config
from core.event_eval import LABEL_KINDS, load_labels, merge_labels

JSON_INDENT = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="label file exported by the viewer")
    ap.add_argument("--into", help="events file to merge into (default data/events/<video stem>.events.json)")
    args = ap.parse_args()

    new = load_labels(args.labels)
    if not new["video"]:
        raise ValueError(f"{args.labels} names no source video")
    target = Path(args.into) if args.into else config.DATA_DIR / "events" / f"{Path(new['video']).stem}.events.json"
    base = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"video": new["video"], "fps": new["fps"]}
    if base.get("video") not in (None, new["video"]):
        raise ValueError(f"{target} is for {base['video']}, the labels are for {new['video']}")

    merged = merge_labels(base, new)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=JSON_INDENT) + "\n", encoding="utf-8")
    counts = ", ".join(f"{len(merged[k])} {k}" for k in LABEL_KINDS)
    print(f"Wrote {target}: {counts}; reviewed ranges {merged['ranges']}")


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: {e}")
