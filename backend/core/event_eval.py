"""
Scoring of detected events (hits, smashes, landings, shot types) against hand-labelled frames.

Pure functions, no video or model needed. The label file format is described in
load_events() and load_labels(); scripts/eval_events.py and scripts/eval_shots.py produce the
predictions.
"""
import json
from typing import Dict, List, Optional, Sequence, Tuple

EVENT_KINDS = ("hits", "smashes", "landings")


def match_events(pred_frames: Sequence[int], true_frames: Sequence[int], tol: int) -> Dict:
    """
    One-to-one matching of predicted frames to labelled frames within +-tol frames.

    Candidate pairs are taken closest first (ties: earlier true frame, then earlier
    prediction), and every prediction and every label is used at most once, so one
    detection cannot vouch for two labels nor the other way round.

    Returns {"tp": [(pred, true), ...], "fp": [pred, ...], "fn": [true, ...],
             "precision": float | None, "recall": float | None, "f1": float | None};
    a ratio is None when its denominator is zero (nothing predicted / nothing labelled).
    """
    pairs = sorted(
        (abs(p - t), t, p)
        for p in pred_frames for t in true_frames if abs(p - t) <= tol
    )
    used_pred, used_true, tp = set(), set(), []
    for _, t, p in pairs:
        if p in used_pred or t in used_true:
            continue
        used_pred.add(p)
        used_true.add(t)
        tp.append((p, t))

    # remaining = listed but unmatched; a frame listed twice can vouch for one label only
    fp = _remaining(pred_frames, [p for p, _ in tp])
    fn = _remaining(true_frames, [t for _, t in tp])
    precision = len(tp) / len(pred_frames) if len(pred_frames) else None
    recall = len(tp) / len(true_frames) if len(true_frames) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": sorted(tp, key=lambda pt: pt[1]), "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def _remaining(all_frames: Sequence[int], matched: List[int]) -> List[int]:
    """`all_frames` minus one occurrence per entry of `matched`, order kept."""
    left = list(matched)
    out = []
    for f in all_frames:
        if f in left:
            left.remove(f)
        else:
            out.append(f)
    return sorted(out)


def load_events(path) -> Dict[str, List[dict]]:
    """
    Read a label file:

        {"video": "tran04_cam1.mp4", "fps": 60,
         "hits":     [{"frame": 346}],
         "smashes":  [{"frame": 394, "note": "optional"}],
         "landings": [{"frame": 1498, "call": "IN" | "OUT" | null}]}

    Keys starting with "_" are ignored (comments). Returns {kind: [entry, ...]} for the kinds
    in EVENT_KINDS, each entry a dict with an integer "frame".
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    events = {}
    for kind in EVENT_KINDS:
        entries = data.get(kind, [])
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("frame"), int):
                raise ValueError(f"{path}: every '{kind}' entry needs an integer 'frame', got {e!r}")
        events[kind] = entries
    return events


def call_agreement(pred_calls: Sequence[dict], true_landings: Sequence[dict], tol: int) -> Optional[float]:
    """
    Among landings matched within tol that carry a labelled call ("IN"/"OUT"), the fraction
    where the umpire's result agrees. None if no matched landing has a labelled call.
    pred_calls: [{"frame_idx": int, "result": "IN"|"OUT"}, ...].
    """
    labelled = [t for t in true_landings if t.get("call") in ("IN", "OUT")]
    matched = match_events([c["frame_idx"] for c in pred_calls], [t["frame"] for t in labelled], tol)["tp"]
    if not matched:
        return None
    result_at = {c["frame_idx"]: c["result"] for c in pred_calls}
    call_at = {t["frame"]: t["call"] for t in labelled}
    agree = sum(result_at[p] == call_at[t] for p, t in matched)
    return agree / len(matched)


# ---- labels made in the clip viewer (shot types) ----------------------------------------------

LABEL_KINDS = EVENT_KINDS + ("shots",)


def load_labels(path) -> Dict:
    """
    Read a label file written by the clip viewer (or an events file extended with shots):

        {"video": "tran04_cam1.mp4", "fps": 60,
         "ranges": [[start_frame, end_frame]],     # stretches in which EVERY event was reviewed
         "hits": [{"frame": 646}], "smashes": [{"frame": 646}],
         "landings": [{"frame": 679, "call": "IN"}],
         "shots": [{"frame": 646, "type": "smash", "hitter": "near" | "far" | "unknown",
                    "source": "auto" | "manual", "auto_type": "smash"}]}

    Frames are absolute frames of the source video. `ranges` tells where the absence of a label
    means "nothing happened": without it only recall can be scored, not precision. Keys starting
    with "_" are ignored. Returns the dict with every kind present and ranges as (start, end) tuples.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {"video": data.get("video"), "fps": data.get("fps"), "ranges": []}
    for kind in LABEL_KINDS:
        entries = data.get(kind, [])
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("frame"), int):
                raise ValueError(f"{path}: every '{kind}' entry needs an integer 'frame', got {e!r}")
            if kind == "shots" and not (isinstance(e.get("type"), str) and isinstance(e.get("hitter"), str)):
                raise ValueError(f"{path}: every 'shots' entry needs a 'type' and a 'hitter', got {e!r}")
        out[kind] = entries
    for r in data.get("ranges", []):
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(v, int) for v in r) and r[0] <= r[1]):
            raise ValueError(f"{path}: a range must be [start_frame, end_frame], got {r!r}")
        out["ranges"].append((r[0], r[1]))
    return out


def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def merge_labels(base: Dict, new: Dict) -> Dict:
    """
    Add the labels of `new` (load_labels() output) to the raw label dict `base` (a loaded events
    file): per kind, entries with the same frame are replaced by the new one, the rest are kept,
    and the reviewed ranges are united. Keys of `base` that are not label kinds (video, fps,
    comments) are kept. Returns a new dict; `base` is not modified.
    """
    out = dict(base)
    for kind in LABEL_KINDS:
        entries = {e["frame"]: e for e in base.get(kind, [])}
        entries.update({e["frame"]: e for e in new.get(kind, [])})
        out[kind] = [entries[f] for f in sorted(entries)]
    old = [tuple(r) for r in base.get("ranges", [])]
    out["ranges"] = [list(r) for r in _merge_ranges(old + list(new.get("ranges", [])))]
    return out


def in_ranges(frame: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    return any(a <= frame <= b for a, b in ranges)


def evaluate_shots(pred_shots: Sequence[Dict], true_shots: Sequence[Dict], tol: int,
                   ranges: Sequence[Tuple[int, int]] = ()) -> Dict:
    """
    Score predicted shots against labelled ones, both lists of {"frame", "type", "hitter"} with
    absolute frames. A prediction and a label are the same hit when within +-tol frames
    (match_events). With `ranges` only the shots inside them are considered, and then false
    positives (predicted hits nobody labelled) are meaningful; without, only recall is.

    Returns {"hits": match_events result (precision None without ranges),
             "hitter_agreement": share of matched hits with the same hitter | None,
             "type_accuracy": share of matched labelled near-player hits whose type was predicted
                              right (labelled "unknown" types are not counted) | None,
             "n_typed": that denominator,
             "confusion": {labelled type: {predicted type: n}},
             "by_true_type": {labelled type: {"n": labelled, "missed": not matched, "correct": right type}}}.
    A matched hit that was attributed to someone other than the near player counts as predicted "unknown".
    """
    if ranges:
        pred_shots = [p for p in pred_shots if in_ranges(p["frame"], ranges)]
        true_shots = [t for t in true_shots if in_ranges(t["frame"], ranges)]
    hits = match_events([p["frame"] for p in pred_shots], [t["frame"] for t in true_shots], tol)
    if not ranges:
        hits["precision"], hits["f1"], hits["fp"] = None, None, []
    pred_at = {p["frame"]: p for p in pred_shots}
    true_at = {t["frame"]: t for t in true_shots}

    confusion: Dict[str, Dict[str, int]] = {}
    by_type: Dict[str, Dict[str, int]] = {}
    same_hitter = n_typed = n_right = 0
    matched_true = set()
    for p_frame, t_frame in hits["tp"]:
        pred, true = pred_at[p_frame], true_at[t_frame]
        matched_true.add(t_frame)
        same_hitter += pred["hitter"] == true["hitter"]
        if true["hitter"] == "near" and true["type"] != "unknown":
            predicted = pred["type"] if pred["hitter"] == "near" else "unknown"
            confusion.setdefault(true["type"], {}).setdefault(predicted, 0)
            confusion[true["type"]][predicted] += 1
            n_typed += 1
            n_right += predicted == true["type"]
            by_type.setdefault(true["type"], {"n": 0, "missed": 0, "correct": 0})["correct"] += predicted == true["type"]
    for t in true_shots:
        if t["hitter"] == "near" and t["type"] != "unknown":
            counts = by_type.setdefault(t["type"], {"n": 0, "missed": 0, "correct": 0})
            counts["n"] += 1
            counts["missed"] += t["frame"] not in matched_true
    return {"hits": hits,
            "hitter_agreement": same_hitter / len(hits["tp"]) if hits["tp"] else None,
            "type_accuracy": n_right / n_typed if n_typed else None, "n_typed": n_typed,
            "confusion": confusion, "by_true_type": by_type}
