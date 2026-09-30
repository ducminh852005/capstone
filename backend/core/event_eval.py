"""
Scoring of detected events (hits, smashes, landings) against hand-labelled frames.

Pure functions, no video or model needed. The label file format is described in
load_events(); scripts/eval_events.py produces the predictions.
"""
import json
from typing import Dict, List, Optional, Sequence

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
