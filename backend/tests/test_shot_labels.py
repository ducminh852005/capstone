import json

import pytest

from core.event_eval import evaluate_shots, in_ranges, load_labels, merge_labels


def write(tmp_path, data, name="labels.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def shot(frame, kind, hitter="near"):
    return {"frame": frame, "type": kind, "hitter": hitter}


# ---- loading ----------------------------------------------------------------------------------

def test_load_labels_reads_every_kind_and_the_reviewed_ranges(tmp_path):
    path = write(tmp_path, {"video": "v.mp4", "fps": 60, "ranges": [[100, 200]], "hits": [{"frame": 120}],
                            "smashes": [{"frame": 120}], "landings": [{"frame": 150, "call": "IN"}],
                            "shots": [shot(120, "smash")], "_comment": "ignored"})
    labels = load_labels(path)
    assert labels["video"] == "v.mp4" and labels["ranges"] == [(100, 200)]
    assert [e["frame"] for e in labels["hits"]] == [120] and labels["shots"][0]["type"] == "smash"


def test_missing_kinds_are_empty_and_an_old_events_file_still_loads(tmp_path):
    labels = load_labels(write(tmp_path, {"video": "v.mp4", "hits": [{"frame": 5}]}))
    assert labels["shots"] == [] and labels["landings"] == [] and labels["ranges"] == []


@pytest.mark.parametrize("bad", [
    {"shots": [{"frame": 1, "type": "smash"}]},                 # no hitter
    {"shots": [{"frame": "1", "type": "smash", "hitter": "near"}]},
    {"hits": [{"nope": 1}]},
    {"ranges": [[5, 1]]},
    {"ranges": [[1]]},
])
def test_load_labels_rejects_malformed_entries(tmp_path, bad):
    with pytest.raises(ValueError):
        load_labels(write(tmp_path, bad))


# ---- merging ----------------------------------------------------------------------------------

def test_merge_adds_new_labels_replaces_the_same_frame_and_keeps_the_rest(tmp_path):
    base = {"video": "v.mp4", "fps": 60, "_note": "keep", "hits": [{"frame": 50}, {"frame": 120}],
            "landings": [{"frame": 90, "call": None, "note": "seed"}], "ranges": [[0, 100]]}
    new = load_labels(write(tmp_path, {"hits": [{"frame": 120}, {"frame": 300}],
                                       "landings": [{"frame": 90, "call": "OUT"}],
                                       "shots": [shot(300, "clear")], "ranges": [[101, 400]]}))
    merged = merge_labels(base, new)
    assert [e["frame"] for e in merged["hits"]] == [50, 120, 300]
    assert merged["landings"] == [{"frame": 90, "call": "OUT"}]            # the new label wins
    assert merged["shots"] == [shot(300, "clear")] and merged["_note"] == "keep" and merged["video"] == "v.mp4"
    assert merged["ranges"] == [[0, 400]]                                  # touching ranges are united
    assert base["hits"] == [{"frame": 50}, {"frame": 120}]                 # the input is not modified


def test_merge_keeps_separate_ranges_apart():
    merged = merge_labels({"ranges": [[0, 10]]}, {"ranges": [(50, 60)]})
    assert merged["ranges"] == [[0, 10], [50, 60]]


def test_in_ranges():
    assert in_ranges(5, [(0, 5)]) and not in_ranges(6, [(0, 5)]) and not in_ranges(1, [])


# ---- scoring ----------------------------------------------------------------------------------

def test_matched_hits_are_scored_for_hitter_and_type_with_a_confusion_matrix():
    pred = [shot(100, "smash"), shot(200, "clear"), shot(300, "drop"), shot(400, "unknown", "far")]
    true = [shot(103, "smash"), shot(198, "drop"), shot(299, "drop"), shot(402, "unknown", "far"), shot(500, "lift")]
    r = evaluate_shots(pred, true, tol=6, ranges=[(0, 1000)])
    assert len(r["hits"]["tp"]) == 4 and r["hits"]["fn"] == [500] and r["hits"]["fp"] == []
    assert r["hitter_agreement"] == 1.0
    assert (r["n_typed"], r["type_accuracy"]) == (3, pytest.approx(2 / 3))          # the far hit has no type to score
    assert r["confusion"] == {"smash": {"smash": 1}, "drop": {"clear": 1, "drop": 1}}
    assert r["by_true_type"]["drop"] == {"n": 2, "missed": 0, "correct": 1}
    assert r["by_true_type"]["lift"] == {"n": 1, "missed": 1, "correct": 0}


def test_a_hit_attributed_to_the_opponent_counts_as_an_unknown_type():
    r = evaluate_shots([shot(100, "unknown", "far")], [shot(100, "smash")], tol=6, ranges=[(0, 200)])
    assert r["hitter_agreement"] == 0.0 and r["confusion"] == {"smash": {"unknown": 1}}
    assert r["type_accuracy"] == 0.0


def test_frames_outside_the_reviewed_ranges_are_ignored_and_extra_predictions_are_false_positives():
    pred = [shot(100, "smash"), shot(150, "clear"), shot(900, "drop")]
    true = [shot(100, "smash")]
    r = evaluate_shots(pred, true, tol=6, ranges=[(0, 500)])
    assert r["hits"]["fp"] == [150] and r["hits"]["precision"] == 0.5          # 900 is outside the range


def test_without_ranges_only_recall_is_meaningful():
    r = evaluate_shots([shot(100, "smash"), shot(150, "clear")], [shot(100, "smash")], tol=6)
    assert r["hits"]["recall"] == 1.0 and r["hits"]["precision"] is None and r["hits"]["fp"] == []


def test_nothing_to_score():
    r = evaluate_shots([], [], tol=6, ranges=[(0, 10)])
    assert r["type_accuracy"] is None and r["hitter_agreement"] is None and r["confusion"] == {}
