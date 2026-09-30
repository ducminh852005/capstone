import json

import pytest

from core.event_eval import call_agreement, load_events, match_events


def test_matches_within_tolerance_and_reports_fp_fn():
    r = match_events([100, 200, 500], [104, 210, 300], tol=12)
    assert r["tp"] == [(200, 210), (100, 104)] or sorted(r["tp"]) == [(100, 104), (200, 210)]
    assert r["fp"] == [500] and r["fn"] == [300]
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_outside_tolerance_is_a_miss():
    r = match_events([100], [113], tol=12)
    assert r["tp"] == [] and r["fp"] == [100] and r["fn"] == [113]
    assert r["precision"] == 0 and r["recall"] == 0 and r["f1"] == 0.0


def test_one_prediction_cannot_vouch_for_two_labels():
    r = match_events([105], [100, 110], tol=12)
    assert len(r["tp"]) == 1 and len(r["fn"]) == 1 and r["fp"] == []


def test_one_label_cannot_be_matched_twice_and_closest_wins():
    r = match_events([98, 101], [100], tol=12)
    assert r["tp"] == [(101, 100)] and r["fp"] == [98]


def test_closest_pairs_first_beats_greedy_left_to_right():
    # 100 could take label 90 or 108; 112 only reaches 108. Closest-first gives both a match.
    r = match_events([100, 112], [90, 108], tol=12)
    assert sorted(r["tp"]) == [(100, 90), (112, 108)]


def test_empty_sides_give_none_ratios():
    r = match_events([], [10], tol=5)
    assert r["precision"] is None and r["recall"] == 0 and r["f1"] is None
    r = match_events([10], [], tol=5)
    assert r["precision"] == 0 and r["recall"] is None
    r = match_events([], [], tol=5)
    assert r["precision"] is None and r["recall"] is None and r["f1"] is None


def test_call_agreement_only_counts_matched_labelled_landings():
    preds = [{"frame_idx": 100, "result": "IN"}, {"frame_idx": 300, "result": "OUT"},
             {"frame_idx": 900, "result": "IN"}]
    truth = [{"frame": 104, "call": "IN"}, {"frame": 298, "call": "IN"}, {"frame": 500, "call": "OUT"},
             {"frame": 902, "call": None}]
    assert call_agreement(preds, truth, tol=12) == pytest.approx(0.5)   # 100 agrees, 300 does not
    assert call_agreement(preds, [{"frame": 5000, "call": "IN"}], tol=12) is None


def test_load_events_validates_and_ignores_comment_keys(tmp_path):
    good = tmp_path / "e.json"
    good.write_text(json.dumps({"_note": "x", "hits": [{"frame": 1}], "smashes": [],
                                "landings": [{"frame": 9, "call": "IN"}]}), encoding="utf-8")
    ev = load_events(good)
    assert ev["hits"] == [{"frame": 1}] and ev["smashes"] == [] and ev["landings"][0]["call"] == "IN"
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"hits": [{"frame": "12"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_events(bad)
    missing_kind = tmp_path / "m.json"
    missing_kind.write_text("{}", encoding="utf-8")
    assert load_events(missing_kind) == {"hits": [], "smashes": [], "landings": []}
