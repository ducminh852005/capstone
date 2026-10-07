import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_shots  # noqa: E402
import merge_labels  # noqa: E402
from core import config  # noqa: E402


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def analysis(video="v.mp4", start=1000, first=120, shots=()):
    """A minimal analysis.json: shots are (clip frame, type, hitter)."""
    return {"source": {"video": video, "start_frame": start, "analysis_start_frame": first},
            "events": {"shots": [{"frame_idx": f, "type": t, "hitter": h, "reasons": ["r"]} for f, t, h in shots]}}


def labels(video="v.mp4", **kw):
    base = {"video": video, "fps": 60, "ranges": [[1120, 1419]], "hits": [{"frame": 1150}], "smashes": [],
            "landings": [], "shots": [{"frame": 1150, "type": "clear", "hitter": "near"}]}
    base.update(kw)
    return base


def test_predictions_use_absolute_frames_and_skip_the_preroll(tmp_path):
    path = write(tmp_path / "a.json", analysis(shots=[(100, "smash", "near"), (150, "clear", "near")]))
    preds, video = eval_shots.predictions([path])
    assert video == "v.mp4" and [(p["frame"], p["type"]) for p in preds] == [(1150, "clear")]


def test_predictions_of_several_clips_of_one_video_are_pooled_but_two_videos_are_refused(tmp_path):
    a = write(tmp_path / "a.json", analysis(start=0, shots=[(130, "smash", "near")]))
    b = write(tmp_path / "b.json", analysis(start=500, shots=[(130, "drop", "near")]))
    assert [p["frame"] for p in eval_shots.predictions([a, b])[0]] == [130, 630]
    other = write(tmp_path / "c.json", analysis(video="other.mp4"))
    with pytest.raises(ValueError, match="other.mp4"):
        eval_shots.predictions([a, other])


def run_merge(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["merge_labels.py", *argv])
    merge_labels.main()


def test_merge_creates_the_events_file_of_the_video_and_then_extends_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    run_merge(monkeypatch, str(write(tmp_path / "nhan.json", labels())))
    target = tmp_path / "events" / "v.events.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["video"] == "v.mp4" and data["shots"][0]["type"] == "clear" and data["ranges"] == [[1120, 1419]]
    run_merge(monkeypatch, str(write(tmp_path / "nhan2.json", labels(
        ranges=[[1420, 1600]], hits=[{"frame": 1500}], shots=[{"frame": 1500, "type": "drop", "hitter": "near"}]))))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert [s["frame"] for s in data["shots"]] == [1150, 1500] and data["ranges"] == [[1120, 1600]]
    assert "2 shots" in capsys.readouterr().out


def test_merge_refuses_labels_of_another_video(tmp_path, monkeypatch):
    target = write(tmp_path / "events.json", {"video": "other.mp4"})
    with pytest.raises(ValueError, match="other.mp4"):
        run_merge(monkeypatch, str(write(tmp_path / "nhan.json", labels())), "--into", str(target))


def test_eval_shots_reports_the_wrong_types(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "BENCHMARK_DIR", tmp_path / "bench")
    a = write(tmp_path / "a.json", analysis(shots=[(150, "smash", "near")]))
    (tmp_path / "events").mkdir()
    write(tmp_path / "events" / "v.events.json", labels())
    monkeypatch.setattr(sys, "argv", ["eval_shots.py", str(a), "--out", "t"])
    eval_shots.main()
    out = capsys.readouterr().out
    assert "type accuracy 0%" in out and "WRONG frame 1150: labelled clear, predicted smash (r)" in out
    report = json.loads((tmp_path / "bench" / "shots_t.json").read_text(encoding="utf-8"))
    assert report["result"]["confusion"] == {"clear": {"smash": 1}}


def test_eval_shots_needs_labelled_shots(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    a = write(tmp_path / "a.json", analysis())
    (tmp_path / "events").mkdir()
    write(tmp_path / "events" / "v.events.json", labels(shots=[]))
    monkeypatch.setattr(sys, "argv", ["eval_shots.py", str(a)])
    with pytest.raises(ValueError, match="no labelled shots"):
        eval_shots.main()
