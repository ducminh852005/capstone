import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from core import config, court_model
from core.clip_report import (ANALYSIS_FILE, HTML_FILE, PLACEHOLDER_DATA, PLACEHOLDER_VIDEO, inline_json, render_html,
                              write_clip_files)

TEMPLATE = (f"<html><head><title>__TITLE__</title></head><body>"
            f'<video src="{PLACEHOLDER_VIDEO}"></video>'
            f'<script id="analysis-data" type="application/json">{PLACEHOLDER_DATA}</script></body></html>')

DATA_BLOCK = re.compile(r'<script id="analysis-data" type="application/json">(.*?)</script>', re.S)


def embedded(html_text):
    return json.loads(DATA_BLOCK.search(html_text).group(1))


def test_render_fills_data_video_and_title_and_the_json_round_trips():
    analysis = {"label": "Ngoài sân", "n": [1, 2.5, None]}
    out = render_html(TEMPLATE, analysis, "clip.mp4", title="Clip <1> & co")
    assert 'src="clip.mp4"' in out
    assert "<title>Clip &lt;1&gt; &amp; co</title>" in out
    assert embedded(out) == analysis


def test_data_cannot_break_out_of_the_script_element():
    nasty = {"a": "</script><script>alert(1)</script>", "b": "<!-- x -->", "c": "line\u2028sep\u2029"}
    out = render_html(TEMPLATE, nasty, "clip.mp4")
    assert out.count("</script>") == 1                           # only the closing tag of the data block
    assert "<!--" not in out and "\u2028" not in out and "\u2029" not in out
    assert embedded(out) == nasty


def test_placeholder_text_inside_the_data_is_left_alone():
    analysis = {"x": PLACEHOLDER_VIDEO, "y": PLACEHOLDER_DATA}
    assert embedded(render_html(TEMPLATE, analysis, "clip.mp4")) == analysis


@pytest.mark.parametrize("bad", ["", "/abs/clip.mp4", "C:/data/clip.mp4", "C:\\data\\clip.mp4", "sub\\clip.mp4",
                                 "\\\\server\\share\\clip.mp4"])
def test_video_path_must_be_relative_posix(bad):
    with pytest.raises(ValueError, match="relative"):
        render_html(TEMPLATE, {}, bad)


@pytest.mark.parametrize("broken", [TEMPLATE.replace(PLACEHOLDER_DATA, ""), TEMPLATE + PLACEHOLDER_DATA,
                                    TEMPLATE.replace(PLACEHOLDER_VIDEO, "")])
def test_template_must_contain_each_placeholder_exactly_once(broken):
    with pytest.raises(ValueError, match="exactly once"):
        render_html(broken, {}, "clip.mp4")


def test_inline_json_is_compact_keeps_utf8_and_has_no_raw_script_breakers():
    text = inline_json({"vi": "Trong", "ngoai": "Ngoài", "x": "<b>" + chr(0x2028)})
    assert "Ngoài" in text and " " not in text
    assert "<" not in text and chr(0x2028) not in text and chr(0x2029) not in text


def test_write_clip_files_writes_the_data_and_a_page_that_embeds_it(tmp_path):
    template = tmp_path / "t.html"
    template.write_text(TEMPLATE, encoding="utf-8")
    analysis = {"label": "Ngoài"}
    json_path, html_path = write_clip_files(tmp_path / "out" / "c1", analysis, "clip.mp4", title="c1",
                                            template_path=template)
    assert (json_path.name, html_path.name) == (ANALYSIS_FILE, HTML_FILE)
    assert json.loads(json_path.read_text(encoding="utf-8")) == analysis
    assert embedded(html_path.read_text(encoding="utf-8")) == analysis


def test_missing_template_names_the_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="telestrator.html"):
        write_clip_files(tmp_path, {}, "clip.mp4", template_path=tmp_path / "nope.html")


# ---- the real viewer template -----------------------------------------------------------------

def real_template():
    return config.TELESTRATOR_TEMPLATE_PATH.read_text(encoding="utf-8")


def test_the_real_template_has_the_placeholders_once_and_no_absolute_paths():
    text = real_template()
    for placeholder in (PLACEHOLDER_DATA, PLACEHOLDER_VIDEO):
        assert text.count(placeholder) == 1
    assert not re.search(r'(src|href)="(file:|[A-Za-z]:[\\/]|/)', text)
    assert "http://" not in text and "https://" not in text      # no external resources: works offline


def test_the_real_template_renders_with_vietnamese_labels():
    out = render_html(real_template(), {"x": 1}, "clip.mp4", title="Trận 4")
    assert "Mũi tên mặt sân" in out and "Khung xương" in out and "<title>Trận 4</title>" in out


# ---- the viewer's math (pure JS functions) must agree with Python -----------------------------

NODE = shutil.which("node")
MATH = re.compile(r"/\* MATH-BEGIN.*?\*/(.*?)/\* MATH-END \*/", re.S)


def run_js(tmp_path, body, data):
    """Evaluate `body` in node after the template's pure-math region; `input` is `data`, the printed
    JSON is returned."""
    region = MATH.search(real_template()).group(1)
    script = tmp_path / "math.js"
    script.write_text(region + "\nconst input = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" + body,
                      encoding="utf-8")
    result = subprocess.run([NODE, str(script)], input=json.dumps(data), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_homography_matches_opencv(tmp_path):
    H = court_model.homography_from_points([(300, 900), (1600, 900), (500, 500), (1400, 500)])
    H_inv = np.linalg.inv(H)
    world = [(0.0, 0.0), (6.7, 6.1), (3.35, 3.05), (2.0, 1.0), (10.0, 3.0)]
    img = court_model.world_to_img(world, H)
    body = """
      const toImg = makeProjector(input.H, 3.35, 3.05);
      const toWorld = makeProjector(input.H_inv, input.img[2][0], input.img[2][1]);
      console.log(JSON.stringify({img: input.world.map(p => toImg(p[0], p[1])),
                                  world: input.img.map(p => toWorld(p[0], p[1]))}));
    """
    out = run_js(tmp_path, body, {"H": H.tolist(), "H_inv": H_inv.tolist(), "world": world, "img": img.tolist()})
    assert np.asarray(out["img"]) == pytest.approx(img, abs=1e-2)
    assert np.asarray(out["world"]) == pytest.approx(court_model.img_to_world(img, H_inv), abs=1e-3)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_projector_rejects_points_beyond_the_horizon(tmp_path):
    H = [[1, 0, 0], [0, 1, 0], [0.1, 0, 1]]                       # w = 0.1 x + 1: negative for x < -10
    body = """
      const p = makeProjector(input.H, 1, 0);
      console.log(JSON.stringify({front: p(2, 3), behind: p(-20, 3), horizon: p(-10, 3)}));
    """
    out = run_js(tmp_path, body, {"H": H})
    assert out["front"] == pytest.approx([2 / 1.2, 3 / 1.2])
    assert out["behind"] is None and out["horizon"] is None


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_frame_time_mapping_is_exact_for_every_frame(tmp_path):
    body = """
      const bad = [];
      for (const fps of [30, 59.94, 60]) {
        for (let f = 0; f < 3000; f++) if (frameAtTime(timeOfFrame(f, fps), fps, 3000) !== f) bad.push([fps, f]);
      }
      console.log(JSON.stringify({bad, clampLow: frameAtTime(-1, 60, 10), clampHigh: frameAtTime(99, 60, 10)}));
    """
    out = run_js(tmp_path, body, {})
    assert out == {"bad": [], "clampLow": 0, "clampHigh": 9}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_arrow_visibility_and_floor_measurement(tmp_path):
    body = """
      const stub = w => 10 + w[0];                                // cm per px grows with the distance
      const nullStub = () => null;
      console.log(JSON.stringify({
        thisFrame: lastVisibleFrame(100, 0, 60, 500), second: lastVisibleFrame(100, 1, 60, 500),
        half: lastVisibleFrame(100, '0.5', 60, 500), toEnd: lastVisibleFrame(100, 'end', 60, 500),
        clipped: lastVisibleFrame(480, 2, 60, 500),
        near: measureFloor([1, 1], [4, 5], 6.7, stub), far: measureFloor([5, 1], [8, 1], 6.7, stub),
        noScale: measureFloor([1, 1], [2, 1], 6.7, nullStub)}));
    """
    out = run_js(tmp_path, body, {})
    assert (out["thisFrame"], out["second"], out["half"], out["toEnd"], out["clipped"]) == (100, 160, 130, 499, 499)
    assert out["near"] == {"len": 5.0, "dx": 3.0, "dy": 4.0, "far": False, "cm": 14.0}
    assert out["far"]["far"] is True and out["far"]["len"] == 3.0 and out["far"]["cm"] == 18.0
    assert out["noScale"]["cm"] is None


LABEL_CASE = {
    "ctx": {"video": "v.mp4", "fps": 60, "startFrame": 1000, "preFrames": 120, "nFrames": 420,
            "hits": [{"index": 0, "contact_frame_idx": 150, "valid": True, "hitter": "near", "player_id": 1},
                     {"index": 1, "contact_frame_idx": 200, "valid": True, "hitter": "near", "player_id": 1},
                     {"index": 2, "contact_frame_idx": 250, "valid": False, "hitter": "far", "player_id": None},
                     {"index": 3, "contact_frame_idx": 300, "valid": True, "hitter": "far", "player_id": None},
                     {"index": 4, "contact_frame_idx": 350, "valid": True, "hitter": "near", "player_id": 1}],
            "shotType": {"0": "smash", "1": "clear", "4": "drop"},
            "landings": [{"frame_idx": 320, "result": "IN"}, {"frame_idx": 340, "result": "OUT"},
                         {"frame_idx": 360, "result": "IN"}]},
    "state": {"hits": {"0": {"reviewed": True},                                  # accepted as suggested
                       "1": {"reviewed": True, "type": "lift"},                 # corrected
                       "2": {"reviewed": True},                                 # rejected suggestion, stays no hit
                       "3": {"reviewed": True, "isHit": False},                 # suggested hit that is not one
                       "4": {"type": "net"}},                                   # not reviewed
              "landings": {"0": {"reviewed": True}, "1": {"reviewed": True, "call": "IN"},
                           "2": {"reviewed": True, "call": "none"}},
              "added": [{"frame": 180, "type": "drive", "hitter": "near"}], "rangeReviewed": True},
}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_label_export_has_absolute_frames_only_reviewed_items_and_the_reviewed_range(tmp_path):
    body = "console.log(JSON.stringify(buildLabelExport(input.ctx, input.state)));"
    out = run_js(tmp_path, body, LABEL_CASE)
    assert (out["video"], out["fps"]) == ("v.mp4", 60) and out["ranges"] == [[1120, 1419]]
    assert [h["frame"] for h in out["hits"]] == [1150, 1180, 1200]                 # P3 and P4 and the unreviewed one are out
    assert [s["frame"] for s in out["smashes"]] == [1150]
    shots = {s["frame"]: s for s in out["shots"]}
    assert shots[1150] == {"frame": 1150, "type": "smash", "hitter": "near", "player_id": 1, "source": "auto",
                           "auto_type": "smash"}
    assert (shots[1200]["type"], shots[1200]["source"], shots[1200]["auto_type"]) == ("lift", "manual", "clear")
    assert (shots[1180]["type"], shots[1180]["source"], shots[1180]["auto_type"]) == ("drive", "manual", None)
    assert out["landings"] == [{"frame": 1320, "call": "IN"}, {"frame": 1340, "call": "IN"}]   # call corrected, "none" dropped


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_label_export_without_the_whole_range_reviewed_has_no_ranges(tmp_path):
    case = json.loads(json.dumps(LABEL_CASE))
    case["state"]["rangeReviewed"] = False
    body = "console.log(JSON.stringify(buildLabelExport(input.ctx, input.state)));"
    assert run_js(tmp_path, body, case)["ranges"] == []


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_exported_labels_load_and_score_in_python(tmp_path):
    from core.event_eval import evaluate_shots, load_labels
    body = "console.log(JSON.stringify(buildLabelExport(input.ctx, input.state)));"
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(run_js(tmp_path, body, LABEL_CASE)), encoding="utf-8")
    labels = load_labels(path)
    assert labels["ranges"] == [(1120, 1419)] and len(labels["shots"]) == 3
    pred = [{"frame": 1150, "type": "smash", "hitter": "near"}, {"frame": 1200, "type": "clear", "hitter": "near"}]
    r = evaluate_shots(pred, labels["shots"], tol=6, ranges=labels["ranges"])
    assert r["type_accuracy"] == 0.5 and r["hits"]["recall"] == pytest.approx(2 / 3)    # the hand-added drive was missed


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_viewer_script_is_valid_javascript(tmp_path):
    """The page's main script has no other test than the pure-math region: at least it must parse."""
    text = real_template()
    script = re.search(r"<script>\n\(\(\) => \{.*?\}\)\(\);\n</script>", text, re.S).group(0)[len("<script>"):-len("</script>")]
    path = tmp_path / "viewer.js"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
