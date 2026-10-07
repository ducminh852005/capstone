import shutil

import cv2
import numpy as np
import pytest

from core.video_processor import build_cut_command, cut_clip

FPS = 30.0
STEP = 7            # grey-level step between consecutive frames: far above codec noise


def make_video(path, n_frames=36, size=(64, 48)):
    """Frame k is a flat grey image of level k * STEP, so a decoded frame tells which one it is."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)
    assert writer.isOpened()
    for k in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), k * STEP, np.uint8))
    writer.release()
    return path


def grey_levels(path):
    cap = cv2.VideoCapture(str(path))
    levels = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        levels.append(float(frame.mean()))
    cap.release()
    return levels


def test_cut_command_seeks_half_a_frame_before_the_first_wanted_frame():
    cmd = build_cut_command("in.mp4", "out.mp4", start_frame=300, n_frames=120, fps=60.0,
                            crf=18, preset="veryfast", gop_frames=12)
    assert cmd[cmd.index("-ss") + 1] == f"{(300 - 0.5) / 60.0:.6f}"
    assert cmd.index("-ss") < cmd.index("-i")                     # input seek: fast, and exact with a re-encode
    assert cmd[cmd.index("-frames:v") + 1] == "120"
    assert cmd[cmd.index("-bf") + 1] == "0" and cmd[cmd.index("-g") + 1] == "12"
    assert cmd[cmd.index("-r") + 1] == "60"
    assert "-an" in cmd and cmd[-1] == "out.mp4"


def test_cut_command_from_the_first_frame_does_not_seek_before_zero():
    cmd = build_cut_command("in.mp4", "out.mp4", start_frame=0, n_frames=10, fps=60.0)
    assert cmd[cmd.index("-ss") + 1] == "0.000000"


@pytest.mark.parametrize("start, n, fps", [(-1, 10, 60.0), (0, 0, 60.0), (5, 10, 0.0)])
def test_cut_command_rejects_invalid_ranges(start, n, fps):
    with pytest.raises(ValueError):
        build_cut_command("in.mp4", "out.mp4", start_frame=start, n_frames=n, fps=fps)


def test_cut_clip_missing_source_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        cut_clip(str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4"), 0, 10, FPS)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_cut_clip_is_frame_accurate(tmp_path):
    src = make_video(tmp_path / "src.avi")
    out = tmp_path / "sub" / "clip.mp4"
    cut_clip(str(src), str(out), start_frame=10, n_frames=8, fps=FPS)
    levels = grey_levels(out)
    assert len(levels) == 8
    expected = [(10 + k) * STEP for k in range(8)]
    assert levels == pytest.approx(expected, abs=STEP / 2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_cut_clip_failure_raises_runtimeerror_with_ffmpeg_output(tmp_path):
    bad = tmp_path / "not_a_video.avi"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        cut_clip(str(bad), str(tmp_path / "out.mp4"), 0, 5, FPS)
