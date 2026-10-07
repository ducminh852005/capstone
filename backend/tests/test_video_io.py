import threading

import cv2
import numpy as np
import pytest

from core.court_calibration import CourtCalibrator
from core import config
from core.video_io import ThreadedVideoReader, probe_video


def make_video(path, n_frames=30, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i, np.uint8))
    writer.release()
    return path


def test_reader_yields_every_frame_with_absolute_index(tmp_path):
    path = make_video(tmp_path / "v.avi")
    with ThreadedVideoReader(str(path)) as reader:
        idxs = [i for i, _ in reader]
    assert idxs == list(range(30))


def test_release_with_unconsumed_frames_does_not_hang(tmp_path):
    path = make_video(tmp_path / "v.avi", n_frames=60)
    reader = ThreadedVideoReader(str(path), queue_size=2)
    next(iter(reader))                       # consume one frame, leave the queue full
    done = threading.Event()
    threading.Thread(target=lambda: (reader.release(), done.set()), daemon=True).start()
    assert done.wait(timeout=5), "release() blocked on a full queue"


def test_unreadable_video_raises_ioerror_everywhere(tmp_path):
    missing = str(tmp_path / "missing.avi")
    with pytest.raises(IOError):
        ThreadedVideoReader(missing)
    with pytest.raises(IOError):
        CourtCalibrator().extract_clean_background(missing)


def test_decoder_thread_error_reaches_the_consumer_instead_of_hanging(tmp_path, monkeypatch):
    path = make_video(tmp_path / "v.avi", n_frames=20)
    real_capture = cv2.VideoCapture

    class Flaky:
        """A capture whose decoder dies on the 4th read."""

        def __init__(self, p):
            self._cap, self._n = real_capture(p), 0

        def read(self):
            self._n += 1
            if self._n > 3:
                raise RuntimeError("decoder blew up")
            return self._cap.read()

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", Flaky)
    reader = ThreadedVideoReader(str(path))
    got, result = [], {}

    def consume():
        try:
            for item in reader:
                got.append(item[0])
        except RuntimeError as e:
            result["error"] = str(e)

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "iteration hung after the decoder thread died"
    assert got == [0, 1, 2] and result.get("error") == "decoder blew up"
    reader.release()


def test_probe_video_reads_container_properties_without_decoding(tmp_path):
    path = make_video(tmp_path / "v.avi", n_frames=12, size=(64, 48))
    probe = probe_video(path)
    assert probe.fps == pytest.approx(30.0)
    assert probe.size == (64, 48)
    assert probe.frame_count == 12


def test_probe_video_unreadable_raises_ioerror(tmp_path):
    with pytest.raises(IOError):
        probe_video(tmp_path / "missing.avi")


def test_probe_video_warns_and_assumes_cfr_fps_when_the_container_reports_none(tmp_path, monkeypatch, caplog):
    path = make_video(tmp_path / "v.avi")
    real_capture = cv2.VideoCapture

    class NoFps:
        """A capture that reports no frame rate."""

        def __init__(self, p):
            self._cap = real_capture(p)

        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", NoFps)
    with caplog.at_level("WARNING", logger="core.video_io"):
        probe = probe_video(path)
    assert probe.fps == config.CFR_FPS
    assert any("no fps" in r.message for r in caplog.records)


def test_default_queue_is_small():
    import inspect
    assert inspect.signature(ThreadedVideoReader.__init__).parameters["queue_size"].default == 4
