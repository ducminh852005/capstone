import threading

import cv2
import numpy as np
import pytest

from core.court_calibration import CourtCalibrator
from core.video_io import ThreadedVideoReader


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
