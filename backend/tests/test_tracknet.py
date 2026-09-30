import os

import numpy as np
import pytest

from core.tracknet_model import in_dim_for
from core.tracknet import heatmap_to_candidates, scale_candidates, DEFAULT_WEIGHTS_PATH

H, W = 288, 512


def gaussian_heatmap(cx, cy, sigma=3.0, size=(H, W), amplitude=0.9):
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    return amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def test_heatmap_to_candidates_finds_the_peak():
    heat = gaussian_heatmap(320.0, 150.0)
    cands, confs = heatmap_to_candidates(heat, threshold=0.5)
    assert len(cands) == 1
    assert np.hypot(*(cands[0] - [320.0, 150.0])) < 2.0
    assert confs[0] > 0.5


def test_heatmap_to_candidates_empty_when_nothing_above_threshold():
    heat = gaussian_heatmap(100, 100, amplitude=0.3)  # never crosses 0.5
    cands, confs = heatmap_to_candidates(heat, threshold=0.5)
    assert len(cands) == 0 and len(confs) == 0

    heat = np.zeros((H, W), dtype=np.float64)
    cands, confs = heatmap_to_candidates(heat, threshold=0.5)
    assert len(cands) == 0 and len(confs) == 0


def test_heatmap_to_candidates_reports_every_blob():
    heat = np.maximum(gaussian_heatmap(100, 80), gaussian_heatmap(400, 200, amplitude=0.7))
    cands, confs = heatmap_to_candidates(heat, threshold=0.5)
    assert len(cands) == 2
    order = np.argsort(cands[:, 0])
    assert np.hypot(*(cands[order[0]] - [100, 80])) < 2.0
    assert np.hypot(*(cands[order[1]] - [400, 200])) < 2.0
    # the brighter blob has the higher confidence
    assert confs[order[0]] > confs[order[1]]


def test_scale_candidates_maps_network_space_to_full_frame():
    # roi is a 1024x576 crop starting at (200, 100) -> exactly 2x the network's 512x288
    roi = (200, 100, 200 + 1024, 100 + 576)
    cands = np.array([[0.0, 0.0], [512.0, 288.0], [256.0, 144.0]])
    out = scale_candidates(cands, roi)
    expected = np.array([[200, 100], [200 + 1024, 100 + 576], [200 + 512, 100 + 288]])
    assert np.allclose(out, expected)


def test_scale_candidates_empty_input():
    assert scale_candidates(np.empty((0, 2)), (0, 0, 512, 288)).shape == (0, 2)


@pytest.mark.parametrize("bg_mode,seq_len,expected", [
    ("", 8, 24),
    ("subtract", 8, 8),
    ("subtract_concat", 8, 32),
    ("concat", 8, 27),
    ("concat", 3, 12),
])
def test_in_dim_for_matches_upstream_get_model(bg_mode, seq_len, expected):
    assert in_dim_for(seq_len, bg_mode) == expected


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_tracknet_candidate_source_real_checkpoint_forward_pass():
    from core.tracknet import TrackNetCandidateSource

    src = TrackNetCandidateSource()
    assert src.seq_len > 0
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    roi = (100, 50, 100 + 800, 50 + 450)

    last_mask = None
    for _ in range(src.bg_frames_needed + src.seq_len + 1):
        cands, confs, mask = src.generate(frame, roi)
        assert cands.shape[1] == 2 if len(cands) else True
        assert len(cands) == len(confs)
        assert mask.shape == (roi[3] - roi[1], roi[2] - roi[0])
        last_mask = mask
    assert src.ready()
    assert last_mask is not None


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_tracknet_batch_stride_shorter_than_seq_len_runs_more_often():
    """
    batch_stride < seq_len must still feed the model a full seq_len-frame window (a past
    bug ran the model on only batch_stride frames, mismatching the checkpoint's in_dim).
    A smaller stride should resolve a batch more often (finer temporal resolution) for
    the same number of frames.
    """
    from core.tracknet import TrackNetCandidateSource

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    roi = (100, 50, 100 + 800, 50 + 450)
    n_frames = 100

    def count_resolutions(src):
        n = 0
        for _ in range(n_frames):
            src.generate(frame, roi)
            if src.ready() and src._since_last_run == 0 and len(src._window) == src.seq_len:
                n += 1
        return n

    from core import config
    src_default = TrackNetCandidateSource(batch_stride=None)      # config.TRACKNET_BATCH_STRIDE
    assert src_default.batch_stride == config.TRACKNET_BATCH_STRIDE
    src_slow = TrackNetCandidateSource(batch_stride=8)            # nonoverlap for the 8-frame checkpoint
    assert src_slow.batch_stride == src_slow.seq_len
    resolved_slow = count_resolutions(src_slow)
    src_fast = TrackNetCandidateSource(batch_stride=2)
    resolved_fast = count_resolutions(src_fast)
    resolved_default = count_resolutions(TrackNetCandidateSource(batch_stride=None))

    assert src_fast.batch_stride == 2
    assert resolved_fast > resolved_default > resolved_slow


DEFAULT_ONNX_PATH = os.path.splitext(DEFAULT_WEIGHTS_PATH)[0] + ".onnx"


@pytest.mark.skipif(not os.path.exists(DEFAULT_ONNX_PATH), reason="TrackNet_best.onnx not exported")
def test_tracknet_onnx_candidate_source_real_checkpoint_forward_pass():
    from core.tracknet_onnx import TrackNetONNXCandidateSource

    src = TrackNetONNXCandidateSource()
    assert src.seq_len > 0
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    roi = (100, 50, 100 + 800, 50 + 450)

    last_mask = None
    for _ in range(src.bg_frames_needed + src.seq_len + 1):
        cands, confs, mask = src.generate(frame, roi)
        assert cands.shape[1] == 2 if len(cands) else True
        assert len(cands) == len(confs)
        assert mask.shape == (roi[3] - roi[1], roi[2] - roi[0])
        last_mask = mask
    assert src.ready()
    assert last_mask is not None
