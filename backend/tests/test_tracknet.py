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
    pytest.importorskip("onnxruntime")      # the file can exist while the package is not installed
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


# --- optimisations must not change what the network sees ----------------------------------------

def test_prep_frame_resize_then_swap_equals_swap_then_resize():
    import cv2
    from core.tracknet import TrackNetCandidateSource

    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (450, 800, 3), dtype=np.uint8)
    old = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(TrackNetCandidateSource._prep_frame(crop), old)
    view = rng.integers(0, 256, (900, 1600, 3), dtype=np.uint8)[100:600, 200:1100]      # non-contiguous ROI view
    old = cv2.resize(cv2.cvtColor(view, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(TrackNetCandidateSource._prep_frame(view), old)


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_gpu_side_conversion_is_bit_identical_to_the_numpy_one():
    """The batch input built from uint8 on the device must equal numpy's astype(float32) / 255."""
    import torch
    from core.tracknet import TrackNetCandidateSource

    src = TrackNetCandidateSource()
    captured = {}

    class Stub:
        def __call__(self, x):
            captured["x"] = x.detach().cpu()
            return torch.zeros(1, src.seq_len, H, W, device=x.device)

    src.model = Stub()
    rng = np.random.default_rng(1)
    for _ in range(src.seq_len):
        src._window.append(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    src._bg_tensor = torch.zeros(3, H, W, device=src.device)
    src._run_batch()

    frames = np.stack(list(src._window), axis=0)
    expected = (np.moveaxis(frames, -1, 1).astype(np.float32) / 255.0).reshape(-1, H, W)
    got = captured["x"][0, 3:].numpy()          # skip the 3 background channels
    assert got.dtype == np.float32 and np.array_equal(got, expected)


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_mask_is_cached_between_batches_and_refreshed_when_one_resolves():
    from core.tracknet import TrackNetCandidateSource

    src = TrackNetCandidateSource(batch_stride=4, bg_frames=2)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    roi = (100, 50, 900, 500)
    masks, resolved = [], []
    for _ in range(40):
        _, _, mask = src.generate(frame, roi)
        masks.append(mask)
        resolved.append(src._since_last_run == 0 and len(src._window) == src.seq_len)
    # consecutive frames without a new batch share one array; a batch produces a new one
    same = [masks[i] is masks[i - 1] for i in range(1, len(masks))]
    assert any(same) and not all(same)
    for i in range(1, len(masks)):
        if resolved[i]:
            assert masks[i] is not masks[i - 1]
    assert all(m.shape == (roi[3] - roi[1], roi[2] - roi[0]) for m in masks)


# --- adaptive stride: look less often while nothing is in play --------------------------------

@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_adaptive_stride_uses_idle_stride_until_a_track_is_active():
    from core import config
    from core.tracknet import TrackNetCandidateSource

    src = TrackNetCandidateSource(bg_frames=2)                 # default: adaptive
    assert (src.batch_stride, src.idle_stride) == (config.TRACKNET_BATCH_STRIDE, config.TRACKNET_IDLE_STRIDE)
    assert src.max_frames_between_detections == src.idle_stride > src.batch_stride
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    roi = (100, 50, 900, 500)

    def resolutions(active, n=80):
        src.set_active(active)
        count = 0
        for _ in range(n):
            src.generate(frame, roi)
            count += src._since_last_run == 0 and len(src._window) == src.seq_len
        return count

    resolutions(False, 20)                     # warm up: background + first window
    idle, active = resolutions(False), resolutions(True)
    assert idle == 80 // src.idle_stride and active == 80 // src.batch_stride
    assert idle < active


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_explicit_batch_stride_is_taken_literally_and_standalone_use_is_not_adaptive():
    from core.tracknet import TrackNetCandidateSource

    fixed = TrackNetCandidateSource(batch_stride=3)
    assert fixed.idle_stride == fixed.batch_stride == 3 and fixed.max_frames_between_detections == 3
    explicit_idle = TrackNetCandidateSource(batch_stride=3, idle_stride=7)
    assert explicit_idle.idle_stride == 7
    assert explicit_idle._stride_now == 3       # until someone calls set_active(False)
    assert TrackNetCandidateSource(batch_stride=6, idle_stride=2).idle_stride == 6     # never faster when idle


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_detector_drives_the_stride_from_its_track_state():
    from core.shuttle_tracker import ShuttleDetector

    det = ShuttleDetector(backend="tracknet")
    assert det._adaptive
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    det.detect(frame, roi=(100, 50, 900, 500))
    assert not det.track_active and det._source._stride_now == det._source.idle_stride
    det.track_active = True                     # a track appears
    det.detect(frame, roi=(100, 50, 900, 500))
    assert det._source._stride_now in (det._source.batch_stride, det._source.idle_stride)
    fixed = ShuttleDetector(backend="tracknet", tracknet_kwargs={"batch_stride": 4})
    assert not fixed._adaptive
    assert not ShuttleDetector(backend="cv")._adaptive


# --- all_heatmaps: every heatmap of a pass is used; frames are replayed one per call -----------

@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_all_heatmaps_replays_every_frame_in_order_with_a_constant_lag():
    import torch
    from core.tracknet import TrackNetCandidateSource

    src = TrackNetCandidateSource(all_heatmaps=True, bg_frames=2)
    assert src.lagged and src.batch_stride == src.idle_stride == src.seq_len
    assert src.frames_per_detection == 1 and src.max_frames_between_detections == 1
    lag = src.seq_len - 1

    class Stub:
        """A 'network' that puts a blob at x = 10 + 4 * (the call index painted into that frame)."""

        def __call__(self, x):
            out = torch.zeros(1, src.seq_len, H, W, device=x.device)
            for k in range(src.seq_len):
                call = int(round(float(x[0, 3 + 3 * k, 0, 0]) * 255))     # channels 0-2 are the background
                out[0, k] = torch.from_numpy(gaussian_heatmap(10 + 4 * call, 100.0, sigma=2.0)).float()
            return out

    src.model = Stub()
    roi = (0, 0, W, H)                                                      # scale 1: candidates are in network pixels
    emitted, first = [], None
    for call in range(60):
        frame = np.full((H, W, 3), call, np.uint8)                          # the call index, as a grey level
        cands, confs, mask = src.generate(frame, roi)
        if src.emitted_call_idx is None:
            assert len(cands) == 0 and first is None, "no candidates before the first replay, none skipped afterwards"
            continue
        first = call if first is None else first
        emitted.append(src.emitted_call_idx)
        assert call - src.emitted_call_idx == lag                           # constant lag
        assert len(cands) == 1 and abs(cands[0][0] - (10 + 4 * src.emitted_call_idx)) < 1.0    # the right frame's blob
        assert mask.shape == (H, W) and mask.any()
    assert emitted == list(range(emitted[0], emitted[0] + len(emitted)))    # every frame, in order, once
    assert first is not None and len(emitted) == 60 - first


@pytest.mark.skipif(not os.path.exists(DEFAULT_WEIGHTS_PATH), reason="models/TrackNet_best.pt not downloaded")
def test_all_heatmaps_rejects_a_stride_that_would_overlap_windows():
    from core.tracknet import TrackNetCandidateSource

    with pytest.raises(ValueError):
        TrackNetCandidateSource(all_heatmaps=True, batch_stride=5)
    src = TrackNetCandidateSource(all_heatmaps=True, batch_stride=8)       # = seq_len: fine
    assert src.batch_stride == 8
    assert not TrackNetCandidateSource(all_heatmaps=False).lagged
