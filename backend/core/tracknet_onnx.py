"""
ONNX Runtime candidate source for ShuttleDetector (backend="tracknet-onnx"), an
alternative runtime for the same TrackNetV3 checkpoint used by
TrackNetCandidateSource (core/tracknet.py, backend="tracknet"). Same weights, same
math -- only how the forward pass is executed differs: PyTorch eager mode dispatches
each conv/batchnorm/upsample block as a separate Python call, while ONNX Runtime runs
the whole graph as one fused C++ call, which can matter at batch=1 where per-call
Python overhead is a real fraction of the total. See backend/README.md, "ONNX Runtime
backend", for the measured before/after.

Export the checkpoint first with scripts/export_tracknet_onnx.py.

Measured: the isolated forward pass is genuinely ~12% faster than PyTorch, but the full
pipeline is SLOWER end to end (every stage, including unrelated YOLO/MediaPipe stages) --
likely PyTorch and ONNX Runtime each holding a separate CUDA context/allocator on this
4GB GPU, with per-frame context-switching cost outweighing the per-call win. See
backend/README.md, "ONNX Runtime backend", for the numbers. Not the default backend;
kept correct and available, not adopted, unless YOLO is also ported to ONNX Runtime.
"""
import logging
import os

import numpy as np

from . import config
from .tracknet import TrackNetCandidateSource, heatmap_to_candidates, load_checkpoint, resolve_weights_path
from .tracknet_model import HEIGHT, WIDTH, in_dim_for

logger = logging.getLogger(__name__)


class TrackNetONNXCandidateSource(TrackNetCandidateSource):
    def __init__(self, weights_path=None, onnx_path=None, batch_stride=None,
                 conf_threshold=config.TRACKNET_CONF_THRESHOLD, bg_frames=config.TRACKNET_BG_FRAMES,
                 idle_stride=None):
        """
        weights_path: path to TrackNet_best.pt, read only for its param_dict (seq_len,
                      bg_mode) -- no PyTorch model is built or run.
        onnx_path: path to the exported .onnx file (scripts/export_tracknet_onnx.py).
                   Defaults to weights_path with its extension swapped to '.onnx'.
        batch_stride, conf_threshold, bg_frames: see TrackNetCandidateSource.
        """
        weights_path = resolve_weights_path(weights_path)
        onnx_path = onnx_path or (os.path.splitext(weights_path)[0] + ".onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX export not found at {onnx_path}. Run "
                f"'python scripts/export_tracknet_onnx.py' first."
            )

        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError(
                "backend='tracknet-onnx' needs onnxruntime-gpu (pip install onnxruntime-gpu)."
            ) from e

        ckpt = load_checkpoint(weights_path)
        params = ckpt["param_dict"]
        self.seq_len = int(params["seq_len"])
        self.bg_mode = params["bg_mode"]
        in_dim_for(self.seq_len, self.bg_mode)  # validated here, same as the PyTorch path

        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        logger.info("TrackNet ONNX backend using providers: %s", self.session.get_providers())

        self._init_state(batch_stride, conf_threshold, bg_frames, idle_stride, all_heatmaps=False)

    def _background_from_median(self, chw):
        """Keep a plain float32 ndarray (no torch/device) for ONNX Runtime."""
        return chw

    def _run_batch(self):
        frames = np.stack(self._window, axis=0)                       # (seq_len, H, W, 3) uint8
        chw = np.moveaxis(frames, -1, 1).astype(np.float32) / 255.0   # (L, 3, H, W)
        chw = chw.reshape(-1, HEIGHT, WIDTH)                          # (L*3, H, W)
        if self.needs_bg:
            chw = np.concatenate([self._bg_tensor, chw], axis=0)
        x = chw[np.newaxis, ...]                                      # (1, in_dim, H, W)

        y = self.session.run(None, {self._input_name: x})[0]          # (1, seq_len, H, W)
        heat = y[0, -1]
        if not np.isfinite(heat).all():
            logger.warning("Non-finite TrackNet(ONNX) heatmap, treating this batch as no detection.")
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.zeros_like(heat)
        cands, confs = heatmap_to_candidates(heat, self.conf_threshold)
        return cands, confs, heat
