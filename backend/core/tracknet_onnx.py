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
import torch

from .tracknet import TrackNetCandidateSource, heatmap_to_candidates
from .tracknet_model import HEIGHT, WIDTH, in_dim_for

logger = logging.getLogger(__name__)


class TrackNetONNXCandidateSource(TrackNetCandidateSource):
    def __init__(self, weights_path=None, onnx_path=None, batch_stride=None, conf_threshold=0.5, bg_frames=60):
        """
        weights_path: path to TrackNet_best.pt, read only for its param_dict (seq_len,
                      bg_mode) -- no PyTorch model is built or run.
        onnx_path: path to the exported .onnx file (scripts/export_tracknet_onnx.py).
                   Defaults to weights_path with its extension swapped to '.onnx'.
        batch_stride, conf_threshold, bg_frames: see TrackNetCandidateSource.
        """
        from .tracknet import DEFAULT_WEIGHTS_PATH
        weights_path = weights_path or DEFAULT_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"TrackNetV3 weights not found at {weights_path}. Download TrackNet_best.pt "
                f"from https://github.com/qaz812345/TrackNetV3 (see QUICK_START.md) and place "
                f"it at backend/TrackNet_best.pt, or pass weights_path explicitly."
            )
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

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        params = ckpt["param_dict"]
        self.seq_len = int(params["seq_len"])
        self.bg_mode = params["bg_mode"]
        in_dim_for(self.seq_len, self.bg_mode)  # validated here, same as the PyTorch path

        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        logger.info(f"TrackNet ONNX backend using providers: {self.session.get_providers()}")

        self.batch_stride = batch_stride or self.seq_len
        self.conf_threshold = conf_threshold
        self.needs_bg = bool(self.bg_mode)
        self.bg_frames_needed = bg_frames

        from collections import deque
        self._bg_samples = []
        self._bg_tensor = None  # plain float32 ndarray (3, H, W) here, not a torch tensor -- see _ensure_background
        self._window = deque(maxlen=self.seq_len)
        self._since_last_run = 0
        self._last_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    def _ensure_background(self, resized_frame):
        """Same as TrackNetCandidateSource, but keeps a plain ndarray (no torch/device)."""
        if self._bg_tensor is not None:
            return True
        self._bg_samples.append(resized_frame)
        if len(self._bg_samples) < self.bg_frames_needed:
            return False
        median = np.median(np.stack(self._bg_samples, axis=0), axis=0).astype(np.uint8)  # (H, W, 3)
        self._bg_tensor = np.moveaxis(median, -1, 0).astype(np.float32) / 255.0  # (3, H, W)
        self._bg_samples = []
        return True

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
