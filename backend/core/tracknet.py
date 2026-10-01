"""
TrackNetV3 candidate source for ShuttleDetector (see core/shuttle_tracker.py).

TrackNet always looks at a sliding window of the last `seq_len` frames (seq_len=8 for
the official checkpoint) and predicts one heatmap per frame in that window, but a
forward pass is not run every frame: `batch_stride` controls how many NEW frames
arrive between forward passes, independent of the (fixed) window size. Re-running the
network every frame (batch_stride=1, "dense") costs ~seq_len times more GPU work than
running it once every seq_len frames (batch_stride=seq_len, "nonoverlap") -- measured
on this project's GPU (T550 4GB) in FP32 (see `use_half` below): 103 ms/window either
way, i.e. ~9.7 FPS dense vs ~77 FPS nonoverlap (see scripts/bench_tracknet_forward.py).
An earlier FP16 measurement on this same GPU showed 364 ms/window -- FP16 was both
~3.5x SLOWER (this GPU has no real FP16 throughput advantage) and numerically unsafe
(see `use_half`), so it is no longer the default; do not compare FP16 and FP32 timings
as a speed/accuracy tradeoff, FP32 wins both. The default stride is
config.TRACKNET_BATCH_STRIDE (5): nonoverlap's 1-in-8 detection rate is too coarse for
RallyUmpire's landing detection (it needs to see the shuttle slow down over a couple of
consecutive detections, which nonoverlap mostly skips over), and a smaller stride
recovers temporal resolution at a proportional GPU cost.

Whatever the stride, `generate()` only has fresh candidates on the calls where a batch
resolves; the other calls return empty arrays. This is deliberately compatible with
the existing ShuttleDetector.detect() loop: an empty candidate array makes the Kalman
filter coast (predict-only), which it already does for ordinary missed detections,
and the frames without a detection are estimated afterwards by core/gap_fill.py
(ShuttleDetector.fill_gaps). The one place this needed a real design decision is
track *initiation*: the CV backend's 3-consecutive-frame consistency check assumes
candidates every frame, which never happens here at batch_stride>1. Since TrackNetV3
candidates are already high precision (a purpose-trained detector, not a generic blob
heuristic), ShuttleDetector uses a simpler single-confident-point init rule for this
backend instead (see `simple_init` in shuttle_tracker.py).
"""
import logging
import os
import pickle
from collections import deque

import cv2
import numpy as np
import torch

from . import config
from .tracknet_model import TrackNet, WIDTH, HEIGHT, in_dim_for

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = config.TRACKNET_WEIGHTS_PATH


def resolve_weights_path(weights_path=None):
    """Existing checkpoint path (default: config.TRACKNET_WEIGHTS_PATH) or FileNotFoundError."""
    weights_path = weights_path or DEFAULT_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"TrackNetV3 weights not found at {weights_path}. Download TrackNet_best.pt "
            f"from https://github.com/qaz812345/TrackNetV3 (see QUICK_START.md) and place "
            f"it at backend/models/TrackNet_best.pt, or pass weights_path explicitly."
        )
    return weights_path


def load_checkpoint(weights_path, map_location="cpu"):
    """
    torch.load restricted to tensors and plain containers (weights_only=True). Only if the
    checkpoint contains other pickled objects do we fall back to full unpickling, which can
    execute arbitrary code -- so that is logged, and the file must come from a trusted source.
    """
    try:
        return torch.load(weights_path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        logger.warning("%s needs full unpickling (weights_only=False); load only trusted checkpoints",
                       weights_path)
        return torch.load(weights_path, map_location=map_location, weights_only=False)


def heatmap_to_candidates(heat, threshold=0.5):
    """
    Pure post-processing (no model needed -- unit-testable with a hand-built heatmap).

    heat: (H, W) float array in [0, 1] (sigmoid output for one frame).
    Returns (candidates Nx2 float array in heatmap pixel space, confidences (N,)).
    Every connected component above `threshold` is a candidate (the official repo's
    predict_location() only keeps the single largest one; ShuttleDetector's Kalman
    gating benefits from seeing all of them when more than one blob is bright).
    """
    mask = (heat > threshold).astype(np.uint8) * 255
    if not mask.any():
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands, confs = [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h == 0:
            continue
        cands.append((x + w / 2.0, y + h / 2.0))
        confs.append(float(heat[y:y + h, x:x + w].max()))
    return np.array(cands, dtype=np.float64).reshape(-1, 2), np.array(confs, dtype=np.float64)


def scale_candidates(cands, roi, network_size=(WIDTH, HEIGHT)):
    """
    Pure coordinate transform (no model needed -- unit-testable).

    cands: Nx2 in network input space (network_size = (W, H)).
    roi: (x0, y0, x1, y1) of the crop that was resized to network_size.
    Returns Nx2 in full-frame pixels: scale by the crop/network size ratio, then
    offset by the crop's top-left corner. Same idea as ShuttleDetector's existing
    `_extract_candidates(fg_mask, offset)`, plus the extra resize factor TrackNet
    needs (its input is always exactly network_size regardless of ROI size).
    """
    if len(cands) == 0:
        return np.empty((0, 2), dtype=np.float64)
    x0, y0, x1, y1 = roi
    nw, nh = network_size
    sx, sy = (x1 - x0) / nw, (y1 - y0) / nh
    return np.asarray(cands, dtype=np.float64) * [sx, sy] + [x0, y0]


class TrackNetCandidateSource:
    def __init__(self, weights_path=None, device=None, batch_stride=None,
                 conf_threshold=config.TRACKNET_CONF_THRESHOLD,
                 bg_frames=config.TRACKNET_BG_FRAMES,
                 use_half=False, idle_stride=None, all_heatmaps=None):
        """
        all_heatmaps: decode every heatmap of a forward pass, not just the newest (default
            config.TRACKNET_ALL_HEATMAPS). Needs batch_stride == seq_len. generate() then replays
            the frames one per call, seq_len - 1 calls late (see `emitted_call_idx`).
        weights_path: path to TrackNet_best.pt (see QUICK_START.md for the download step).
        idle_stride: stride while no shuttle track is active (see set_active); None =
                     config.TRACKNET_IDLE_STRIDE, or batch_stride itself if that was given.
        batch_stride: how many new frames arrive between forward passes; None =
                      config.TRACKNET_BATCH_STRIDE (5), or seq_len from the checkpoint if that
                      is 0 (nonoverlap batching -- the fastest, but only 1-in-seq_len frames get
                      a detection, which is too coarse for RallyUmpire's landing detection). The model always sees a full
                      sliding window of the last seq_len frames regardless of stride, so a
                      smaller stride (e.g. 2-4) trades GPU cost for temporal resolution
                      without changing per-batch cost; batch_stride=1 is the fully dense,
                      most accurate, most expensive mode (~seq_len times the GPU cost of
                      nonoverlap -- see scripts/bench_tracknet_forward.py).
        bg_frames: frames accumulated once at start-up to build the background image
                   the checkpoint's bg_mode needs (median, same technique as
                   CourtCalibrator.extract_clean_background). Skipped if the
                   checkpoint's bg_mode is '' (no background channel).
        use_half: FP16 on CUDA. Defaults to False -- measured on this project's reference
                  GPU (T550 4GB, torch 2.5.1+cu121), FP16 overflows to non-finite on ~80%
                  of forward passes for this checkpoint, silently collapsing the real
                  detection rate from ~3% of frames to ~0% (the non-finite guard below
                  swallows it as "no detection" instead of erroring). FP32 on the same GPU
                  restores the ~3% rate with zero non-finite batches. Only pass True if
                  you've verified finite outputs on your specific GPU/torch/cuDNN build.
        """
        weights_path = resolve_weights_path(weights_path)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._d255 = torch.tensor(255.0, device=self.device)   # see _run_batch
        ckpt = load_checkpoint(weights_path, map_location=self.device)
        params = ckpt["param_dict"]
        self.seq_len = int(params["seq_len"])
        self.bg_mode = params["bg_mode"]
        in_dim = in_dim_for(self.seq_len, self.bg_mode)

        self.model = TrackNet(in_dim=in_dim, out_dim=self.seq_len).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.use_half = use_half and self.device.type == "cuda"
        if self.use_half:
            self.model = self.model.half()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self._init_state(batch_stride, conf_threshold, bg_frames, idle_stride, all_heatmaps)

    def _init_state(self, batch_stride, conf_threshold, bg_frames, idle_stride=None, all_heatmaps=None):
        """Runtime-independent state; needs self.seq_len and self.bg_mode to be set already."""
        self.batch_stride = batch_stride or config.TRACKNET_BATCH_STRIDE or self.seq_len
        # Adaptive stride: batch_stride is the stride while a shuttle track is active, idle_stride
        # the one while there is none (set_active()). An explicit batch_stride is taken literally.
        if idle_stride is None:
            idle_stride = config.TRACKNET_IDLE_STRIDE if batch_stride is None else batch_stride
        self.idle_stride = max(self.batch_stride, idle_stride or 0)
        self.all_heatmaps = config.TRACKNET_ALL_HEATMAPS if all_heatmaps is None else all_heatmaps
        if self.all_heatmaps:
            if batch_stride not in (None, 0, self.seq_len):
                raise ValueError(f"all_heatmaps needs batch_stride == seq_len ({self.seq_len}), got {batch_stride}")
            self.batch_stride = self.idle_stride = self.seq_len     # windows must not overlap
        self._stride_now = self.batch_stride       # standalone use never calls set_active: no change
        # all_heatmaps replay: (call_idx, candidates, confidences, mask) of the frames whose heatmaps
        # were already computed but not yet handed out, and the frame handed out by the last call
        self._replay = deque()
        self.emitted_call_idx = None
        self._calls = 0
        self.conf_threshold = conf_threshold
        self.needs_bg = bool(self.bg_mode)
        self.bg_frames_needed = bg_frames

        self._bg_samples = []
        self._bg_tensor = None          # background in the runtime's format, set once warm-up completes
        self._window = deque(maxlen=self.seq_len)  # sliding window of the last seq_len resized frames
        self._since_last_run = 0        # new frames seen since the last forward pass
        self._last_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self._mask_cache = None         # ((roi_w, roi_h), _last_mask resized to the ROI)

    @staticmethod
    def _prep_frame(crop):
        # resize first, then swap channels on the small image: per-channel resizing commutes with
        # the swap, and converting the 512x288 result is ~40x cheaper than the full ROI crop
        small = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _ensure_background(self, resized_frame):
        """Accumulate resized RGB frames until bg_frames_needed, then freeze their median."""
        if self._bg_tensor is not None:
            return True
        self._bg_samples.append(resized_frame)
        if len(self._bg_samples) < self.bg_frames_needed:
            return False
        median = np.median(np.stack(self._bg_samples, axis=0), axis=0).astype(np.uint8)  # (H, W, 3)
        self._bg_tensor = self._background_from_median(np.moveaxis(median, -1, 0).astype(np.float32) / 255.0)
        self._bg_samples = []
        return True

    def _background_from_median(self, chw):
        """(3, H, W) float32 in [0, 1] -> the format _run_batch expects (torch tensor here)."""
        t = torch.from_numpy(chw).to(self.device)
        return t.half() if self.use_half else t

    def set_active(self, active):
        """Tell the source whether a shuttle track is running: forward passes come every
        batch_stride frames then, every idle_stride frames otherwise."""
        self._stride_now = self.batch_stride if active else self.idle_stride

    @property
    def lagged(self):
        """True when generate() replays past frames (all_heatmaps): the candidates it returns
        belong to call `emitted_call_idx`, not to the current call."""
        return self.all_heatmaps

    @property
    def frames_per_detection(self):
        """Typical frames between two detections of a shuttle in flight (1 when all heatmaps are used)."""
        return 1 if self.all_heatmaps else self.batch_stride

    @property
    def max_frames_between_detections(self):
        """Longest gap (frames) between two detections of a shuttle in flight."""
        return 1 if self.all_heatmaps else self.idle_stride

    @property
    def warmup_frames(self):
        """Calls before the steady state: background frames, then a full window plus one stride
        (the first forward pass also pays cudnn autotuning, ~3 s, so timings must skip it)."""
        return (self.bg_frames_needed if self.needs_bg else 0) + self.seq_len + self.idle_stride

    def ready(self):
        return not self.needs_bg or self._bg_tensor is not None

    def generate(self, frame, roi):
        """
        frame: full BGR frame. roi: (x0, y0, x1, y1).
        Returns (candidates Nx2 float, full-frame pixels; confidences (N,); mask uint8
        sized to the ROI -- the most recently resolved heatmap, upsampled, for display).
        Candidates/confidences are empty on calls where nothing new resolved yet
        (background warm-up, or still filling the frame buffer).
        """
        if roi is None:
            roi = (0, 0, frame.shape[1], frame.shape[0])
        x0, y0, x1, y1 = roi
        roi_w, roi_h = x1 - x0, y1 - y0
        call_idx = self._calls
        self._calls += 1
        crop = frame[y0:y1, x0:x1]
        resized = self._prep_frame(crop) if crop.size else np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        if self.needs_bg and not self._ensure_background(resized):
            return self._nothing(roi_w, roi_h)

        self._window.append(resized)
        self._since_last_run += 1
        if len(self._window) >= self.seq_len and self._since_last_run >= self._stride_now:
            if self.all_heatmaps:
                self._queue_all_frames(call_idx, roi)
            else:
                cands, confs, heat = self._run_batch()
                self._since_last_run = 0
                self._last_mask = (heat > self.conf_threshold).astype(np.uint8) * 255
                self._mask_cache = None
                return scale_candidates(cands, roi), confs, self._mask_for(roi_w, roi_h)
        if self.all_heatmaps and self._replay:
            idx, cands, confs, mask = self._replay.popleft()
            self.emitted_call_idx = idx
            self._last_mask, self._mask_cache = mask, None
            return cands, confs, self._mask_for(roi_w, roi_h)
        return self._nothing(roi_w, roi_h)

    def _nothing(self, roi_w, roi_h):
        """No candidates on this call (warm-up, between batches, or nothing left to replay)."""
        self.emitted_call_idx = None
        return np.empty((0, 2)), np.empty((0,)), self._mask_for(roi_w, roi_h)

    def _queue_all_frames(self, call_idx, roi):
        """One forward pass -> the candidates of all seq_len frames of the window, queued for replay.
        The window holds the last seq_len calls, so frame k of the output is call call_idx - seq_len + 1 + k."""
        results = self._run_batch_all()
        self._since_last_run = 0
        first_call = call_idx - self.seq_len + 1
        for k, (cands, confs, heat) in enumerate(results):
            mask = (heat > self.conf_threshold).astype(np.uint8) * 255
            self._replay.append((first_call + k, scale_candidates(cands, roi), confs, mask))

    def _forward(self):
        """Run the network on the current window: heatmaps (1, seq_len, H, W), oldest frame first."""
        frames = np.stack(self._window, axis=0)                             # (seq_len, H, W, 3) uint8
        # The uint8 frames go to the device and are converted there (4x less to transfer, no
        # float32 pass on the CPU: ~13 ms -> ~1.5 ms per batch). Dividing by a TENSOR is
        # bit-identical to numpy's astype(float32) / 255; a python scalar would be turned into a
        # multiplication by 1/255 and differ in the last bit.
        x = (torch.from_numpy(frames).to(self.device)
             .permute(0, 3, 1, 2).reshape(-1, HEIGHT, WIDTH)                # (L*3, H, W)
             .float().div_(self._d255))
        if self.needs_bg:
            x = torch.cat([self._bg_tensor.float(), x], dim=0)
        x = x.unsqueeze(0)
        if self.use_half:
            x = x.half()

        with torch.no_grad():
            return self.model(x)                     # (1, seq_len, H, W)

    def _run_batch(self):
        """Candidates of the NEWEST frame of the window: (candidates, confidences, heatmap)."""
        y = self._forward()
        # Only the newest frame in the batch corresponds to "now"; the earlier
        # seq_len-1 heatmaps were for frames already returned (empty) to the caller.
        heat = y[0, -1].float().cpu().numpy()
        return self._decode(heat)

    def _run_batch_all(self):
        """[(candidates, confidences, heatmap)] for every frame of the window, oldest first."""
        heats = self._forward()[0].float().cpu().numpy()       # (seq_len, H, W)
        return [self._decode(h) for h in heats]

    def _decode(self, heat):
        """Heatmap -> (candidates, confidences, heatmap); a non-finite map counts as no detection."""
        if not np.isfinite(heat).all():
            # Defensive fallback, expected to be rare now that use_half defaults to False
            # (with FP16 on this project's GPU this fired on ~80% of batches -- see
            # use_half's docstring above). Treat as no detection rather than let NaN
            # silently propagate into candidate coordinates.
            logger.warning("Non-finite TrackNet heatmap, treating this batch as no detection.")
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64), np.zeros_like(heat)
        cands, confs = heatmap_to_candidates(heat, self.conf_threshold)
        return cands, confs, heat

    def _mask_for(self, roi_w, roi_h):
        """The latest heatmap mask at ROI size. Cached: it only changes when a batch resolves
        (every batch_stride frames), so callers get the same array back in between; do not modify it."""
        if roi_w <= 0 or roi_h <= 0:
            return self._last_mask
        if self._mask_cache is None or self._mask_cache[0] != (roi_w, roi_h):
            self._mask_cache = ((roi_w, roi_h),
                                cv2.resize(self._last_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST))
        return self._mask_cache[1]
