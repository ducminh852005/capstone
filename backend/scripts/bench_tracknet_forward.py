"""
Step-0 gate for the TrackNetV3 integration: load the real downloaded checkpoint and
measure raw forward-pass latency on this machine's GPU, BEFORE building the rest of
the integration around it. Run this first; if it fails to load or is too slow even
after batching, stop and reconsider before writing more code.

Usage: python scripts/bench_tracknet_forward.py [path_to_checkpoint]
"""
import os
import sys
import time

import torch

import _common
from core import config
from core.tracknet import load_checkpoint
from core.tracknet_model import TrackNet, WIDTH, HEIGHT, in_dim_for

DEFAULT_CKPT = config.TRACKNET_WEIGHTS_PATH


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")

    t0 = time.perf_counter()
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    print(f"Checkpoint loaded in {time.perf_counter() - t0:.2f}s")

    params = ckpt["param_dict"]
    seq_len = params["seq_len"]
    bg_mode = params["bg_mode"]
    in_dim = in_dim_for(seq_len, bg_mode)
    print(f"seq_len={seq_len}, bg_mode={bg_mode!r}, in_dim={in_dim}, out_dim={seq_len}")
    print(f"Other param_dict keys: {sorted(params.keys())}")

    model = TrackNet(in_dim=in_dim, out_dim=seq_len).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built OK. {n_params / 1e6:.1f}M parameters.")

    # FP16 measured unsafe (non-finite output) AND slower than FP32 on this project's
    # reference GPU (T550) -- see core/tracknet.py's use_half docstring: benchmark FP32.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    x = torch.rand(1, in_dim, HEIGHT, WIDTH, device=device, dtype=torch.float32)

    # warm-up (cudnn autotune + CUDA context / kernel JIT)
    with torch.no_grad():
        for _ in range(5):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_iters = 30
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            y = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_batch = 1000 * elapsed / n_iters
    ms_per_frame_nonoverlap = ms_per_batch / seq_len  # one forward call covers seq_len frames
    print(f"Output shape: {tuple(y.shape)}, dtype {y.dtype}, range [{y.min().item():.3f}, {y.max().item():.3f}]")
    print(f"Forward pass: {ms_per_batch:.1f} ms/batch (batch = {seq_len} frames)")
    print(f"  -> nonoverlap mode: {ms_per_frame_nonoverlap:.2f} ms/frame amortized "
          f"({1000 / ms_per_frame_nonoverlap:.1f} FPS)")
    print(f"  -> dense/sliding mode (no batching benefit): {ms_per_batch:.2f} ms/frame "
          f"({1000 / ms_per_batch:.1f} FPS)")

    if device.type == "cuda":
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")

    if ms_per_frame_nonoverlap > 200:
        print("\nGATE FAILED: >200 ms/frame even in nonoverlap mode. Reconsider before integrating.")
        sys.exit(1)
    print("\nGATE PASSED.")


if __name__ == "__main__":
    _common.setup_logging()
    main()
