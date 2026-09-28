"""
Exports the TrackNetV3 checkpoint (backend/TrackNet_best.pt) to ONNX, for the
"tracknet-onnx" ShuttleDetector backend (core/tracknet_onnx.py) -- see backend/README.md,
"ONNX Runtime backend", for why (Python-dispatch overhead per conv block, non-trivial at
batch=1, that PyTorch eager mode pays and a single fused ONNX Runtime graph doesn't).

Exports in FP32, matching TrackNetCandidateSource's default (see core/tracknet.py's
use_half docstring -- FP16 was measured unsafe and slower on this project's GPU).

Usage: python scripts/export_tracknet_onnx.py [path_to_checkpoint] [path_to_onnx_out]
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.tracknet_model import TrackNet, WIDTH, HEIGHT, in_dim_for

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "TrackNet_best.pt")


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(ckpt_path)[0] + ".onnx"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = ckpt["param_dict"]
    seq_len = int(params["seq_len"])
    bg_mode = params["bg_mode"]
    in_dim = in_dim_for(seq_len, bg_mode)
    print(f"seq_len={seq_len}, bg_mode={bg_mode!r}, in_dim={in_dim}")

    model = TrackNet(in_dim=in_dim, out_dim=seq_len)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.rand(1, in_dim, HEIGHT, WIDTH, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=17,
    )
    print(f"Exported -> {onnx_path}")

    import onnxruntime
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]

    if not np.isfinite(onnx_out).all():
        print("GATE FAILED: ONNX output contains non-finite values.")
        sys.exit(1)
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"Max abs diff vs PyTorch: {max_diff:.6f}")
    if max_diff > 1e-3:
        print("GATE FAILED: ONNX output diverges from PyTorch beyond tolerance (1e-3).")
        sys.exit(1)
    print("GATE PASSED. ONNX export matches PyTorch output.")


if __name__ == "__main__":
    main()
