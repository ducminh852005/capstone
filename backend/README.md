# Badminton AI Analysis System - Technical Documentation

This document serves as the primary technical handover for the Badminton AI Analysis System. It outlines the core algorithms, architectures, and running procedures for future developers to continue the work.

---

## 1. System Architecture

The system is modularized into independent computer vision pipelines that can be combined to form a complete Tactical Analysis and Auto-Umpiring product.

### Core Pipelines:
1. **Spatial Calibration**: Maps 2D video pixels to 3D physical world coordinates using Homography.
2. **Player Tracking (Tactical Board)**: YOLOv8 + ByteTrack object tracking projected onto a 2D minimap.
3. **Shuttlecock Tracking**: High-speed, lightweight classical CV (Background Subtraction) enhanced with a Physics-based tracker.
4. **Auto Umpire (Hawk-Eye)**: State machine logic combining spatial calibration and shuttle trajectory to make IN/OUT decisions.

---

## 2. Core Modules & Algorithms

### A. Court Calibration (`core/court_calibration.py`)
- **Purpose**: Establishes the relationship between the camera angle and the physical court dimensions.
- **Mechanism**: The user interactively clicks 4 corners of the *Near Court*. The script uses `cv2.getPerspectiveTransform` to generate a Homography Matrix (`H`) and its inverse (`H_inv`).
- **Output**: Saves the matrix data to `data/calibration.json`.

### B. Physics-Based Shuttle Tracker (`core/shuttle_tracker.py`)
- **Isolation**: Uses `cv2.createBackgroundSubtractorKNN(history=50)` combined with morphological Top-Hat filtering to isolate moving bright objects.
- **Physics Tracker**: Instead of blindly tracking the closest object, it simulates a **Parabolic Trajectory**:
  1. **Velocity Calculation**: `vx, vy` are calculated from the last 2 frames.
  2. **Gravity Bias**: A downward pull (`vy += 2`) is applied.
  3. **Scoring System**: Objects are scored based on their distance to the *predicted momentum path* (70% weight) and the *last known position* (30% weight).
  4. **Result**: This heavily penalizes false positives (like moving white shoes) and strictly locks onto objects following a smooth, high-speed ballistic curve. Bounces and racket hits are caught by the 30% spatial proximity fallback.

### C. Auto Umpire (`scripts/test_auto_umpire.py`)
- **Landing Detection**: Triggers a "Landing Event" if the shuttle is lost for `> 15 frames` (due to stopping on the floor and being absorbed by the Background Subtractor).
- **Anti-Pickup Sensor (Height Filter)**: Tracks the `min_y` (highest physical point) during a rally. If the shuttle never exceeds the virtual net height (Net base Y - 50 pixels), the system ignores the landing event. This prevents scoring when players pick up or roll the shuttle.
- **Dynamic ROI Cropping**: Crops 15% off the left/right sides to blind the system to audience members and off-court noise.
- **Scoring**: Applies `H_inv` to the landing pixel to get physical meters. Checks if `0 <= X <= 6.7` (Near Court) and `0 <= Y <= 6.1`.

### D. Tactical Heatmap (`scripts/test_player_tracker.py`)
- YOLOv8 detects players, ByteTrack maintains IDs.
- Extracts the "foot point" (bottom-center of the bounding box).
- Uses `H_inv` to map the foot point onto a static 2D green minimap, drawing translucent red circles to form a movement heatmap.

### E. TrackNet Shuttle Detector (`core/tracknet.py`) -- FP16 correctness/perf fix
The default `--shuttle-backend` is `tracknet` (a CNN candidate source feeding the same Physics-Based Tracker as section B), not the classical CV backend described there.

**Bug found and fixed: never run TrackNet in FP16 on this project's reference GPU (T550 Laptop, 4GB).** `TrackNetCandidateSource` used to default `use_half = (device.type == "cuda")`, i.e. FP16 automatically whenever CUDA was available. Measured on this GPU with real video frames: **FP16 overflowed to non-finite output on ~80% of forward passes** (silently caught by an existing guard that logs a warning and treats the batch as "no detection" instead of crashing), collapsing real shuttle detection from ~3% of frames to effectively 0% -- with no error, no crash, just an empty trajectory. This surfaced after upgrading torch to a CUDA build (`cu121`) to actually use the GPU (see git history / conversation log around the initial "torch runs on CPU" fix) -- the same checkpoint had apparently been more stable under whatever torch/cuDNN build was in use before.

Forcing FP32 on the same GPU didn't just fix correctness, it was also **~3.5x faster** (103 ms/window vs. FP16's 364 ms/window) -- this GPU has no real FP16 throughput advantage, so FP16 was strictly worse on both counts, not a speed/accuracy tradeoff. `use_half` now defaults to `False` in `TrackNetCandidateSource.__init__` (`core/tracknet.py`); only pass `use_half=True` after verifying finite output on your specific GPU/torch/cuDNN combination.

Full-pipeline effect, measured on `data/cfr/tran04_cam1.mp4`, 1800 frames (`data/benchmarks/tracknet_fp32fix.json` vs. the broken-FP16 `tracknet.json`):

| | FP16 (broken, old default) | FP32 (fixed, new default) |
|---|---|---|
| wall_fps | 9.67 | **15.97** |
| shuttle ms/frame | 55.11 | **21.25** |
| pct_frames_detected | 3.2% | **3.6%** |
| non-finite batches | frequent (see above) | none |

Cross-check with `scripts/bench_tracknet_forward.py` (raw forward-pass latency, no video/CV overhead): now defaults to `use_half = False` to match, so it reports the same ~103 ms/window number instead of the old (fast-looking but broken) 364 ms/window FP16 figure -- treat any future FP16 timing on a *different* GPU as a correctness question first, a speed question second.

---

## 3. Workflow & Usage Instructions

Whenever you switch to a new camera angle or video (e.g., from `tran04` to `tran01`), you MUST recalibrate the spatial matrix.

**Step 1: Extract a sample frame from the new video**
```bash
# Example command using ffmpeg or python script
```

**Step 2: Run Calibration (Interactive)**
```bash
python scripts/test_calibration.py ../data/frames/tran01_frame.jpg
# Click the 4 corners of the Near Court: Bottom-Left, Bottom-Right, Net-Left, Net-Right. Press SPACE.
```

**Step 3: Run the Auto Umpire**
```bash
python scripts/test_auto_umpire.py ../data/cfr/tran01_cam1.mp4
```

**Step 4: Run Player Heatmap**
```bash
python scripts/test_player_tracker.py ../data/cfr/tran01_cam1.mp4
```

---

## 4. Future Development Roadmap

For the next developer taking over, here are the immediate areas for improvement:
1. **Full-Court Calibration**: Currently, only the Near Court is accurately mapped. Implement a 6-point or 8-point calibration system to map the Far Court without perspective distortion.
2. **Hit Detection System**: Implement audio analysis (racket *clack* sound) or pose-estimation heuristic (wrist acceleration) to detect exactly *when* and *who* hit the shuttle. This is required to accurately assign points when a shuttle lands OUT.
3. **Multi-Camera Stitching**: Combine tracking data from multiple camera angles to resolve occlusions.
