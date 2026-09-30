# Badminton AI Analysis System - Technical Documentation

This document serves as the primary technical handover for the Badminton AI Analysis System. It outlines the core algorithms, architectures, and running procedures for future developers to continue the work.

---

## 1. System Architecture

The system is modularized into independent computer vision pipelines that can be combined to form a complete Tactical Analysis and Auto-Umpiring product.

### Core Pipelines:
1. **Spatial Calibration**: Maps 2D video pixels to 3D physical world coordinates using Homography.
2. **Player Tracking (Tactical Board)**: YOLOv8 + ByteTrack object tracking projected onto a 2D minimap.
3. **Shuttlecock Tracking**: A candidate generator (TrackNetV3 by default, classical background subtraction as a fallback) feeding a constant-acceleration Kalman tracker with a physics filter.
4. **Auto Umpire (Hawk-Eye)**: Landing detection on shuttle tracks combined with spatial calibration to make IN/OUT decisions.
5. **Smash Detection**: Speed and direction of the shuttle right after each racket hit (`core/smash.py`).

Development rules (coordinate systems, units, config, paths, testing, style) are in [`../CLAUDE.md`](../CLAUDE.md). Read it before changing code.

---

## 2. Core Modules & Algorithms

### A. Court Calibration (`core/court_calibration.py`)
- **Purpose**: Establishes the relationship between the camera angle and the physical court dimensions.
- **Mechanism**: The user interactively clicks 4 corners of the *Near Court* (`scripts/calibrate_court.py`). `court_model.homography_from_points` fits the Homography Matrix (`H`, world to image) with `cv2.findHomography`, and its inverse `H_inv` maps image to world. Optionally (key `R`) `CourtCalibrator.refine_homography` snaps it to the painted lines of a clean-background image.
- **Output**: Saves the corner points (and the refined `H`) to `data/calibration.json`, which `court_model.load_calibration()` reads.

### B. Kalman Shuttle Tracker (`core/shuttle_tracker.py`)
- **Candidates**: a pluggable source returns candidate points per frame: `TrackNetCandidateSource` (default, section E) or `CVCandidateSource` (KNN/MOG2 background subtraction + top-hat filtering).
- **Tracker**: `ConstantAccelerationKalman` (state `x, y, vx, vy, ax, ay`, dt = 1 frame). Each frame a candidate is accepted if it lies inside the Mahalanobis gate (`KALMAN_CHI2_GATE`) or within `min_gate_px` (scaled by the source's `batch_stride`). Candidates on a player's legs/torso are ignored unless the track is fast.
- **Physics filter** (`ShuttleDetector._physics_reject`): a candidate inside the gate that would speed the shuttle up by more than `max_speed_ratio`, slow it by more than `PHYSICS_MAX_DECEL_RATIO`, or turn it by more than `acos(min_cos_angle)` does not continue the flight. The reference velocity is the one measured between the last two detections of the current track (`_chord_velocity`), not the Kalman velocity: with detections a stride apart the Kalman velocity is unreliable (it pointed the wrong way on a shuttle falling to the floor, which made the filter treat the landing as a racket hit and split the flight there). Air drag alone (3-4x slowdown per batch) is accepted. If another in-gate candidate does fit the flight, the unphysical one is a distractor and the track goes on with the good one. If none fits, the track is broken and restarts **on the same call** at the unphysical candidate (trusted without `init_min_confidence`).
- **Flights and hits**: `flight_id` counts flights (it changes on every hit, where `track_active` shows no edge), and `start_source` says how the current track started: `"physics"` (a racket hit: the shuttle left the break at speed), `"stop"` (the shuttle came to rest, `PHYSICS_STOP_SPEED_PX`: same flight, `flight_id` unchanged) or `"new"` (nothing to continue).
- **Filling frames without a detection** (`core/gap_fill.py`, `ShuttleDetector.fill_gaps()`): TrackNet only reports the shuttle every `batch_stride` frames, and sometimes misses more. The frames in between are estimated afterwards from the detections around them, one parabola per image axis (inertia from the velocity on both sides, plus gravity). Detections form chains (at most 1.5 strides apart, same flight). Two gaps are filled: *stride* (inside a chain) and *bridge* (between two chains that each have at least `SHUTTLE_FILL_MIN_CHAIN_LEN` = 2 detections, up to `SHUTTLE_FILL_MAX_GAP_FRAMES`, and only if the next chain did not start at a racket hit). A gap is only filled when one parabola fits the detections on both sides (`SHUTTLE_FILL_MAX_RMS_PX`, tighter for longer gaps), so a hit inside the gap leaves it empty. The court calibration is a sanity check (`gap_fill.court_constraint`): inside the ROI and not below the floor; it cannot tell the shuttle's height. Estimates are for drawing and analysis (drawn in a different color by `demo_trajectory.py`) and are never fed back into the tracker or the umpire. Measured on tran04 by holding out real detections: median error 4.5 px over one stride, ~8 px over 24 frames, and much worse beyond ~30 frames.
- **Track lifecycle**: a track survives `max_coast` frames without a match (`MAX_COAST_CV` for CV; `max(TRACKNET_MIN_COAST, batch_stride + TRACKNET_COAST_MARGIN)` for TrackNet) and at most `MAX_TRACK_LEN` frames.

### C. Auto Umpire (`core/umpire.py`, demo: `scripts/demo_auto_umpire.py`)
- **Flights**: `RallyUmpire.update()` buffers the detections of one flight. A flight ends when the track dies or, if the caller passes `flight_id=detector.flight_id`, when a hit starts the next flight. Always pass it: a hit restarts the track inside one `detect()` call, so without `flight_id` several hits merge into one flight.
- **Landing detection** (`landing_point`): the first sharp break from free fall: the vertical speed, after exceeding the rest speed while falling, drops to the rest speed or less, at a point at least `min_descent` px below the flight's highest point. It is accepted if the shuttle then calms down (`rest_frames` slow steps within `settle_search_frames`, `Call.method == "contact"`) or if the track ends before that can be told (`"contact_unconfirmed"`). The rest-speed threshold is scaled per point by perspective (a pixel means fewer metres near the camera).
- **Hits, bounces and look-ahead**: a shuttle that touches the floor reverses its vertical velocity, which the tracker cannot tell from a racket hit, so a flight that ends at a `flight_id` change is not discarded. It is judged once the next `rest_frames + 1` points have shown whether the shuttle calmed down (a landing, contact at or before the boundary) or flew on (a hit).
- **Resting shuttle** (`UMPIRE_ALLOW_RESTING`, `Call.method == "resting"`): TrackNet often loses the shuttle during the last part of the fall and picks it up again lying still on the floor. If a flight ended while the shuttle was still falling (after a real descent), and within `UMPIRE_REST_MAX_GAP_FRAMES` a flight starts with `UMPIRE_REST_RUN_POINTS` still detections (not on a person, not at the frame edge, on the floor, not higher in the image than where the fall was last seen), the call is made at the first resting position. The impact itself was not seen, so this is less precise than a `contact` call. The rest speed has a floor (`UMPIRE_REST_SPEED_FLOOR_PX`) because TrackNet's localisation noise is a few px between detections.
- **Lost shuttle** (`UMPIRE_ALLOW_EXTRAPOLATION`, off): a flight that ends while the shuttle is still falling can be extrapolated to the floor (`method == "lost"`). Off by default: an airborne pixel does not map to a floor point, and on real footage the extrapolation only ran to the edge of the accepted floor region and produced spurious OUT calls.
- **Anti-pickup filter**: a flight that never rose above the net threshold (`net_top_y`) is ignored (players picking up or rolling the shuttle).
- **Invalid contacts**: a contact on a person, near the ROI edge, or outside the floor region (court plus `UMPIRE_FLOOR_BUFFER_M`) is not a landing.
- **Scoring** (`judge`): `H_inv` maps the contact to metres; the x position against `NET_X` picks the half; lines count as IN (half a line width is added). `close_call` is flagged when `|margin| < max(close_call_m, uncertainty_m)`, where `Call.uncertainty_m` is the court length one image pixel spans at the landing (about 5 cm at the net but 13 cm at the far baseline: far-half calls are inherently coarse, and the far half is an extrapolation of the near-half calibration). Match type (singles/doubles) comes from `data/metadata.csv`.

### D. Smash Detection (`core/smash.py`, demo: `scripts/demo_smash.py`)
- A hit is a track that restarted because the physics filter broke the previous one (`start_source == "physics"`). `SmashDetector` measures the speed (px/frame) and direction from the track start to its first real detection (the first detection after the hit, up to one TrackNet stride after the contact), and flags a smash when speed exceeds `SMASH_SPEED_THRESHOLD`, the hit is below `SMASH_MIN_Y`, and the angle is within `SMASH_MIN_ANGLE..SMASH_MAX_ANGLE`. Tracks that start from nothing are measured too but do not count unless `SMASH_REQUIRE_PHYSICS_HIT` is off.

### D3. Event evaluation (`scripts/eval_events.py`)
There is no automatic ground truth, so hit / smash / landing detection is scored against hand labels in `data/events/<video>.events.json` (format in `core/event_eval.py`; `tran04_cam1.events.json` is a template). Fill in the frames, then:
```bash
python scripts/eval_events.py ../data/cfr/tran04_cam1.mp4 --tol 12 --out mytest
```
It prints precision / recall per event type (landings also per `Call.method`, plus IN/OUT agreement) and lists the false-positive and missed frames. Compare runs before and after changing a threshold.

### D2. Tactical Heatmap (`scripts/demo_player_tracker.py`)
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

### F. ONNX Runtime backend for TrackNet (`core/tracknet_onnx.py`) -- correct and isolated-faster, but net negative in the full pipeline
Same checkpoint, same weights, same FP32 math as `TrackNetCandidateSource` (`core/tracknet.py`) -- only the runtime executing the forward pass differs. Motivation: PyTorch eager mode dispatches each conv/batchnorm/upsample block as its own Python call; ONNX Runtime runs the whole graph as one fused C++ call, which should matter more at batch=1 (this pipeline's case) where per-call Python overhead is a real fraction of total time.

`backend="tracknet-onnx"` on `ShuttleDetector` (`TrackNetONNXCandidateSource`, subclasses `TrackNetCandidateSource` and only overrides the parts that touch PyTorch: model construction/session, `_ensure_background`, `_run_batch`). Export first: `python scripts/export_tracknet_onnx.py` (also self-checks the export against PyTorch output, `<1e-6` max abs diff on a random input -- trust but verify). Needs `onnxruntime-gpu==1.19.2` pinned in `requirements.txt`: the latest (1.30+) requires CUDA 13.x/cuDNN 9 system libs and *silently* falls back to `CPUExecutionProvider` with no error if they're missing -- caught this by explicitly checking `session.get_providers()` after construction, not by trusting a clean run. `nvidia-cudnn-cu12`/`nvidia-cublas-cu12` (pip wheels) supply the CUDA/cuDNN DLLs onnxruntime needs, same idea as torch's own bundled CUDA libs.

**Isolated forward-pass latency (same process, no video/YOLO/pose), 30 iterations:** ONNX Runtime is genuinely faster -- 94.4 ms/batch vs. PyTorch's 107.0 ms/batch (~12%). This confirms the Python-dispatch-overhead theory is directionally correct.

**Full-pipeline result is the opposite**, measured back-to-back (same process load state) on `data/cfr/tran04_cam1.mp4`, 1800 frames (`data/benchmarks/tracknet_onnx.json` vs. `tracknet_fp32_recheck.json`):

| | PyTorch (tracknet) | ONNX Runtime (tracknet-onnx) |
|---|---|---|
| wall_fps | **17.57** | 11.52 |
| yolo ms/frame | 25.55 | 39.74 |
| shuttle ms/frame | 20.15 | 24.02 |
| pose ms/frame | 10.65 | 22.11 |
| pct_frames_detected / n_runs | 3.6% / 64 | 3.6% / 64 (identical -- correctness confirmed) |

Every stage got slower with the ONNX backend active -- including YOLO and MediaPipe pose, whose code didn't change at all between these two runs. Since the *isolated* ONNX forward pass is faster, this can't be the ONNX model itself; the working explanation is that **PyTorch and ONNX Runtime each keep their own separate CUDA context/memory allocator** in the same process. `ShuttleDetector(backend="tracknet-onnx")` still runs YOLO through PyTorch (`core/player_tracker.py` is untouched) and TrackNet through ONNX Runtime -- every frame now switches between two independent CUDA contexts on a 4GB card that's already tight, and that switching cost outweighs ONNX's ~12% per-call advantage and then some, dragging down unrelated stages too. Not confirmed with a CUDA profiler, only inferred from this pattern (isolated win, full-pipeline loss spread across unrelated stages, correctness untouched) -- flagged as a hypothesis, not a proven mechanism.

**Conclusion: keep `backend="tracknet"` (PyTorch) as the default; `tracknet-onnx` is correct and available but not adopted.** It would plausibly need YOLO ported to ONNX Runtime too (removing the cross-framework context switching entirely) to realize the underlying per-call speedup -- untested, a bigger change than this session's scope. `data/benchmarks/tracknet_onnx.json` and `tracknet_fp32_recheck.json` are kept as the paired evidence.

---

## 3. Workflow & Usage Instructions

Whenever you switch to a new camera angle or video (e.g., from `tran04` to `tran01`), you MUST recalibrate the spatial matrix.

Commands run from `backend/` (paths are resolved from the file locations, so any working directory works).

**Step 1: Convert the raw video to constant frame rate (60 fps)**
```bash
python scripts/process_all_videos.py   # data/raw/*.mp4 -> data/cfr/*.mp4
```

**Step 2: Run Calibration (Interactive)**
```bash
python scripts/calibrate_court.py --video ../data/cfr/tran01_cam1.mp4
# Click/drag the 4 corners of the Near Court: Bottom-Left, Bottom-Right, Net-Left, Net-Right. R refines, SPACE saves.
```

**Step 3: Run the Auto Umpire**
```bash
python scripts/demo_auto_umpire.py ../data/cfr/tran01_cam1.mp4
```

**Step 4: Run Player Heatmap / Smash demo**
```bash
python scripts/demo_player_tracker.py ../data/cfr/tran01_cam1.mp4
python scripts/demo_smash.py ../data/cfr/tran01_cam1.mp4
```

**Tests**
```bash
python -m pytest        # runs backend/tests only (pytest.ini excludes scripts/)
```
`scripts/demo_*.py` are interactive OpenCV demos and are not tests. `scripts/debug_track_state.py` prints the tracker state frame by frame for a frame window.

---

## 4. Future Development Roadmap

For the next developer taking over, here are the immediate areas for improvement:
1. **Full-Court Calibration**: Currently, only the Near Court is accurately mapped. Implement a 6-point or 8-point calibration system to map the Far Court without perspective distortion.
2. **Hit Detection System**: Implement audio analysis (racket *clack* sound) or pose-estimation heuristic (wrist acceleration) to detect exactly *when* and *who* hit the shuttle. This is required to accurately assign points when a shuttle lands OUT.
3. **Multi-Camera Stitching**: Combine tracking data from multiple camera angles to resolve occlusions.
