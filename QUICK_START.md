# CầuLôngStats Quick Start Guide

This guide will help you quickly set up the CầuLôngStats project and start processing badminton videos.

## 1. Project Structure
The project is divided into three main components:
- `data/`: Contains all data assets (`raw`, `cfr`, `frames`, `labels`).
- `backend/`: FastAPI backend and core computer vision logic (YOLO, MediaPipe, OpenCV).
- `frontend/`: React + Vite frontend for data visualization.

## 2. Prerequisites
Ensure you have the following installed on your system:
- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg** (Must be added to system PATH). You can quickly install it on Windows using:
  ```cmd
  winget install ffmpeg
  ```
  *(Restart your terminal after installation)*

## 3. Installation

### Backend Setup
1. Navigate to the backend directory:
   ```cmd
   cd backend
   ```
2. Activate the virtual environment (it should already be created):
   ```cmd
   .\venv\Scripts\activate
   ```
3. (Optional) If dependencies are not installed, run:
   ```cmd
   pip install -r requirements.txt
   ```
4. Install PyTorch with CUDA (required: `core/tracknet.py` imports torch, so even the `cv` backend and
   the tests need it; it is not in requirements.txt on purpose, see the comment there):
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
5. Model weights live in `backend/models/` (git-ignored, download manually):
   - `TrackNet_best.pt` ([TrackNetV3](https://github.com/qaz812345/TrackNetV3), MIT license):
     ```cmd
     pip install gdown
     gdown "https://drive.google.com/uc?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA" -O ckpts.zip
     tar -xf ckpts.zip ckpts/TrackNet_best.pt
     move ckpts\TrackNet_best.pt models\TrackNet_best.pt
     ```
     Run these from `backend/`, then delete `ckpts.zip` and `ckpts/` (both are git-ignored).
   - `yolov8n.pt` (Ultralytics YOLOv8 nano person detector): download `yolov8n.pt` from
     https://github.com/ultralytics/assets/releases and put it in `backend/models/`.
   - `pose_landmarker_lite.task` and `pose_landmarker_heavy.task` (MediaPipe Pose Landmarker):
     download from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker and put them in `backend/models/`.
   Then sanity-check it loads and measure its speed on your GPU:
   ```cmd
   cd backend
   python scripts\bench_tracknet_forward.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```cmd
   cd frontend
   ```
2. Install dependencies:
   ```cmd
   npm install
   ```

## 4. Preparing Video Data

Before analysis, videos must be converted to a Constant Frame Rate (CFR) to maintain tracking and timing accuracy.

1. Place your raw badminton videos into `data/raw/`. 
   - Please follow the naming convention: `tranXX_camY.mp4` (e.g., `tran01_cam1.mp4`).
2. Update `data/metadata.csv` with the information of your new videos.

### Test Processing on a Single Video
If you want to test the FFmpeg conversion on just one video to ensure your setup is working:
```cmd
cd backend
.\venv\Scripts\activate
python -c "from core.video_processor import standardize_video_fps; standardize_video_fps(r'..\data\raw\tran04_cam1.mp4', r'..\data\cfr\tran04_cam1.mp4', 60)"
```

### Process All Videos
To automatically process all videos in the `data/raw` folder at once:
```cmd
cd backend
.\venv\Scripts\activate
python scripts\process_all_videos.py
```
This script will convert all videos in `data/raw` to 60fps CFR and save them to `data/cfr/`.

## 5. Testing Court Calibration

To test if the court corner detection and tracking homography work well on your camera angle:
1. Capture a sample frame from your video (e.g., `sample.jpg`).
2. Run the calibration test script:
   ```cmd
   cd backend
   .\venv\Scripts\activate
   python scripts\calibrate_court.py "path/to/your/sample.jpg"
   ```
   Or, better, build a clean background (median of 60 frames, players removed) from the video itself:
   ```cmd
   python scripts\calibrate_court.py --video ..\data\cfr\tran04_cam1.mp4
   ```
3. A window will open. Drag the 4 corners of the **near half-court** in this order:
   - Bottom-Left
   - Bottom-Right
   - Net-Left
   - Net-Right
4. The system draws the projected near half-court (all inner lines) over the image for validation.
5. Press **R** to refine the homography on the white court lines (the line-overlap score before/after is shown), then **SPACE** to save. The refined matrix is stored as `"H"` in `data/calibration.json` and is used by every script.

## 6. Tests and Benchmark

Unit tests (synthetic data, no video needed):
```cmd
cd backend
.\venv\Scripts\activate
python -m pytest -q
```
Only `backend/tests` is collected (`backend/pytest.ini`). The `scripts/demo_*.py` files are interactive
demos, not tests. Development rules are in `CLAUDE.md`.

Headless benchmark of the player + shuttle pipeline (per-stage ms/frame and quality proxies):
```cmd
python scripts\benchmark_pipeline.py ..\data\cfr\tran04_cam1.mp4 --frames 1800 --out ..\data\benchmarks\run.json
```
`ShuttleDetector` defaults to `backend="tracknet"` (needs `backend/models/TrackNet_best.pt`, see step 5 above);
pass `--shuttle-backend cv` to use the older classical background-subtraction detector instead
(no GPU/weights needed). Quality numbers are proxies (ID stability, track continuity, shuttle
jumps) until hand labels exist. Laptop GPUs throttle: compare runs made back to back, not runs
from different sessions.

## 7. Running the Application (Coming Soon)

Once the core processing pipeline is complete, you can start the application:

**Start the Backend:**
```cmd
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload
```
*API will be available at http://localhost:8000*

**Start the Frontend:**
```cmd
cd frontend
npm run dev
```
*Frontend will be available at http://localhost:5173*
