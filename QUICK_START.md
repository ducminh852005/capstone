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
   pip install mediapipe
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
   python scripts\test_calibration.py "path/to/your/sample.jpg"
   ```
3. A window will open. Click the 4 corners of the **near half-court** in this order:
   - Bottom-Left
   - Bottom-Right
   - Net-Left
   - Net-Right
4. The system will draw the projected court frame over the image for validation.

## 6. Running the Application (Coming Soon)

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
