# CầuLôngStats: Theoretical Foundations

This document explains the core computer vision, mathematical, and machine learning theories utilized in the CầuLôngStats project pipeline.

---

## 1. Video Preprocessing: CFR vs VFR
Smartphones typically record video in **Variable Frame Rate (VFR)** to optimize storage. However, in sports analytics, precise time calculations (like speed, distance per second) and frame-by-frame annotation require a strictly constant time delta between frames.
We use FFmpeg to re-encode the video into a **Constant Frame Rate (CFR)**. This guarantees that every frame exactly corresponds to `1/FPS` seconds, allowing accurate speed extraction and synchronization across multiple cameras.

## 2. Court Calibration & Perspective Transformation

### 2.1. Homography Matrix
To map 2D image coordinates (pixels from the camera) to 2D real-world coordinates (meters on the badminton court), we use a **Perspective Transformation** (Homography). 
Given 4 coplanar points on the court (e.g., the 4 corners of a half-court) and their known real-world measurements, we calculate a $3 \times 3$ Homography matrix $H$ using the **Direct Linear Transform (DLT)** combined with **RANSAC** (Random Sample Consensus) to reject outliers.
For any point $(x, y)$ in the image, the real-world coordinate $(X, Y)$ is calculated by multiplying the homogeneous image point by $H$.

### 2.2. Line Intersection via Homogeneous Coordinates
When the camera angle is too tight, some corners of the court may be out of frame. We recover these virtual corners by calculating the intersection of the visible boundary lines using projective geometry (homogeneous coordinates).
- A point is represented as $P = [x, y, 1]^T$.
- A line passing through two points $P_1$ and $P_2$ is computed by their cross product: $L = P_1 \times P_2$.
- The intersection of two lines $L_1$ and $L_2$ is found by their cross product: $P_{intersect} = L_1 \times L_2$.
Normalizing the resulting vector $[x', y', w']^T$ by $w'$ gives the Cartesian coordinate $(x'/w', y'/w')$.

### 2.3. Clean Background Extraction
To isolate the white court lines from moving players, we compute the **Temporal Median** of N frames evenly distributed across the video. Moving objects (players) appear at different pixels in different frames and are filtered out, leaving a "clean background".

### 2.4. Line Enhancement (Top-Hat Transform)
To accurately detect the white court lines regardless of lighting conditions, we apply a **Morphological Top-Hat Transform**. This operation highlights bright objects smaller than the structuring element (the white lines) against a dark background (the green court mat). It is followed by **Otsu's Thresholding**, which automatically determines the optimal threshold value to binarize the image.

## 3. Player Pose Estimation
While YOLOv8 detects the player's bounding box, the bottom center of a bounding box is not an accurate representation of a player's feet (especially during lunges or jumps).
We crop the player using the YOLO bounding box and pass it to **MediaPipe Pose Landmarker** (Heavy Model). MediaPipe outputs a 33-point 3D skeletal topology. We extract indices `29` (left heel), `30` (right heel), `31` (left foot index), and `32` (right foot index) to calculate the lowest foot point touching the ground, ensuring highly accurate distance tracking.

## 4. Shuttlecock Tracking & Trajectory Interpolation

### 4.1. TrackNet (Fast Object Tracking)
Shuttlecocks travel at extreme speeds (up to 400+ km/h), causing severe motion blur. Standard object detectors like YOLO fail because the shuttlecock becomes a faint, elongated streak.
**TrackNet** solves this by taking multiple consecutive frames as input (e.g., $t-1, t, t+1$). It learns the temporal dynamics and motion patterns of the shuttlecock rather than relying strictly on spatial appearance.

### 4.2. Trajectory Interpolation (Polynomial Curve Fitting)
Due to camera limitations or occlusions, the shuttlecock often flies out of the camera's Field of View (FOV). 
In physics, a projectile influenced by gravity roughly follows a parabolic path. When the shuttlecock goes out of frame, we have a sequence of frames with missing $(x, y)$ coordinates.
We apply **Polynomial Curve Fitting (Least Squares Method)** of degree 2 ($y = ax^2 + bx + c$) to the valid points immediately before the shuttlecock disappeared and immediately after it reappeared. By evaluating this polynomial over the missing time frames, we can accurately interpolate the unseen trajectory, generating a smooth "comet tail" animation for the viewer.
