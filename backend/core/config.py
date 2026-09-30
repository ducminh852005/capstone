"""
Central configuration for all tunable parameters in the badminton analysis pipeline.

Every module in core/ and every script in scripts/ should import its defaults
from here instead of hardcoding magic numbers. Grouping all knobs in one file
makes it easy to tune the system for a new camera angle, court, or lighting
condition without hunting through dozens of source files.

Physical court dimensions (meters) live in court_model.py because they are
constants of the sport, not tuning parameters.

Rules for this file:
- One definition per name, each followed by a docstring saying what it does and its unit.
- No imports from other core modules (config must stay a leaf).
- Filesystem locations are derived from this file's location, never from the CWD.
"""
from pathlib import Path

# =====================================================================
#  PATHS  (derived from this file, independent of the working directory)
# =====================================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
"""Absolute path of the backend/ directory."""

REPO_ROOT = BACKEND_DIR.parent
"""Absolute path of the repository root."""

DATA_DIR = REPO_ROOT / "data"
"""Videos, calibration and metadata."""

MODELS_DIR = BACKEND_DIR / "models"
"""Model weights (git-ignored; download instructions in QUICK_START.md)."""

CALIBRATION_PATH = DATA_DIR / "calibration.json"
"""Court calibration written by scripts/calibrate_court.py."""

METADATA_PATH = DATA_DIR / "metadata.csv"
"""Per-video metadata (match type etc.)."""

BENCHMARK_DIR = DATA_DIR / "benchmarks"
"""Output directory of scripts/benchmark_pipeline.py."""

TRACKNET_WEIGHTS_PATH = MODELS_DIR / "TrackNet_best.pt"
"""TrackNetV3 checkpoint."""

PLAYER_YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
"""YOLOv8 person-detection weights."""

# =====================================================================
#  SHUTTLE DETECTION  (shuttle_tracker.py, tracknet.py)
# =====================================================================

# --- Kalman tracker ---
KALMAN_CHI2_GATE = 9.21
"""Mahalanobis gate: chi-squared critical value for 2 degrees of freedom at 99%"""

KALMAN_MIN_GATE_PX = 15.0
"""Minimum Euclidean gate (px) — always allows candidates this close even if
the Mahalanobis test rejects them (useful when the filter's covariance
has not converged yet, e.g. right after track initiation)."""

KALMAN_MAX_SPEED_RATIO = 3.0
"""Physics check: max allowed instantaneous speed jump ratio. If a candidate causes
a higher speed jump, it is rejected (forces track to break on racket hits)."""

KALMAN_MIN_COS_ANGLE = 0.0
"""Physics check: min allowed cosine of angle change (0.0 = 90 deg). Sharp turns are rejected."""

MIN_BODY_SPEED_PX = 8.0
"""Tracks slower than this (px/frame) ignore candidates that overlap a player
body — prevents locking onto a limb instead of the shuttle."""

# --- Track lifecycle ---
MAX_COAST_CV = 8
"""Maximum consecutive frames without a matching candidate before a track ends.
CV backend sees every frame, so a short coast is fine; TrackNet's batched
inference skips frames, so it needs more headroom (see TRACKNET_MIN_COAST)."""

TRACKNET_BATCH_STRIDE = 5
"""New frames between two TrackNet forward passes, i.e. one shuttle detection every this many
frames (the model always sees a full seq_len = 8 frame window). Smaller = finer detection
resolution but proportionally more GPU work (each pass costs ~100 ms on the reference GPU):
5 means 1.6x the passes of the 8-frame (non-overlapping) mode. 0/None = seq_len."""

TRACKNET_MIN_COAST = 12
"""Minimum coast (frames) for TrackNet backends."""

TRACKNET_COAST_MARGIN = 4
"""Extra frames added to batch_stride when sizing TrackNet coast:
max_coast = max(TRACKNET_MIN_COAST, batch_stride + TRACKNET_COAST_MARGIN)."""

MAX_TRACK_LEN = 180
"""Maximum length of a single track (frames). Tracks longer than this are
assumed to be locked on clutter (a ceiling light, a logo) and are killed."""

MAX_CANDIDATES = 150
"""Maximum number of raw candidates per frame before the frame is considered
too noisy to use (camera shake, light flicker)."""

KALMAN_MEAS_STD_PX = 2.0
"""Measurement noise std-dev (px) of a detected shuttle position."""

KALMAN_ACCEL_NOISE = 3.0
"""Process noise (jerk) scale of the constant-acceleration model."""

KALMAN_INIT_STD = (4.0, 4.0, 256.0, 256.0, 25.0, 25.0)
"""Initial covariance diagonal (x, y, vx, vy, ax, ay). The large velocity
variance keeps the Mahalanobis gate wide enough to catch a fast shuttle on the
very next batch (stride frames later) when a track starts from a single point."""

PHYSICS_MIN_SPEED_PX = 2.0
"""Physics filter only runs when the current track speed (px/frame) exceeds this;
below it the direction/speed of the track is not reliable."""

PHYSICS_STOP_SPEED_PX = 2.0
"""When the physics filter breaks a track and the shuttle leaves the break slower than this
(px/frame), it came to rest (landed or resting on the floor): the track restarts but it is
not a racket hit and not a new flight."""

PHYSICS_MAX_DECEL_RATIO = 10.0
"""Reject a candidate if it would slow the shuttle by more than this factor.
Air drag can cost 3-4x in one batch, so 10x means it was not the same shuttle."""

PHYSICS_LOOKBACK_FRAMES = 15
"""How far back in the trajectory to look for the last valid detection."""

PHYSICS_MIN_SPEED_AFTER_PX = 0.1
"""Floor on the candidate speed (px/frame) to avoid division by zero in ratios."""

BODY_TOP_FRAC = 1 / 3
"""Fraction of a player's box (from the top) where a candidate is still allowed:
the racket arm and hit zone; below it, candidates are legs/torso."""

# --- Gap filling hit detection (ShuttleTrajectoryProcessor._likely_hit) ---
HIT_ANGLE_COS_THRESH = 0.3
"""Velocity direction across a gap with cosine below this counts as a hit."""

HIT_SPEED_RATIO_THRESH = 2.5
"""Speed change across a gap by more than this factor counts as a hit."""

HIT_MIN_SPEED_PX = 1.0
"""Below this speed (px/frame) direction/speed cannot be read; defer to the fit RMS."""

# --- CV backend (background subtraction) ---
CV_BG_METHOD = "knn"
"""Background subtractor: "knn" or "mog2"."""

CV_MIN_BLOB_AREA = 2
"""Smallest blob area (px) accepted as a shuttle candidate."""

CV_MAX_BLOB_AREA = 500
"""Largest raw blob area (px) accepted as a shuttle candidate."""

CV_MAX_MERGED_AREA = 900
"""Components larger than this after dilation are bodies, not a shuttle."""

CV_MAX_ELONGATION = 6.0
"""Max bounding-box aspect ratio of a candidate blob."""

CV_BG_HISTORY = 50
"""Background subtractor history length (frames)."""

CV_KNN_DIST2_THRESHOLD = 400.0
"""KNN squared-distance threshold for the foreground decision."""

CV_MOG2_VAR_THRESHOLD = 16.0
"""MOG2 variance threshold for the foreground decision."""

CV_MAX_INIT_PRODUCT = 2_000_000
"""Cap on candidates_A * candidates_B * candidates_C in the 3-frame init search."""

CV_INIT_MAX_RESIDUAL_PX = 12.0
"""3-frame init: max residual (px)"""

CV_INIT_SPEED_RANGE = (5.0, 150.0)
"""3-frame init: speed range (px/frame)"""

# --- TrackNet backend ---
TRACKNET_INIT_MIN_CONFIDENCE = 0.6
"""Minimum detection confidence to start a new track from a single point."""

TRACKNET_CONF_THRESHOLD = 0.5
"""Heatmap threshold for converting the sigmoid output to binary candidates."""

TRACKNET_BG_FRAMES = 60
"""Number of frames used to build the background image at startup."""

# --- Trajectory gap filling (ShuttleTrajectoryProcessor) ---
SHUTTLE_MAX_GAP_FRAMES = 15
"""Longest gap (frames) that will be interpolated; longer gaps are left empty."""

SHUTTLE_FIT_WINDOW = 8
"""Points on each side of a gap used for the parabola fit."""

SHUTTLE_MAX_FIT_RMS_PX = 4.0
"""If the parabola residual exceeds this, fall back to linear interpolation."""

# --- Gap filling between detections (core/gap_fill.py) ---
SHUTTLE_FILL_LINK_STRIDES = 1.5
"""Two detections belong to the same chain when they are at most this many detection
strides apart (stride = TrackNet batch_stride, 1 for the CV backend)."""

SHUTTLE_FILL_MIN_CHAIN_LEN = 2
"""A chain must hold at least this many detections on both sides of a gap for the gap to
be bridged: fewer cannot tell which way the shuttle was flying."""

SHUTTLE_FILL_MAX_GAP_FRAMES = 48
"""Longest gap (frames) bridged between two chains (48 = 6 TrackNet strides)."""

SHUTTLE_FILL_STRIDE_FIT_POINTS = 2
"""Detections taken from each side of a gap inside a chain for the inertia + gravity fit.
On real footage (leave-one-out on tran04) 2 per side predicts a held-out detection to a median
of 4.5 px; a wider window spans several flights and stops fitting one parabola."""

SHUTTLE_FILL_BRIDGE_FIT_POINTS = 3
"""Detections taken from each side of a gap between two chains (fewer if the chain is shorter)."""

SHUTTLE_FILL_MAX_RMS_PX = 6.0
"""A gap is only filled when one parabola (constant acceleration) explains the detections
on both sides this well (px RMS). A racket hit inside the gap breaks the fit."""

SHUTTLE_FILL_TRUSTED_GAP_FRAMES = 32
"""Prediction error grows with the gap (median 4 px over 16 frames, 8 px over 24, p90 ~90 px
over 32). Beyond this many frames the RMS limit shrinks in proportion to the gap."""


# =====================================================================
#  LANDING / UMPIRE  (umpire.py)
# =====================================================================

UMPIRE_EDGE_MARGIN_PX = 25
"""Pixels from the ROI edge: a shuttle ending near the edge probably left the
frame rather than landing."""

UMPIRE_REST_SPEED_PX = 2.0
"""A shuttle moving slower than this for rest_frames consecutive frames is
considered to have come to rest (confirming a landing)."""

UMPIRE_REST_SPEED_FLOOR_PX = 1.5
"""Lower bound (px/frame) of the perspective-scaled rest speed. Far from the camera the scaled
threshold drops below 1 px/frame, which is under TrackNet's own localisation noise (a few px
between detections a stride apart), so a resting shuttle would never look at rest."""

UMPIRE_ALLOW_RESTING = True
"""Call a landing when a shuttle that was falling is lost and then reappears lying still on
the floor (Call.method == "resting"): the impact itself was not seen, the call uses the first
resting position. Set False to only call impacts that were seen."""

UMPIRE_REST_RUN_POINTS = 3
"""Consecutive detections that must be still to count as lying on the floor."""

UMPIRE_REST_MAX_GAP_FRAMES = 90
"""The resting shuttle must reappear within this many frames of the last falling detection."""

UMPIRE_REST_Y_SLACK_PX = 20
"""A shuttle that fell cannot come to rest more than this many px HIGHER in the image than
where it was last seen falling."""

UMPIRE_REST_FRAMES = 2
"""How many consecutive slow frames confirm a landing."""

UMPIRE_CLOSE_CALL_M = 0.10
"""Signed distance (meters) below which a call is flagged as "close"."""

UMPIRE_MIN_FLIGHT_LEN = 5
"""Minimum track length (detections) for a flight to be judged."""

UMPIRE_MIN_DESCENT_PX = 80
"""Minimum vertical descent (px) from the flight's apex before looking for
an impact — prevents false positives from gentle tosses."""

UMPIRE_SETTLE_SEARCH_FRAMES = 30
"""Frames after a candidate impact to look for the shuttle calming down."""

UMPIRE_FLOOR_BUFFER_M = 3.0
"""Margin (meters) around the full court in which a landing is still accepted.
Anything further out is a wall/ceiling hit, not a court landing. The region
itself is built from the court dimensions in umpire.default_floor_region()."""

UMPIRE_PERSPECTIVE_PROBE_PX = 10
"""Vertical image offset (px) used to measure the local metres-per-pixel scale
when converting the rest-speed threshold to perspective-corrected values."""

UMPIRE_ALLOW_EXTRAPOLATION = False
"""Call a landing for a flight that ends while the shuttle is still falling, by
extrapolating its last velocity (Call.method == "lost"). Off: a shuttle in the air does
not map to a floor point, and on real footage the extrapolation only ever ran to the edge
of the accepted floor region, producing spurious OUT calls."""

UMPIRE_EXTRAPOLATE_MAX_FRAMES = 30
"""When a falling shuttle is lost before touching the floor, extrapolate its
last velocity for at most this many frames to find where it would land."""

UMPIRE_EXTRAPOLATE_WINDOW = 4
"""Number of trailing detections used to estimate the final velocity."""

UMPIRE_EXTRAPOLATE_MIN_POINTS = 3
"""Minimum detections in a flight to allow extrapolated landing."""

UMPIRE_MIN_METERS_PER_PX = 1e-6
"""Floor on the local metres-per-pixel scale (avoids division by zero)."""


# =====================================================================
#  PLAYER TRACKING  (player_tracker.py)
# =====================================================================

PLAYER_YOLO_CONF = 0.4
"""YOLO person detection confidence threshold."""

PLAYER_YOLO_IMGSZ = 640
"""YOLO input image size (pixels)."""

PLAYER_POSE_EVERY = 5
"""Run MediaPipe pose on the selected player every N update() calls."""

PLAYER_YOLO_EVERY = 1
"""Run YOLO+ByteTrack every N process() calls; in between, reuse last boxes."""

# --- Player selector ---
SELECTOR_WINDOW_S = 10.0
"""Sliding window (seconds) for computing the in-court score of a track."""

SELECTOR_MIN_TRACK_S = 1.0
"""Minimum seconds a track must exist before being normalised at full weight."""

SELECTOR_MIN_SCORE = 0.5
"""Minimum in-court fraction to be considered a player candidate."""

SELECTOR_SWITCH_MARGIN = 0.3
"""A challenger must beat the current player's score by this margin to take over."""

SELECTOR_REID_WINDOW_S = 3.0
"""After the player's track disappears, wait this long for a re-id match."""

SELECTOR_REID_DIST_M = 2.0
"""Maximum distance (meters) for re-identification of a lost player."""


# =====================================================================
#  TRAJECTORY CLEANING  (trajectory.py)
# =====================================================================

TRAJECTORY_MAX_SPEED = 7.0
"""Maximum plausible player speed (m/s); faster jumps are rejected as outliers."""

TRAJECTORY_MAX_GAP_S = 0.5
"""Gaps shorter than this (seconds) are linearly interpolated."""

TRAJECTORY_SMOOTH_WINDOW_S = 0.4
"""Savitzky-Golay smoothing window (seconds)."""

TRAJECTORY_SMOOTH_POLYORDER = 2
"""Smoothing polynomial order."""


# =====================================================================
#  SMASH DETECTION  (core/smash.py)
# =====================================================================

SMASH_SPEED_THRESHOLD = 12.0
"""Minimum speed (px/frame) to classify a hit as a smash.
The two detection points used to measure speed are one TrackNet stride
(TRACKNET_BATCH_STRIDE) apart, so the per-frame speed is the displacement
divided by that. A real smash covers ~12+ px/frame (~100 px in 8 frames)."""

SMASH_REQUIRE_PHYSICS_HIT = True
"""Only a track that restarted because the physics filter broke the previous one (a racket
hit) can be a smash. A track that starts from nothing is measured but not counted: it may be
a shuttle first seen while falling fast (not a hit)."""

SMASH_MIN_Y = 50.0
"""Ignore hits whose image y (px) is above this: the shuttle is out of the play area."""

SMASH_MIN_ANGLE = 10.0
"""Minimum downward angle (degrees, 0 = horizontal) of a smash's direction."""

SMASH_MAX_ANGLE = 170.0
"""Maximum angle (degrees) of a smash's direction; with SMASH_MIN_ANGLE it bounds
the accepted cone (0 = +x axis, 90 = straight down in image coordinates)."""


# =====================================================================
#  DISPLAY / UI  (scripts/)
# =====================================================================

DISPLAY_SIZE = (1080, 720)
"""Default display window size (width, height)."""

TRAJECTORY_TAIL_LENGTH = 60
"""Number of recent trajectory points to draw as the shuttle's tail."""

SMASH_DISPLAY_DURATION = 45
"""How many frames the "SMASH!" overlay stays visible."""

DEBUG_LOG_INTERVAL = 60
"""How often (every N frames) to print a debug heartbeat to the terminal."""

DEFAULT_VIDEO = str(DATA_DIR / "cfr" / "tran04_cam1.mp4")
"""Default video path when none is provided on the command line."""

PLAYER_DEMO_YOLO_CONF = 0.5
"""YOLO confidence used by the demo/benchmark scripts (stricter than PLAYER_YOLO_CONF)."""

UMPIRE_ALERT_FRAMES = 60
"""How many frames the IN/OUT call overlay stays visible in the demos."""

DEFAULT_BACKEND = "tracknet"
"""Default shuttle detection backend."""

# --- Minimap ---
MINIMAP_SCALE = 45
"""Pixels per meter on the 2D minimap."""

MINIMAP_SIZE = (360, 720)
"""Minimap canvas size (width, height) in pixels."""

MINIMAP_BG_COLOR = (0, 100, 0)
"""Minimap background color (BGR)."""

# --- Tactical board (demo_player_tracker.py) ---
BOARD_SCALE = 100
"""Pixels per meter on the tactical board."""
