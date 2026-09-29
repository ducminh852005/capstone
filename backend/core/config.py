"""
Central configuration for all tunable parameters in the badminton analysis pipeline.

Every module in core/ and every script in scripts/ should import its defaults
from here instead of hardcoding magic numbers. Grouping all knobs in one file
makes it easy to tune the system for a new camera angle, court, or lighting
condition without hunting through dozens of source files.

Physical court dimensions (meters) live in court_model.py because they are
constants of the sport, not tuning parameters.
"""

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

MIN_BODY_SPEED_PX = 8.0
"""Tracks slower than this (px/frame) ignore candidates that overlap a player
body — prevents locking onto a limb instead of the shuttle."""

# --- Track lifecycle ---
MAX_COAST_CV = 8
"""Maximum consecutive frames without a matching candidate before a track ends.
CV backend sees every frame, so a short coast is fine; TrackNet's batched
inference skips frames, so it needs more headroom (auto-calculated)."""

MAX_TRACK_LEN = 180
"""Maximum length of a single track (frames). Tracks longer than this are
assumed to be locked on clutter (a ceiling light, a logo) and are killed."""

MAX_CANDIDATES = 150
"""Maximum number of raw candidates per frame before the frame is considered
too noisy to use (camera shake, light flicker)."""

# --- CV backend (background subtraction) ---
CV_BG_METHOD = "knn"
CV_MIN_BLOB_AREA = 2
CV_MAX_BLOB_AREA = 500
CV_MAX_MERGED_AREA = 900
CV_MAX_ELONGATION = 6.0

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


# =====================================================================
#  LANDING / UMPIRE  (umpire.py)
# =====================================================================

UMPIRE_EDGE_MARGIN_PX = 25
"""Pixels from the ROI edge: a shuttle ending near the edge probably left the
frame rather than landing."""

UMPIRE_REST_SPEED_PX = 2.0
"""A shuttle moving slower than this for rest_frames consecutive frames is
considered to have come to rest (confirming a landing)."""

UMPIRE_REST_FRAMES = 2
"""How many consecutive slow frames confirm a landing."""

UMPIRE_CLOSE_CALL_M = 0.10
"""Signed distance (meters) below which a call is flagged as "close"."""

UMPIRE_MIN_FLIGHT_LEN = 5
"""Minimum track length (detections) for a flight to be judged."""

UMPIRE_MIN_DESCENT_PX = 80
"""Minimum vertical descent (px) from the flight's apex before looking for
an impact — prevents false positives from gentle tosses."""

UMPIRE_BOUNCE_DECEL_RATIO = 0.4
"""Impact is detected when vertical speed drops to this fraction of the peak
fall speed (a real bounce loses most of its speed; a smooth deceleration
approaching the top of an arc does not)."""

UMPIRE_SETTLE_SEARCH_FRAMES = 30
"""Frames after a candidate impact to look for the shuttle calming down."""

from . import court_model
UMPIRE_FLOOR_REGION = (-3.0, court_model.COURT_LENGTH + 3.0, -3.0, court_model.COURT_WIDTH + 3.0)
"""World-coordinate bounding box for valid landing positions (meters).
Anything outside this is a wall/ceiling hit, not a court landing.
(full court + 3 m buffer)"""


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
#  SMASH DETECTION  (scripts/test_smash.py)
# =====================================================================

SMASH_SPEED_THRESHOLD = 12.0
"""Minimum speed (px/frame) to classify a hit as a smash.
With TrackNet's default stride of 8, the two detection points used to
measure speed are ~8 frames apart, so the per-frame speed is the
displacement / 8. A real smash covers ~100+ px in 8 frames = ~12+ px/frame."""


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

DEFAULT_VIDEO = r"..\data\cfr\tran04_cam1.mp4"
"""Default video path when none is provided on the command line."""

DEFAULT_BACKEND = "tracknet"
"""Default shuttle detection backend."""

# --- Minimap ---
MINIMAP_SCALE = 45
"""Pixels per meter on the 2D minimap."""

MINIMAP_SIZE = (360, 720)
"""Minimap canvas size (width, height) in pixels."""

MINIMAP_BG_COLOR = (0, 100, 0)
"""Minimap background color (BGR)."""

# --- Tactical board (test_player_tracker.py) ---
BOARD_SCALE = 100
"""Pixels per meter on the tactical board."""
