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

STANDARDIZE_CRF = 18
"""x264 quality (CRF) of the 60 fps CFR copies made by core/video_processor.standardize_video_fps."""

STANDARDIZE_PRESET = "medium"
"""x264 preset of those CFR copies (they are made once, so size over speed)."""

CLIPS_DIR = DATA_DIR / "clips"
"""Output directory of scripts/analyze_clip.py: one sub-folder per analysed clip (git-ignored)."""

TELESTRATOR_TEMPLATE_PATH = BACKEND_DIR / "web" / "telestrator.html"
"""HTML template of the clip viewer (core/clip_report.py fills in the analysis JSON)."""

CFR_FPS = 60.0
"""Constant frame rate (frames per second) every analysed video is standardised to
(scripts/process_all_videos.py). A video with another rate is analysed with its real rate, with a warning."""

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

KALMAN_MAX_SPEED_RATIO_PER_FRAME = 2.0
"""Same check for a detection stream with a candidate on EVERY frame (TRACKNET_ALL_HEATMAPS). The
velocities compared are chords over >= PHYSICS_CHORD_MIN_FRAMES frames of a continuously updated
track, so a hit shows up as a smaller ratio than with detections a stride apart; 2.0 finds the
frame-394 smash (20 px/frame) that 3.0 lets through as ordinary flight."""

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

TRACKNET_ALL_HEATMAPS = False
"""Use all seq_len (8) heatmaps of every forward pass instead of only the newest one. The stride
is then seq_len (no overlap: 14.6 ms/frame of GPU instead of 23.3 at stride 5) and there is a
detection candidate for EVERY frame, but they arrive in bursts, so the source replays them one
per call with a constant delay of seq_len - 1 frames (ShuttleDetector.frame_lag). Consumers must
use `frame_idx - detector.frame_lag` as the frame of the returned point."""

TRACKNET_IDLE_STRIDE = 8
"""Stride used while no shuttle track is active (>= TRACKNET_BATCH_STRIDE; 0 = same as the active
stride). About 90% of a match has no shuttle in play, and a new track only needs one confident
detection to start, so looking less often then saves forward passes (8 frames: 14.6 ms/frame of
GPU instead of 23.3) at the price of noticing a new flight up to 3 frames later. Only applies
when the caller did not ask for an explicit batch_stride."""

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

PHYSICS_STOP_BASELINE_FRAMES = 2
"""A physics break is classified as a stop (the shuttle came to rest) from its speed measured against
the newest detection at least this old. Over fewer frames position noise alone exceeds
PHYSICS_STOP_SPEED_PX; but measuring against the 4-frame reference of the physics test itself would
read a shuttle that reversed (net displacement ~0) as a stop. Detections >= 5 frames apart already
satisfy it. The CV backend uses 1."""

PHYSICS_STOP_SPEED_PX = 2.0
"""When the physics filter breaks a track and the shuttle leaves the break slower than this
(px/frame), it came to rest (landed or resting on the floor): the track restarts but it is
not a racket hit and not a new flight."""

PHYSICS_MAX_DECEL_RATIO = 10.0
"""Reject a candidate if it would slow the shuttle by more than this factor.
Air drag can cost 3-4x in one batch, so 10x means it was not the same shuttle."""

PHYSICS_HIT_GATE_FRAMES = 5
"""A racket hit can move the shuttle away from where the Kalman filter expects it by roughly what
it travels in this many frames. The distance within which an unphysical candidate is taken for
"the shuttle after a hit" (rather than an unrelated blob) is min_gate_px scaled to this many
frames. With detections >= 5 frames apart the ordinary gate is already that wide; with a
detection on every frame it is 5x wider than the tracking gate."""

PHYSICS_CHORD_MIN_FRAMES = 4
"""The physics filter compares velocities measured over at least this many frames: the newest
detection at least this old is the reference point, and the velocity before it is measured
against an earlier detection at least this far back. With a detection every TrackNet stride
(>= 4 frames) that is simply the last two detections; with a detection on EVERY frame it stops
position noise of a few px turning into velocity noise as large as the speed itself. The CV
backend (a detection per frame, tuned that way) uses 1."""

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

SHUTTLE_FILL_LINK_FRAMES = 12
"""Default chain link (frames) for gap_fill.fill_gaps callers that do not know the stride;
ShuttleDetector.fill_gaps derives it from SHUTTLE_FILL_LINK_STRIDES instead."""

SHUTTLE_FILL_MIN_CHAIN_LEN = 2
"""A chain must hold at least this many detections on both sides of a gap for the gap to
be bridged: fewer cannot tell which way the shuttle was flying."""

SHUTTLE_FILL_MAX_GAP_FRAMES = 48
"""Longest gap (frames) bridged between two chains (~10 strides at the default stride of 5)."""

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
over 32; measured on tran04 at stride 8, order of magnitude the same at 5). Beyond this many
frames the RMS limit shrinks in proportion to the gap."""


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

UMPIRE_MIN_STEP_FRAMES = 5
"""The landing logic (impact, rest speed, min flight length, look-ahead) is tuned for detections
about 5 frames apart (the default TrackNet stride). A denser stream (a detection on every frame)
is thinned to at least this spacing before it is analysed; sparser streams are untouched."""

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

PLAYER_YOLO_HALF = False
"""Run YOLO in FP16 on CUDA. Off: on the reference GPU (T550, no useful FP16 throughput) FP16
made every track() call ~2x slower (30-35 ms vs 14-16 ms). GPUs with real FP16 units may gain."""

PLAYER_YOLO_IMGSZ = 640
"""YOLO input image size (pixels)."""

PLAYER_POSE_EVERY = 5
"""Run MediaPipe pose on the selected player every N update() calls."""

PLAYER_YOLO_EVERY = 3
"""Run YOLO+ByteTrack every N process() calls; in between, reuse last boxes. Boxes are only used
to exclude players' bodies from the shuttle candidates and to pick the player, so a box that is
2 frames old is accurate enough; 3 halves the player cost (26 -> 14 ms/frame, FP32). Callers that
draw or measure player positions every frame (demo_player_tracker) pass yolo_every=1."""

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

SMASH_MIN_DT_FRAMES = 4
"""The hit's speed is measured to the first detection at least this many frames after the track
started: over fewer frames a few px of position noise is as large as the displacement. With a
detection every TrackNet stride (>= 4) this is the first detection; with one on every frame it
skips the first few."""

SMASH_MIN_Y = 50.0
"""Ignore hits whose image y (px) is above this: the shuttle is out of the play area."""

SMASH_MIN_ANGLE = 10.0
"""Minimum downward angle (degrees, 0 = horizontal) of a smash's direction."""

SMASH_MAX_ANGLE = 170.0
"""Maximum angle (degrees) of a smash's direction; with SMASH_MIN_ANGLE it bounds
the accepted cone (0 = +x axis, 90 = straight down in image coordinates)."""


# =====================================================================
#  CLIP ANALYSIS  (clip_pipeline.py, clip_analysis.py, scripts/analyze_clip.py)
# =====================================================================

CLIP_PREROLL_S = 2.0
"""Seconds analysed before the requested clip start and then excluded from the statistics. The
trackers need them: TrackNet builds its background from TRACKNET_BG_FRAMES (60) frames and needs a
window of 8 more before its first detection (~76 frames, 1.3 s at 60 fps), and the player selector
needs ~30 in-court frames before it picks anybody."""

CLIP_DEFAULT_FRAMES = 300
"""Frames of a clip when scripts/analyze_clip.py is not given --frames (5 s at 60 fps: a rally or two)."""

CLIP_MAX_PLAYERS = 2
"""Near-half players tracked (and analysed one by one) per clip, so the viewer can offer a choice
when a referee or a second person stands on the court. Passed to PlayerSelector(max_players)."""

CLIP_YOLO_EVERY = 1
"""YOLO+ByteTrack on every frame in clip mode: the clip is a few hundred frames, so the extra cost
is small and the player boxes drawn over the video are never stale."""

CLIP_POSE_EVERY = 1
"""MediaPipe pose on every frame of the selected players in clip mode: the skeleton is drawn over
the video and the hit analysis needs the wrist position within +-1 frame of the contact."""

CLIP_POSE_VARIANT = "heavy"
"""MediaPipe model of clip mode. The heavy model keeps the racket arm better than "lite" when the
player is seen from behind at ~2 m (self-occlusion), at ~2x the cost per call."""

CLIP_PREROLL_POSE_EVERY = 3
"""Pose cadence (frames) of the selected players during the pre-roll: it only has to let the player
selector lock on, and the statistics ignore those frames, so a cached foot offset between pose calls
is enough (~15-25% of the pose calls of a clip are pre-roll). Set to CLIP_POSE_EVERY to disable."""

CLIP_CUT_CRF = 18
"""x264 quality (CRF) of the clip cut out of the source video. The same pixels are analysed and
shown in the browser, so keep it high (18 is visually lossless)."""

CLIP_CUT_PRESET = "veryfast"
"""x264 preset of the clip cut (speed over file size: the clips are a few hundred frames)."""

CLIP_CUT_GOP_FRAMES = 12
"""Keyframe interval (frames) of the cut clip. A short GOP without B-frames makes stepping one
frame backwards in the browser cheap and exact."""

CLIP_EXPORT_PX_DECIMALS = 1
"""Decimals kept for pixel coordinates in analysis.json (0.1 px is far below the detection noise)."""

CLIP_EXPORT_WORLD_DECIMALS = 3
"""Decimals kept for court coordinates (metres) in analysis.json: millimetres, far below the
~1 cm calibration accuracy."""

CLIP_SKELETON_MIN_SCORE = 0.4
"""Landmark visibility below which a joint is exported as hidden (-1) and not drawn by the viewer (same
as PoseEstimator.draw_landmarks)."""

CLIP_EXPORT_SPEED_DECIMALS = 2
"""Decimals kept for speeds (px/frame, body heights/s, m/s) and floor distances (m) in analysis.json."""

CLIP_EXPORT_ANGLE_DECIMALS = 1
"""Decimals kept for angles (degrees) in analysis.json."""

CLIP_EXPORT_RATIO_DECIMALS = 3
"""Decimals kept for distances in body heights and for times of flight (s) in analysis.json."""

CLIP_EXPORT_CONFIDENCE_DECIMALS = 2
"""Decimals kept for confidences and shares (0..1) in analysis.json."""

CLIP_TRAIL_FRAMES = 45
"""Frames of shuttle trail the viewer draws behind the current frame."""

CLIP_MIN_PLAYER_FRAMES = 30
"""A tracked person seen in fewer clip frames than this is not offered as a player in the viewer
(a passer-by or a one-off tracker identity)."""

# --- Pose ---
POSE_MIN_SCORE = 0.5
"""Landmark visibility (0..1) a joint needs to be used for a measurement (hand position, ...)."""

POSE_CONTACT_WINDOW_FRAMES = 2
"""Frames either side of a hit's contact frame searched for a skeleton when there is none on the
contact frame itself (pose fails on a frame now and then)."""

# --- Hit attribution (hit_events.py) ---
# Initial guesses: there are no hand-labelled hits yet (see CLAUDE.md, Known issues).
HITTER_MAX_WRIST_DIST_BODY = 0.5
"""A hit belongs to a tracked player when one of his hand points (wrist, index fingertip) comes this
close to the shuttle's path across the contact, in body heights (distance in px / the player's box
height). The racket head is ~0.3 body heights beyond the wrist; a smash measured on tran04 gave 0.12."""

HITTER_MIN_NEAR_CONFIDENCE = 0.5
"""Confidence of a hit attributed at exactly HITTER_MAX_WRIST_DIST_BODY; it grows linearly to 1 as the
hand gets closer to the shuttle."""

HITTER_WINDOW_MARGIN_FRAMES = 2
"""Frames added on each side of the contact window (last detection before the hit .. first detection
after it) when looking for the hand: pose and shuttle detections are not perfectly synchronous."""

HITTER_DEFAULT_WINDOW_FRAMES = 5
"""Contact window (frames before the first detection after the hit) when no earlier detection of the
shuttle is known: one default TrackNet stride (TRACKNET_BATCH_STRIDE)."""

HITTER_MAX_PREV_GAP_FRAMES = 15
"""How far back (frames) to look for the last detection before a hit. Older ones belong to another flight."""

HITTER_LANDING_EXCLUDE_FRAMES = 8
"""A track restart within this many frames of a landing call, with no hand near the shuttle, is the
shuttle bouncing on the floor (the tracker sees a bounce as a direction reversal like a racket hit),
not a hit by the opponent."""

HITTER_FAR_CONFIDENCE = 0.3
"""Confidence of a hit inferred to be the opponent's (no near hand close to the shuttle) that follows
a hit of the near player, as the alternation of a singles rally requires. Low: it is only an inference."""

HITTER_FAR_BREAK_CONFIDENCE = 0.2
"""Same, when it does NOT alternate (it follows another inferred opponent hit, or opens a rally): a
missed hit or a tracker glitch is as likely as a real opponent hit."""

# --- Shot classification (shots.py) ---
# Initial guesses: there are no hand-labelled shots yet (see CLAUDE.md, Known issues). The pose is 2D
# and filmed from behind, and the shuttle's height is unknown, so these are weak rules, not measurements.
SHOT_OVERHEAD_MIN_ABOVE_HEAD_BODY = -0.1
"""The racket hand is overhead when its wrist is at least this far above the nose (body heights, i.e.
pixels / the height of the player's box; positive = above). Negative: the racket head sits ~0.3 body
heights beyond the wrist, so a wrist a little below the nose already hits overhead. The only smash
measured on tran04 had its wrist 0.013 below the nose; this is not calibrated on more than that."""

SHOT_LOW_MIN_BELOW_HIP_BODY = 0.0
"""The racket hand is low when its wrist is at least this far below the hips (body heights)."""

SHOT_SERVE_MAX_X_M = 4.7
"""A serve is hit from behind this distance (m) from the near baseline: inside the service court
(the short service line is at 4.72 m)."""

SHOT_SMASH_MIN_ELBOW_DEG = 140.0
"""Elbow angle (degrees, 180 = straight arm) above which an overhead hit counts as made with a
fully extended arm: supporting evidence for a smash."""

SHOT_SMASH_MIN_SPEED_BODY_PER_S = 2.5
"""Shuttle speed right after the hit, in body heights per second (speed_px * fps / box height px), from
which an overhead hit is a smash. Image-plane speed: a shot flying away from the camera looks slower."""

SHOT_FLAT_ELEVATION_DEG = 10.0
"""Direction of the shuttle right after the hit (degrees above the horizontal in the image): within
+-this it is flat, above it rises, below it falls."""

SHOT_DROP_MAX_LANDING_FROM_NET_M = 2.5
"""An overhead soft hit that lands within this distance (m) of the net is a drop."""

SHOT_DROP_MAX_FLIGHT_S = 0.9
"""An overhead soft hit that reaches the other side (next hit or landing) within this many seconds is a
drop (a clear takes longer)."""

SHOT_CLEAR_MIN_FLIGHT_S = 1.1
"""An overhead hit that rises and flies at least this many seconds is a clear."""

SHOT_LIFT_MIN_FLIGHT_S = 1.0
"""A low hit that rises and flies at least this many seconds is a lift (lob)."""

SHOT_NET_MAX_FLIGHT_S = 0.7
"""A soft hit made at the net that flies at most this many seconds is a net shot."""

SHOT_DRIVE_MIN_SPEED_BODY_PER_S = 2.0
"""A flat hit at mid height faster than this (body heights per second) is a drive."""

SHOT_DRIVE_MAX_FLIGHT_S = 0.6
"""A drive reaches the other side within this many seconds."""

SHOT_NO_POSE_MAX_CONF = 0.4
"""Confidence ceiling of a shot classified without a skeleton (shuttle and court features only)."""

SHOT_EVAL_TOL_FRAMES = 6
"""Frames within which a predicted hit and a hand-labelled hit are the same one when shot types are
scored (scripts/eval_shots.py). The hit's contact frame is refined to the hand's closest approach, so
this is tighter than the +-12 frames of the raw hit detection (scripts/eval_events.py)."""

SHOT_BASE_STRENGTH = 0.6
"""A rule that matches on its required conditions alone has this strength (0..1); each supporting
condition that also holds raises it linearly to 1. Shot confidence = hit confidence x strength."""

RALLY_DEAD_TIME_S = 4.0
"""Seconds without play (since the previous rally ended) before a hit is taken for a serve."""

# --- Zones and movement (zones.py, movement.py) ---
ZONE_DEPTH_EDGES_M = (2.2, 4.4)
"""x (m from the near baseline) where the rear zone ends and the front zone begins: the near half
(0..NET_X = 6.7 m) in three bands of about a third. Initial guess; the guideline asks for a 6-zone split
(front / mid / rear x left / right)."""

MOVE_SPEED_LEVELS_MPS = (0.5, 2.0, 4.0)
"""Speed (m/s) edges of the levels standing, walking, running, sprinting (guideline: 0.5, 2 and 4 m/s)."""

MOVE_SPEED_PERCENTILE = 95
"""Percentile of the speed samples reported as the player's top speed (never the maximum: a single
mis-tracked frame would set it)."""

MOVE_STILL_SPEED_MPS = 0.5
"""A player slower than this (m/s) is standing still: waiting position and rest are measured on it."""

MOVE_REST_MIN_S = 2.0
"""Seconds of standing still in a row that count as a rest (between points, not a split-step)."""

MOVE_BASE_MIN_S = 0.5
"""Seconds of standing still needed to call its centroid the player's waiting position."""

MOVE_HOME_RADIUS_M = 0.75
"""The player is back at the waiting position when within this distance (m) of it."""

MOVE_RECOVERY_MAX_S = 3.0
"""Longest wait (s) for the player to get back to the waiting position after a hit; later (or when
the next hit comes first) the recovery is reported as not completed."""

MOVE_MIN_VALID_RATIO = 0.6
"""Below this fraction of frames with a usable position the movement numbers are marked unreliable."""

MOVE_SMALL_SAMPLE_S = 10.0
"""A clip analysed over less than this many seconds gives only a small sample of movement."""

# --- Rally segmentation (rally.py) ---
RALLY_MAX_HIT_GAP_S = 3.0
"""Seconds between two hits after which the rally is considered over (shuttle dead, call missed). A
high clear flies ~1.5-2 s, so this leaves room for a couple of missed detections."""

RALLY_CLOSE_CALL_FACTOR = 0.5
"""Winner confidence is multiplied by this when the landing call is a close call."""

RALLY_RESTING_FACTOR = 0.7
"""Winner confidence is multiplied by this when the landing was called from a resting shuttle
(Call.method == "resting": the impact itself was not seen)."""

RALLY_OWN_HALF_FACTOR = 0.5
"""Winner confidence is multiplied by this when the shuttle landed on the half of the player who hit
it last: a net fault, or (as likely) an opponent hit that was not detected."""

RALLY_INFERRED_HITTER_CONFIDENCE = 0.3
"""Confidence of the last hitter of a rally with no detected hit at all, inferred as the side opposite
to where the shuttle landed."""


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
