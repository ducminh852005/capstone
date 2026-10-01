"""
Test player tracking with YOLOv8 + ByteTrack + MediaPipe pose and a tactical heatmap.

Usage:
    python demo_player_tracker.py [video_path] [--stride N] [--pose lite|heavy]

Controls:
    q - Quit
"""
import argparse

import cv2
import numpy as np

# Shared utilities (also sets up sys.path)
from _common import setup_logging, setup_gui, FPSCounter, LiveTuner
from core import court_model, trajectory
from core.config import BOARD_SCALE, DEFAULT_VIDEO, PLAYER_DEMO_YOLO_CONF
from core.court_calibration import CourtCalibrator
from core.player_tracker import PlayerTracker
from core.video_io import ThreadedVideoReader


def board_point(xy):
    """World (x, y) -> tactical board pixel, net at the top as in the guideline."""
    x_min, x_max, y_min, y_max = court_model.REGION_BUFFERED
    return int((xy[1] - y_min) * BOARD_SCALE), int((x_max - xy[0]) * BOARD_SCALE)


def create_tactical_board():
    x_min, x_max, y_min, y_max = court_model.REGION_BUFFERED
    board = np.zeros((int((x_max - x_min) * BOARD_SCALE), int((y_max - y_min) * BOARD_SCALE), 3), np.uint8)
    board[:] = (40, 150, 40)
    for a, b in court_model.near_half_lines():
        cv2.line(board, board_point(a), board_point(b), (255, 255, 255), 2)
    return board


def render_heatmap(board, xy, fps):
    grid = trajectory.occupancy_heatmap(xy, fps)
    if grid.max() <= 0:
        return board.copy()
    heat = cv2.resize(np.flipud(grid), (board.shape[1], board.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.uint8(255 * heat / heat.max())
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    alpha = (heat_u8.astype(np.float32) / 255.0)[..., None] * 0.8
    out = (board * (1 - alpha) + colored * alpha).astype(np.uint8)
    for a, b in court_model.near_half_lines():
        cv2.line(out, board_point(a), board_point(b), (255, 255, 255), 1)
    return out


def run_tracking(video_path, stride=2, pose_variant="lite"):
    print(f"Opening video: {video_path}")
    reader = ThreadedVideoReader(video_path)
    fps = reader.fps / stride

    H, H_inv = court_model.load_calibration()
    if H is None:
        print("WARNING: calibration.json not found. Court lines will not be drawn.")
    calibrator = CourtCalibrator()

    tracker = PlayerTracker(conf_thresh=PLAYER_DEMO_YOLO_CONF, fps=fps, pose_variant=pose_variant,
                           yolo_every=1)
    print("Starting video playback... Press 'q' to stop.")

    tuner = LiveTuner("Tracker Tuner")
    # Bind to the live selector instance: config.SELECTOR_* are only read at construction.
    tuner.bind("Min Score", tracker.selector, "min_score", 0.0, 1.0, 0.05)
    tuner.bind("Switch Margin", tracker.selector, "switch_margin", 0.0, 1.0, 0.05)
    tuner.bind("ReID Dist (m)", tracker.selector, "reid_dist", 0.5, 5.0, 0.25)
    tuner.bind("ReID Window (frames)", tracker.selector, "reid_window", 30, 600, 30)

    board = create_tactical_board()
    board_view = board.copy()
    history = {}  # player_id -> ([frame_idx], [(x, y)])

    fps_counter = FPSCounter()
    last_display = None

    for frame_idx, frame in reader:
        if frame_idx % stride and last_display is not None:
            continue

        players = tracker.process(frame, frame_idx, H, H_inv)

        for pid, p in players.items():
            if p.foot_world is not None:
                idxs, pts = history.setdefault(pid, ([], []))
                idxs.append(frame_idx // stride)
                pts.append(p.foot_world)

        annotated = calibrator.draw_court_frame(frame, H, copy=False) if H is not None else frame
        annotated = tracker.draw_tracking(annotated, players, copy=False)

        if history and (frame_idx // stride) % 15 == 0:
            pid = max(history, key=lambda k: len(history[k][0]))
            _, clean = trajectory.clean_player_track(*history[pid], fps)
            board_view = render_heatmap(board, clean, fps)
            stats = [f"P{pid} distance: {trajectory.path_length(clean):.1f} m",
                     f"valid: {100 * trajectory.valid_ratio(clean):.0f}%"]
            for i, s in enumerate(stats):
                cv2.putText(board_view, s, (10, 60 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_fps = fps_counter.tick()
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        display_video = cv2.resize(annotated, (1024, 576))
        display_board = cv2.resize(board_view, (int(576 * board.shape[1] / board.shape[0]), 576))
        cv2.putText(display_board, "2D Movement Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        last_display = cv2.hconcat([display_video, display_board])
        cv2.imshow("YOLOv8 + ByteTrack: Player Tracking", last_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    setup_gui()
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    ap.add_argument("--stride", type=int, default=2, help="process every N-th frame")
    ap.add_argument("--pose", default="lite", choices=["lite", "heavy"])
    args = ap.parse_args()
    run_tracking(args.video, args.stride, args.pose)
