import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import court_model, trajectory
from core.court_calibration import CourtCalibrator
from core.player_tracker import PlayerTracker
from core.video_io import ThreadedVideoReader

BOARD_SCALE = 100  # pixels per meter on the tactical board


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
    # grid rows run from x_min upwards; the board has the net (x_max) at the top
    heat = cv2.resize(np.flipud(grid), (board.shape[1], board.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.uint8(255 * heat / heat.max())
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    alpha = (heat_u8.astype(np.float32) / 255.0)[..., None] * 0.8
    out = (board * (1 - alpha) + colored * alpha).astype(np.uint8)
    for a, b in court_model.near_half_lines():
        cv2.line(out, board_point(a), board_point(b), (255, 255, 255), 1)
    return out


def test_tracking(video_path, stride=2, pose_variant="lite"):
    print(f"Opening video: {video_path}")
    reader = ThreadedVideoReader(video_path)
    fps = reader.fps / stride

    H, H_inv = court_model.load_calibration()
    if H is None:
        print("WARNING: calibration.json not found. Court lines will not be drawn and no court filtering is applied.")
    calibrator = CourtCalibrator()

    tracker = PlayerTracker(model_path="models/yolov8n.pt", conf_thresh=0.5, fps=fps, pose_variant=pose_variant)
    print("Starting video playback... Press 'q' to stop.")

    board = create_tactical_board()
    board_view = board.copy()
    history = {}  # player_id -> ([frame_idx], [(x, y)])

    fps_start_time, fps_frame_count, current_fps = time.time(), 0, 0.0
    last_display = None

    for frame_idx, frame in reader:
        if frame_idx % stride and last_display is not None:
            continue

        # inference runs on the clean frame; drawing happens afterwards
        players = tracker.process(frame, frame_idx, H, H_inv)

        for pid, p in players.items():
            if p.foot_world is not None:
                idxs, pts = history.setdefault(pid, ([], []))
                idxs.append(frame_idx // stride)
                pts.append(p.foot_world)

        annotated = calibrator.draw_court_frame(frame, H) if H is not None else frame
        annotated = tracker.draw_tracking(annotated, players)

        if history and (frame_idx // stride) % 15 == 0:
            pid = max(history, key=lambda k: len(history[k][0]))
            _, clean = trajectory.clean_player_track(*history[pid], fps)
            board_view = render_heatmap(board, clean, fps)
            stats = [f"P{pid} distance: {trajectory.path_length(clean):.1f} m",
                     f"valid: {100 * trajectory.valid_ratio(clean):.0f}%"]
            for i, s in enumerate(stats):
                cv2.putText(board_view, s, (10, 60 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed > 1.0:
            current_fps = fps_frame_count / elapsed
            fps_start_time, fps_frame_count = time.time(), 0
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
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=r"..\data\cfr\tran04_cam1.mp4")
    ap.add_argument("--stride", type=int, default=2, help="process every N-th frame")
    ap.add_argument("--pose", default="lite", choices=["lite", "heavy"])
    args = ap.parse_args()
    test_tracking(args.video, args.stride, args.pose)
