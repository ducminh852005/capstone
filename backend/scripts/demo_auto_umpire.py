"""
Test the full auto-umpire pipeline: shuttle tracking + player tracking + line calls.

Usage:
    python scripts/demo_auto_umpire.py [video_path] [backend]

Controls:
    q - Quit
"""
import sys

import cv2
import numpy as np

# Shared utilities (also sets up sys.path)
from _common import create_detector, setup_logging, setup_gui, FPSCounter, LiveTuner
from core import court_model
from core.config import DEFAULT_VIDEO, DEFAULT_BACKEND, PLAYER_DEMO_YOLO_CONF, UMPIRE_ALERT_FRAMES
from core.player_tracker import PlayerTracker
from core.umpire import RallyUmpire, match_type_from_metadata
from core.video_io import ThreadedVideoReader


def run_auto_umpire(video_path, shuttle_backend=DEFAULT_BACKEND):
    print(f"Opening video: {video_path}")
    H, H_inv = court_model.load_calibration()
    if H is None:
        print("ERROR: Please run calibrate_court.py first!")
        return

    reader = ThreadedVideoReader(video_path)
    w, h = reader.size
    roi = court_model.shuttle_roi(H, (h, w))
    match_type = match_type_from_metadata(video_path)
    print(f"Match type: {match_type}, shuttle ROI: {roi}")

    detector = create_detector(shuttle_backend)
    print(f"Shuttle backend: {detector.backend}")

    tracker = PlayerTracker(conf_thresh=PLAYER_DEMO_YOLO_CONF, fps=reader.fps)
    umpire = RallyUmpire(H_inv, match_type=match_type, frame_size=(h, w), roi=roi,
                         net_top_y=court_model.net_top_threshold_y(H))

    tuner = LiveTuner("Umpire Tuner")
    tuner.bind("Umpire Rest Speed", umpire, "rest_speed", 0.0, 10.0, 0.1)
    tuner.bind("Umpire Rest Frames", umpire, "rest_frames", 1, 20, 1)
    tuner.bind("Umpire Min Descent", umpire, "min_descent", 5.0, 200.0, 5.0)

    score_near = 0
    score_far = 0

    # UI Overlay States
    show_alert_until = 0
    alert_text = ""
    alert_color = (255, 255, 255)
    last_landing_pt = None

    full_court_img = np.int32(court_model.world_to_img(
        [(court_model.BASELINE_X, court_model.SIDELINE_DOUBLES_Y[0]),
         (court_model.COURT_LENGTH, court_model.SIDELINE_DOUBLES_Y[0]),
         (court_model.COURT_LENGTH, court_model.SIDELINE_DOUBLES_Y[1]),
         (court_model.BASELINE_X, court_model.SIDELINE_DOUBLES_Y[1])], H)).reshape(-1, 1, 2)
    net_img = np.int32(court_model.world_to_img(
        [(court_model.NET_X, court_model.SIDELINE_DOUBLES_Y[0]),
         (court_model.NET_X, court_model.SIDELINE_DOUBLES_Y[1])], H))

    fps_counter = FPSCounter()
    x0, y0, x1, y1 = roi

    # Side panel showing the detector's mask, placed where the ROI sits in the 640x360 overview.
    # It only changes when the detector produces a new mask (TrackNet: every batch, i.e. not every
    # frame), so it is rebuilt then, on a black canvas that already has the ROI's position.
    panel_w, panel_h = 640, 360
    px0, py0 = int(round(x0 * panel_w / w)), int(round(y0 * panel_h / h))
    px1, py1 = min(panel_w, int(round(x1 * panel_w / w))), min(panel_h, int(round(y1 * panel_h / h)))
    mask_label = "Background Subtractor Mask" if detector.backend == "cv" else "TrackNet Heatmap"
    mask_panel, last_mask = None, None
    combined = np.empty((panel_h, 2 * panel_w, 3), np.uint8)

    for frame_idx, frame in reader:
        players = tracker.process(frame, frame_idx, H, H_inv)
        pt, fg_mask_roi = detector.detect(frame, roi=roi,
                                          exclude_boxes=tracker.non_player_boxes(players),
                                          player_boxes=[p.bbox for p in players.values()])

        call = umpire.update(frame_idx - detector.frame_lag, pt, detector.track_active,
                             people_boxes=list(tracker.last_boxes),
                             flight_id=detector.flight_id)
        if call is not None:
            last_landing_pt = call.image_pt
            hitter = "far" if call.half == "near" else "near"
            winner = hitter if call.result == "IN" else ("near" if hitter == "far" else "far")
            if winner == "far":
                score_far += 1
                alert_color = (0, 255, 0)
            else:
                score_near += 1
                alert_color = (0, 0, 255)
            alert_text = (f"{call.half.upper()} {call.result}{' (close)' if call.close_call else ''} "
                          f"{call.margin_m:+.2f} m [{call.method}]")
            print(f"frame {frame_idx}: {call.to_dict()}")
            show_alert_until = UMPIRE_ALERT_FRAMES

        annotated = detector.draw_trajectory(frame, copy=False)     # `frame` is not used afterwards

        current_fps = fps_counter.tick()
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (1000, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.polylines(annotated, [full_court_img], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.line(annotated, tuple(int(v) for v in net_img[0]), tuple(int(v) for v in net_img[1]), (0, 0, 255), 2)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 128, 0), 1)

        cv2.rectangle(annotated, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(annotated, f"NEAR: {score_near}  |  FAR: {score_far}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        if show_alert_until > 0:
            show_alert_until -= 1
            cv2.putText(annotated, alert_text, (400, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, alert_color, 4)
            if last_landing_pt:
                radius = 10 + (show_alert_until % 10)
                cv2.circle(annotated, last_landing_pt, radius, alert_color, -1)

        if fg_mask_roi is not last_mask:
            last_mask = fg_mask_roi
            mask_panel = np.zeros((panel_h, panel_w, 3), np.uint8)
            small = cv2.resize(fg_mask_roi, (px1 - px0, py1 - py0), interpolation=cv2.INTER_AREA)
            mask_panel[py0:py1, px0:px1] = small[..., None]
            cv2.putText(mask_panel, mask_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        combined[:, :panel_w] = cv2.resize(annotated, (panel_w, panel_h))
        combined[:, panel_w:] = mask_panel
        cv2.imshow("Left: Auto Umpire (Hawk-Eye) | Right: Mask", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    setup_gui()
    target_video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    backend = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BACKEND
    run_auto_umpire(target_video, backend)
