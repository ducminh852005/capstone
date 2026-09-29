import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import court_model
from core.player_tracker import PlayerTracker
from core.shuttle_tracker import ShuttleDetector
from core.umpire import RallyUmpire, match_type_from_metadata
from core.video_io import ThreadedVideoReader


def test_auto_umpire(video_path, shuttle_backend="tracknet"):
    print(f"Opening video: {video_path}")
    H, H_inv = court_model.load_calibration()
    if H is None:
        print("ERROR: Please run test_calibration.py first!")
        return

    reader = ThreadedVideoReader(video_path)
    w, h = reader.size
    roi = court_model.shuttle_roi(H, (h, w))
    match_type = match_type_from_metadata(video_path)
    print(f"Match type: {match_type}, shuttle ROI: {roi}")

    try:
        detector = ShuttleDetector(backend=shuttle_backend)
    except FileNotFoundError as e:
        if shuttle_backend != "tracknet":
            raise
        print(f"WARNING: {e}\nFalling back to backend='cv'.")
        detector = ShuttleDetector(backend="cv")
    print(f"Shuttle backend: {detector.backend}")
    # people detection is only used to mask bodies/spectators out of the shuttle candidates
    tracker = PlayerTracker(model_path="yolov8n.pt", conf_thresh=0.5, fps=reader.fps, yolo_every=3)
    umpire = RallyUmpire(H_inv, match_type=match_type, frame_size=(h, w), roi=roi,
                         net_top_y=court_model.net_top_threshold_y(H))

    score_near = 0
    score_far = 0

    # UI Overlay States
    show_alert_until = 0
    alert_text = ""
    alert_color = (255, 255, 255)
    last_landing_pt = None

    full_court_img = np.int32(court_model.world_to_img(
        [(0, 0), (court_model.COURT_LENGTH, 0), (court_model.COURT_LENGTH, court_model.COURT_WIDTH), (0, court_model.COURT_WIDTH)], H)).reshape(-1, 1, 2)
    net_img = np.int32(court_model.world_to_img([(court_model.NET_X, 0), (court_model.NET_X, court_model.COURT_WIDTH)], H))

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    x0, y0, x1, y1 = roi

    for frame_idx, frame in reader:
        players = tracker.process(frame, frame_idx, H, H_inv)
        pt, fg_mask_roi = detector.detect(frame, roi=roi,
                                          exclude_boxes=tracker.non_player_boxes(players),
                                          player_boxes=[p.bbox for p in players.values()])
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        fg_mask[y0:y1, x0:x1] = fg_mask_roi

        call = umpire.update(frame_idx, pt, detector.track_active, people_boxes=list(tracker.last_boxes))
        if call is not None:
            last_landing_pt = call.image_pt
            # a landing in a half was hit by the player on the OTHER half; IN means that
            # hitter scores, OUT means the receiving side scores instead
            hitter = "far" if call.half == "near" else "near"
            winner = hitter if call.result == "IN" else ("near" if hitter == "far" else "far")
            if winner == "far":
                score_far += 1
                alert_color = (0, 255, 0)
            else:
                score_near += 1
                alert_color = (0, 0, 255)
            alert_text = f"{call.half.upper()} {call.result}{' (close)' if call.close_call else ''} {call.margin_m:+.2f} m"
            print(f"frame {frame_idx}: {call.to_dict()}")
            show_alert_until = 60

        annotated = detector.draw_trajectory(frame)

        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed > 1.0:
            current_fps = fps_frame_count / elapsed
            fps_start_time = time.time()
            fps_frame_count = 0
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.polylines(annotated, [full_court_img], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.line(annotated, tuple(int(v) for v in net_img[0]), tuple(int(v) for v in net_img[1]), (0, 0, 255), 2)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 128, 0), 1)

        cv2.rectangle(annotated, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(annotated, f"NEAR: {score_near}  |  FAR: {score_far}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        if show_alert_until > 0:
            show_alert_until -= 1
            cv2.putText(annotated, alert_text, (400, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, alert_color, 4)
            if last_landing_pt:
                radius = 10 + (show_alert_until % 10)
                cv2.circle(annotated, last_landing_pt, radius, alert_color, -1)

        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        mask_label = "Background Subtractor Mask" if detector.backend == "cv" else "TrackNet Heatmap"
        cv2.putText(mask_bgr, mask_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        combined = cv2.hconcat([cv2.resize(annotated, (640, 360)), cv2.resize(mask_bgr, (640, 360))])
        cv2.imshow("Left: Auto Umpire (Hawk-Eye) | Right: Mask", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else r"..\data\cfr\tran04_cam1.mp4"
    backend = sys.argv[2] if len(sys.argv) > 2 else "tracknet"
    test_auto_umpire(target_video, backend)
