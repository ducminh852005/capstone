import os
import sys
import time
import json
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import court_model
from core.player_tracker import PlayerTracker
from core.shuttle_tracker import ShuttleDetector
from core.umpire import RallyUmpire, match_type_from_metadata
from core.video_io import ThreadedVideoReader

def generate_highlights(video_path, shuttle_backend="tracknet", output_html="highlights.html"):
    print(f"Opening video: {video_path}")
    H, H_inv = court_model.load_calibration()
    if H is None:
        print("ERROR: Please run test_calibration.py first!")
        return

    reader = ThreadedVideoReader(video_path)
    w, h = reader.size
    fps = reader.fps
    roi = court_model.shuttle_roi(H, (h, w))
    match_type = match_type_from_metadata(video_path)
    
    try:
        detector = ShuttleDetector(backend=shuttle_backend)
    except FileNotFoundError as e:
        if shuttle_backend != "tracknet":
            raise
        print(f"WARNING: {e}\nFalling back to backend='cv'.")
        detector = ShuttleDetector(backend="cv")
        
    tracker = PlayerTracker(model_path="yolov8n.pt", conf_thresh=0.5, fps=fps, yolo_every=3)
    umpire = RallyUmpire(H_inv, match_type=match_type, frame_size=(h, w), roi=roi,
                         net_top_y=court_model.net_top_threshold_y(H))

    rallies = []
    near_hits = []
    
    current_rally_start_frame = 0
    current_rally_start_time = 0.0
    current_rally_hits = 0
    
    was_active = False
    
    print("Processing video to extract highlights... (this may take a while depending on video length)")
    start_time = time.time()
    
    for frame_idx, frame in reader:
        if frame_idx % 300 == 0:
            print(f"Processing frame {frame_idx}...")
            
        players = tracker.process(frame, frame_idx, H, H_inv)
        pt, fg_mask_roi = detector.detect(frame, roi=roi,
                                          exclude_boxes=tracker.non_player_boxes(players),
                                          player_boxes=[p.bbox for p in players.values()])

        if not was_active and detector.track_active:
            # A new track just started (a hit occurred)
            current_rally_hits += 1
            
            # Check if this hit is a potential smash from the near side
            vx, vy = detector.kf.x[2:4]
            speed = np.hypot(vx, vy)
            start_pt = detector.kf.x[:2]
            
            world_pt = court_model.img_to_world([start_pt], H_inv)[0]
            is_near = world_pt[0] < court_model.NET_X
            
            if is_near:
                near_hits.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'speed': float(speed)
                })

        call = umpire.update(frame_idx, pt, detector.track_active, people_boxes=list(tracker.last_boxes))
        if call is not None:
            # Rally ended
            rallies.append({
                'start_frame': current_rally_start_frame,
                'end_frame': frame_idx,
                'start_time': current_rally_start_time,
                'end_time': frame_idx / fps,
                'hits': current_rally_hits,
                'result': call.result,
                'half': call.half
            })
            # Reset for next rally
            current_rally_start_frame = frame_idx
            current_rally_start_time = frame_idx / fps
            current_rally_hits = 0
            
        was_active = detector.track_active

    reader.release()
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
    
    # Sort rallies by number of hits (longest rally)
    rallies.sort(key=lambda x: x['hits'], reverse=True)
    longest_rally = rallies[0] if rallies else None
    
    # Sort near hits by speed to find smashes (fastest downward hits)
    near_hits.sort(key=lambda x: x['speed'], reverse=True)
    top_smashes = near_hits[:5]
    
    print("\n--- Highlights Extracted ---")
    if longest_rally:
        print(f"Longest Rally: {longest_rally['hits']} hits, from {longest_rally['start_time']:.2f}s to {longest_rally['end_time']:.2f}s")
    print("Top Near Smashes:")
    for i, smash in enumerate(top_smashes):
        print(f"  {i+1}. Time: {smash['time']:.2f}s, Speed: {smash['speed']:.2f} px/frame")
        
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Badminton Highlights Demo</title>
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #121212; color: #fff; padding: 20px; text-align: center; }}
            video {{ width: 80%; max-width: 1000px; border: 2px solid #333; border-radius: 8px; margin-bottom: 20px; }}
            .controls {{ display: flex; flex-direction: column; align-items: center; gap: 15px; }}
            .section {{ background: #1e1e1e; padding: 15px; border-radius: 8px; width: 80%; max-width: 1000px; }}
            .btn {{ background-color: #bb86fc; color: #000; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; font-weight: bold; margin: 5px; transition: background 0.3s; }}
            .btn:hover {{ background-color: #9965f4; }}
            h2 {{ margin-top: 0; color: #cf6679; }}
        </style>
    </head>
    <body>
        <h1>Auto Umpire Highlights</h1>
        
        <!-- Please copy the video to the same directory or use an absolute path -->
        <video id="videoPlayer" controls>
            <source src="{os.path.abspath(video_path)}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        
        <div class="controls">
            <div class="section">
                <h2>🏆 Đoạn Đánh Qua Lại Lâu Nhất (Longest Rally)</h2>
                <p>Số lần chạm cầu: {longest_rally['hits'] if longest_rally else 0}</p>
                <button class="btn" onclick="seekVideo({longest_rally['start_time'] if longest_rally else 0})">
                    ▶ Xem từ đầu Rally ({longest_rally['start_time'] if longest_rally else 0}s)
                </button>
                <button class="btn" onclick="seekVideo({longest_rally['end_time'] if longest_rally else 0})">
                    ▶ Tua đến cuối Rally ({longest_rally['end_time'] if longest_rally else 0}s)
                </button>
            </div>
            
            <div class="section">
                <h2>🔥 Các Pha Đập Cầu Bên Near (Top Smashes)</h2>
                <div id="smash-buttons">
    """
    
    for i, smash in enumerate(top_smashes):
        # We seek to slightly before the smash so it's easier to see
        seek_time = max(0, smash['time'] - 2.0)
        html_content += f"""
                    <button class="btn" onclick="seekVideo({seek_time})">
                        ▶ Pha {i+1} (tại {smash['time']:.2f}s - Tốc độ: {smash['speed']:.1f})
                    </button>
        """
        
    html_content += """
                </div>
            </div>
        </div>

        <script>
            function seekVideo(timeInSeconds) {
                const video = document.getElementById('videoPlayer');
                video.currentTime = timeInSeconds;
                video.play();
            }
        </script>
    </body>
    </html>
    """
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\n✅ Highlights HTML generated at: {os.path.abspath(output_html)}")
    print("Open this file in your web browser to view and click the buttons.")

if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else r"..\data\cfr\tran04_cam1.mp4"
    backend = sys.argv[2] if len(sys.argv) > 2 else "tracknet"
    out_html = sys.argv[3] if len(sys.argv) > 3 else "highlights.html"
    generate_highlights(target_video, backend, out_html)
