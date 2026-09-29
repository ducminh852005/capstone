import os
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.shuttle_tracker import ShuttleDetector
from core.video_io import ThreadedVideoReader
from core import court_model
from core.umpire import RallyUmpire

def test_trajectory(video_path, backend="tracknet"):
    print(f"Opening video: {video_path}")
    reader = ThreadedVideoReader(video_path)
    
    try:
        detector = ShuttleDetector(backend=backend)
    except FileNotFoundError as e:
        if backend != "tracknet":
            raise
        print(f"WARNING: {e}\nFalling back to backend='cv'.")
        detector = ShuttleDetector(backend="cv")
        
    print(f"Shuttle backend: {detector.backend}")
    print("Starting processing... Press 'q' to quit.")
    
    # Khởi tạo Trọng tài (Umpire) để phát hiện điểm rơi
    H, H_inv = court_model.load_calibration()
    umpire = None
    if H_inv is not None:
        w, h = reader.size
        roi = court_model.shuttle_roi(H, (h, w))
        
        # Tính rest_frames: với stride nhỏ (5), việc yêu cầu 6 frame đứng im là quá khó
        # vì VĐV thường sẽ nhặt cầu lên ngay lập tức sau 1-2 giây.
        # Ta fix cứng rest_frames = 2 (hoặc 3) để dễ dàng bắt được trạng thái dừng.
        stride = getattr(detector._source, 'batch_stride', 15) if hasattr(detector, '_source') else 15
        rest_frm = 3 if stride < 15 else 2
        
        umpire = RallyUmpire(H_inv=H_inv, match_type="singles", 
                             frame_size=(h, w), roi=roi,
                             net_top_y=court_model.net_top_threshold_y(H),
                             rest_frames=rest_frm)
        print(f"Umpire initialized. Adjusted for stride={stride} (frames={rest_frm})")
    else:
        print("WARNING: No calibration data, Umpire disabled.")
        
    bounce_events = []
    
    for frame_idx, frame in reader:
        # Nhận diện quả cầu (có kèm ROI để AI tự ngắt track khi cầu rớt ra ngoài biên)
        pt, mask = detector.detect(frame, roi=roi if umpire is not None else None)
        
        # Cho trọng tài theo dõi để bắt điểm rơi
        if umpire is not None:
            call = umpire.update(frame_idx, pt, detector.track_active)
            if call is not None:
                bounce_events.append(call)
                print(f"\n💥 [CẦU RƠI] Frame {call.frame_idx}: Tọa độ ảnh {call.image_pt}, Tọa độ sân (m) {call.world_pt}")
                print(f"👉 Kết quả: {call.result} (Half: {call.half}, Margin: {call.margin_m:.3f}m)\n")
                
        # Log ra terminal để dễ theo dõi
        if pt is not None:
            print(f"Frame {frame_idx}: Shuttle detected at {pt}")
        elif frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: Processing... (no shuttle detected)")
            
        # VẼ QUỸ ĐẠO THÔNG MINH
        recent_points = [p for p in detector.trajectory[-60:] if p is not None]
        
        if recent_points:
            # Dùng trực tiếp 'frame' để vẽ (không cần .copy() gây tốn RAM/CPU)
            for i in range(1, len(recent_points)):
                pt1, pt2 = recent_points[i-1], recent_points[i]
                t = i / len(recent_points)
                cv2.line(frame, pt1, pt2, (0, int(80 + 175 * t), 255), int(1 + 3 * t))
            
            # Vẽ chấm đỏ ở vị trí đầu cầu
            cv2.circle(frame, recent_points[-1], 6, (0, 0, 255), -1)
            
        # Vẽ các vị trí cầu rơi đã được trọng tài xác nhận (trên video)
        for event in bounce_events:
            color = (0, 255, 0) if event.result == "IN" else (0, 0, 255)
            cv2.drawMarker(frame, event.image_pt, color, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
            cv2.putText(frame, f"{event.result}", (event.image_pt[0] + 15, event.image_pt[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        # Vẽ khung sân 3D lên video
        if H is not None:
            # Danh sách các vạch sân đầy đủ
            court_lines = [
                # Ngang
                ((0,0), (0,6.1)), ((0.76,0), (0.76,6.1)), ((4.72,0), (4.72,6.1)), ((6.7,0), (6.7,6.1)),
                ((8.68,0), (8.68,6.1)), ((12.64,0), (12.64,6.1)), ((13.4,0), (13.4,6.1)),
                # Dọc
                ((0,0), (13.4,0)), ((0,0.46), (13.4,0.46)), ((0,5.64), (13.4,5.64)), ((0,6.1), (13.4,6.1)),
                # Giữa
                ((0,3.05), (4.72,3.05)), ((8.68,3.05), (13.4,3.05))
            ]
            for (w1, w2) in court_lines:
                pts = court_model.world_to_img([w1, w2], H)
                pt1, pt2 = (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
                
        frame_resized = cv2.resize(frame, (1080, 720))
        
        # Tạo Minimap 2D (Sân nhìn từ trên xuống)
        import numpy as np
        map_w, map_h = 360, 720
        minimap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        minimap[:] = (0, 100, 0) # Nền xanh
        
        # Scale: 1 mét = 45 pixel
        scale = 45
        offset_x = (map_w - 6.1 * scale) / 2
        offset_y = (map_h - 13.4 * scale) / 2
        
        def w2m(x, y):
            # Tọa độ x của sân (chiều dài) -> y trên minimap, y sân (rộng) -> x minimap
            px = int(y * scale + offset_x)
            py = int(map_h - (x * scale + offset_y))
            return (px, py)
            
        # Vẽ sân trên minimap
        if H is not None:
            for (w1, w2) in court_lines:
                cv2.line(minimap, w2m(*w1), w2m(*w2), (255, 255, 255), 2)
            
        # Vẽ các điểm rơi trên minimap
        for event in bounce_events:
            color = (0, 255, 0) if event.result == "IN" else (0, 0, 255)
            pt = w2m(event.world_pt[0], event.world_pt[1])
            cv2.circle(minimap, pt, 6, color, -1)
            cv2.circle(minimap, pt, 12, color, 2)
            
        # Ghép video và minimap
        display_img = np.hstack((frame_resized, minimap))
        
        # Hiển thị trực tiếp lên màn hình
        cv2.imshow("Test Trajectory", display_img)
        
        # Xử lý phím tắt
        key = cv2.waitKey(0 if getattr(detector, 'paused', False) else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Nhấn Space để Tạm dừng / Tiếp tục
            detector.paused = not getattr(detector, 'paused', False)
            state = "PAUSED" if detector.paused else "RESUMED"
            print(f"[{state}] ở Frame {frame_idx}")
        elif key == ord('s'):  # Nhấn 's' để báo cáo sót điểm rơi
            print(f"🚨 [MISS REPORT] Đã ghi nhận sót điểm rơi quanh Frame {frame_idx}")
            with open("missed_bounces.txt", "a", encoding="utf-8") as f:
                f.write(f"Missed bounce around frame {frame_idx}\n")

    reader.release()
    cv2.destroyAllWindows()
    print("Test finished.")

if __name__ == "__main__":
    target_video = sys.argv[1] if len(sys.argv) > 1 else r"..\data\cfr\tran04_cam1.mp4"
    backend = sys.argv[2] if len(sys.argv) > 2 else "tracknet"
    test_trajectory(target_video, backend)
