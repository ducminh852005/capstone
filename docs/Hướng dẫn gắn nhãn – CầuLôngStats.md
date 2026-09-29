# Hướng dẫn gắn nhãn – CầuLôngStats

Tài liệu này hướng dẫn gắn nhãn dữ liệu cho pipeline CầuLôngStats: giao điểm sân, điểm chân người chơi, vị trí quả cầu và ID theo dõi. Nhãn dùng để đo sai số thật (cm, % gọi đúng) thay cho các chỉ số proxy hiện có trong benchmark.

**Chia tập trước khi gắn nhãn.** Tham số shuttle và trọng tài đã được chỉnh trên **tran04**, nên video này không còn dùng làm tập kiểm tra được nữa.

- **Tập kiểm tra: tran03** (hoặc một trận quay mới, tốt nhất ở sân khác). Chỉ dùng để đánh giá lần cuối, không dùng để chỉnh tham số.
- **Tập phát triển: tran01, tran02, tran04.**

Chia theo trận, không chia theo đoạn cắt.

## 1. Giao điểm vạch sân

\~15 phút/video. Làm đầu tiên — dùng để đo sai số homography trước và sau khi refine.

Mỗi video lấy 1 ảnh nền sạch (chạy `calibrate_court.py --video ...`, hoặc dùng `extract_clean_background`). Chấm mọi giao điểm **nhìn thấy** của nửa sân gần, đặt tên theo tọa độ mét:

| x \\ y | 0 | 0.46 | 3.05 | 5.64 | 6.10 |
| --- | --- | --- | --- | --- | --- |
| 0 (biên cuối) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 0.76 (giao cầu dài đôi) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4.72 (giao cầu ngắn) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6.70 (lưới) | ✓ | ✓ | – | ✓ | ✓ |

Quy ước:

- Chấm vào **tâm** giao điểm của hai dải vạch, không chấm vào mép vạch.
- 4 góc đã dùng để tính H chỉ để tham khảo — đánh giá bằng các điểm còn lại.
- Lưu tại `data/labels/court/tranXX_cam1.json` dạng `{"x0.76_y0.46": [u, v], ...}`.

## 2. Điểm chân người chơi

Quan trọng nhất — dùng để đo sai số vị trí (trung vị, p95 tính bằng cm, tách theo vùng sân) và làm bảng ablation bbox → MediaPipe → làm mượt.

**Chọn frame:**

- Một nửa lấy đều, 1 frame/giây, chỉ trong lúc đang thi đấu.
- Một nửa lấy có chủ đích: frame có `source = "bbox"/"ankle"` (tin cậy thấp), bật nhảy, lunge, mất dấu, chạy nhanh.
- Không lấy frame liền nhau (cách nhau ≥ 0.5s).
- **Tên file = chỉ số frame trong video CFR**, ví dụ `tran03/004812.jpg`, để khớp với output của pipeline.
- Số lượng: **150–200 frame** trên tập kiểm tra là đủ để đánh giá. Chỉ khi quyết định fine-tune mới cần 1.000–2.000 frame.

**Mỗi người trong ảnh** (kể cả khán giả, người ở sân bên cạnh):

| Nhãn | Quy ước |
| --- | --- |
| bbox | Ôm trọn người |
| `foot` (1 điểm) | Trung điểm hai điểm tiếp đất của hai bàn chân. Nếu chỉ một chân chạm đất thì lấy điểm tiếp đất của chân đó |
| `name` | Tên người chơi (theo tên, không theo phía sân); khán giả ghi `other` |
| `airborne` | Đang bật nhảy → không chấm `foot`, bật cờ này |
| `occluded` | Bị che → chấm điểm ước lượng, bật cờ này |

Mức B (chỉ khi cần fine-tune): chấm thêm hông, gối, cổ chân, gót, mũi bàn chân. Tập trung sửa đúng các điểm này.

## 3. Quả cầu

Rẻ nếu chỉ gắn điểm rơi.

**3a. Điểm rơi, mỗi pha cầu một dòng** — đo umpire. Umpire hiện gọi 0 lần trên tran04, cần biết nó bỏ sót bao nhiêu pha.

Lưu tại `data/labels/shuttle/tranXX_landings.csv`:

```csv
rally,frame,x,y,call
1,4821,812,733,IN
2,5310,,,OFFSCREEN
```

`call` nhận một trong: `IN` / `OUT` / `UNSURE` / `OFFSCREEN` (cầu rơi ở nửa sân xa hoặc ngoài khung hình). Chấm tại **frame đầu tiên cầu chạm sàn**.

**3b. Vị trí từng frame** cho vài chục pha — đo tracker (precision/recall). Dùng đúng format TrackNet để sau này có thể train hoặc đánh giá TrackNetV3 luôn:

```csv
Frame,Visibility,X,Y
4801,1,655,402
4802,0,0,0
```

Visibility: 0 = không thấy, 1 = thấy rõ, 2 = bị che hoặc nhòe nhưng vẫn đoán được vị trí.

## 4. ID theo dõi

Nếu còn thời gian.

- Chọn vài đoạn 20–30s có người đi ngang, đứng sát biên, hoặc người chơi đổi chỗ.
- Gắn bbox + ID theo tên cho mọi frame. CVAT có nội suy track nên chỉ cần sửa keyframe.
- Dùng để đo số lần đổi ID, IDF1 và độ đúng khi gán người chơi.

## 5. Công cụ và quy trình

1. **CVAT** (khuyến nghị): có track video, point, attribute (`airborne`, `occluded`, `name`). Label Studio cũng dùng được.
2. **Nhãn sơ bộ:** guideline dùng YOLOv8x-pose để tạo nhãn sơ bộ, người gắn nhãn chỉ sửa chỗ sai. Repo hiện chỉ có `yolov8n.pt`, cần tải thêm `yolov8x-pose.pt`.
3. **Bấm giờ 50 frame đầu** để lập kế hoạch. Có nhãn sơ bộ thì ước lượng khoảng 20–40s/frame.
4. **Kiểm tra chéo:** người thứ hai xem lại ngẫu nhiên 10% số frame.
5. **Xuất ra `data/labels/`**. Không sửa file trong `data/raw/`.

## 6. Ưu tiên nếu ít thời gian

| Việc | Thời gian ước tính | Đo được |
| --- | --- | --- |
| Giao điểm sân, 4 video | \~1 giờ | Sai số homography (px/cm) |
| 150 frame điểm chân, tran03 | \~1.5–2 giờ | Sai số vị trí người chơi |
| Điểm rơi, 1 trận | \~30 phút | Độ chính xác của umpire |

Ba việc này cộng lại mất khoảng nửa ngày và đã đủ để đánh giá được cả ba phần chính của pipeline: hiệu chỉnh sân, vị trí người chơi, trọng tài tự động.
