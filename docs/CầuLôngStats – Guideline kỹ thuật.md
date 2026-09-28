# CầuLôngStats – Guideline kỹ thuật

Sep 27, 2026 · @Minh

## Mục đích, phạm vi và nguyên tắc chung

Guideline này là chuẩn làm việc chung của nhóm cho toàn bộ pipeline: từ quay video, xử lý dữ liệu, hiệu chỉnh sân, nhận diện người chơi, đến tính chỉ số, đánh giá và hiển thị.

**Phạm vi:** đánh đơn; mỗi camera phân tích người chơi ở nửa sân phía mình. Phần lõi gồm vị trí, quãng đường, heatmap và các chỉ số di chuyển. Theo dõi quả cầu, đếm lần đánh, animation 3D và monitor trực tiếp cho BLV là phần mở rộng hoặc demo.

**Nguyên tắc chung:**

- Mọi chỉ số là **ước lượng**, luôn hiển thị kèm độ tin cậy và biên sai số.
- Video gốc chỉ đọc; mọi xử lý làm trên bản sao.
- Chia dữ liệu theo trận; tập kiểm tra chỉ gồm video tự quay, không đụng tới cho đến lúc đánh giá cuối.
- Mỗi cải tiến phải đo được bằng thí nghiệm trước và sau.
- Mọi con số trong báo cáo phải chạy lại được từ script trong repo.
- Lưu sự đồng ý ghi hình; không dùng video của kênh khác khi chưa được phép; kiểm tra giấy phép trước khi dùng mã hoặc dữ liệu công khai.

## Chuẩn quay video góc chéo nửa sân

Camera đặt ngoài góc sân, trên đường chéo kéo dài của nửa sân cần phân tích, cao tối thiểu 2 m, thấy trọn nửa sân cùng khoảng đệm. Muốn có số liệu cả hai người thì đặt hai camera ở hai góc chéo đối diện nhau.

| Tiêu chí | Bắt buộc | Khuyến nghị |
| --- | --- | --- |
| Vị trí | Trên đường chéo kéo dài, ngoài góc sân của nửa sân mình | Đúng đường chéo |
| Khoảng lùi | Đủ để thấy trọn nửa sân cùng khoảng đệm | 1,5–2,5 m với ống 0.5x; 4–5 m với ống 1x |
| Chiều cao | ≥ 2 m | 2,5 m |
| Khung hình | Hai góc sân gần, hai điểm biên dọc ở vị trí lưới, khoảng đệm \~1 m sau biên cuối và hai bên, người chơi trọn vẹn từ đầu đến chân | Thấy đỉnh chữ V (giao điểm biên cuối và biên dọc gần camera) |
| Độ phân giải, tốc độ khung hình | 1080p, 30 fps | 1080p, 60 fps |
| Màn trập | Không chậm hơn 1/60 s | 1/250–1/500 s nếu đủ sáng |
| Cài đặt | Khóa phơi sáng và lấy nét; tắt chống rung điện tử, HDR, bộ lọc; quay ngang | Bật âm thanh để đồng bộ; chế độ máy bay; pin dự phòng |
| Môi trường | Vạch sân rõ, đủ sáng | Không ngược sáng, không đèn nhấp nháy, dọn túi đồ khỏi khung hình |

**Quy trình quay:**

1. Dựng máy, chụp một ảnh thử để kiểm tra khung hình.
2. Bấm quay trên mọi máy, vỗ tay một cái gần lưới để đồng bộ.
3. Quay 5–10 giây sân trống trước khi vào trận.
4. Không chạm vào máy trong trận; máy bị xê dịch thì dừng, căn lại, quay đoạn mới.
5. Nếu dùng ống 0.5x, chụp 15–20 ảnh bàn cờ bằng đúng điện thoại và ống kính đó để hiệu chỉnh méo.

**Góc sân nằm ngoài khung hình:** việc hiệu chỉnh sân vẫn chấp nhận nếu hai vạch biên thấy rõ một đoạn đủ dài (tính góc ảo từ giao điểm hai vạch kéo dài). Nhưng vùng nhìn vẫn phải thấy trọn khoảng đệm sau góc sân; nếu không, video xếp mức **Cảnh báo** và vùng góc sân đánh dấu thiếu dữ liệu.

**Kiểm tra đầu vào:** Đạt (xử lý bình thường), Cảnh báo (xử lý kèm ghi chú), Từ chối (không đủ điểm hiệu chỉnh, dưới 1080p hoặc 30 fps, camera xê dịch).

## Quản lý, chuẩn hóa và đồng bộ dữ liệu

Mọi video đi qua cùng một quy trình: sao lưu, ghi thông tin, chuyển về tốc độ khung hình cố định, đồng bộ các camera, rồi chia tập theo trận.

**Thư mục và tên file:** `raw/` (gốc, chỉ đọc), `cfr/` (đã chuẩn hóa), `frames/`, `labels/`, `metadata.csv`. Tên file theo mẫu `tranXX_camY.mp4`.

**Bảng thông tin (`metadata.csv`):** `file, tran, camera, chieu_cao_m, khoang_lui_m, ong_kinh, fps, do_phan_giai, thoi_luong, loai_tran, nguoi_choi, dong_y, do_lech_dong_bo_s, ghi_chu`. Điền ngay sau buổi quay.

**Chuẩn hóa tốc độ khung hình** (điện thoại thường quay VFR):

```bash
ffmpeg -i raw/tran01_cam1.mp4 -vf fps=60 -c:v libx264 -crf 18 -preset medium -c:a aac cfr/tran01_cam1.mp4
```

**Đồng bộ nhiều camera:** dùng `audio-offset-finder` (BBC, Apache-2.0) để tìm độ lệch từ tiếng vỗ tay và âm thanh sân. Chấp nhận khi điểm chuẩn > 10; dưới 5 thì kiểm tra thủ công. Lưu độ lệch vào `do_lech_dong_bo_s`.

```bash
pip install audio-offset-finder
audio-offset-finder --find-offset-of cam2.wav --within cam1.wav
```

**Kiểm tra chất lượng:** trích 1 ảnh mỗi 10 giây (`-vf fps=1/10`), xem lướt để tìm đoạn mất người, bị che; chồng ảnh đầu và cuối để phát hiện camera xê dịch.

**Chia tập:** theo trận, không theo đoạn cắt. Ví dụ 4 trận: 3 trận phát triển, 1 trận kiểm tra; nếu được, tập kiểm tra có ít nhất một sân khác.

## Gắn nhãn

Gắn nhãn trên các frame được chọn lọc, bắt đầu từ nhãn sơ bộ của mô hình lớn, và gắn đủ mọi người trong ảnh.

**Chọn frame:** khoảng một nửa lấy đều (1 frame/giây trong lúc thi đấu), một nửa lấy có chủ đích từ lần chạy mô hình pretrained: cổ chân kém tin cậy, vị trí nhảy bất thường, mất dấu, vận tốc cao. Không lấy các frame liền nhau.

**Quy trình:**

1. Chạy YOLOv8x-pose tạo nhãn sơ bộ.
2. Nhập vào Label Studio hoặc CVAT; người gắn nhãn chỉ sửa chỗ sai, tập trung vào hông, gối, cổ chân (và gót, mũi chân nếu dùng MediaPipe).
3. Người thứ hai xem lại ngẫu nhiên 10% số frame.
4. Xuất định dạng YOLO pose, chia tập theo trận.

**Quy ước gắn nhãn:**

- **Mọi người trong ảnh đều gắn nhãn**, kể cả người chơi phía xa, khán giả, người ở sân bên cạnh. Bỏ sót sẽ dạy mô hình rằng họ không phải người.
- **Điểm chân:** trung điểm hai điểm tiếp đất của hai bàn chân.
- **Bật nhảy:** không đánh điểm chân, gắn cờ "trên không".
- **Bị che:** đánh điểm ước lượng, gắn cờ "bị che".
- **ID:** theo tên người chơi, không theo phía sân.
- **Giao điểm vạch sân:** đánh cho mọi giao điểm nhìn thấy, theo bảng tọa độ ở phần hiệu chỉnh sân. Có thể dùng công cụ gắn nhãn điểm mốc của repo CourtKeyNet.

**Số lượng gợi ý:** tư thế 1.000–2.000 frame đa dạng; điểm mốc sân vài trăm ảnh từ nhiều sân và góc khác nhau; quả cầu vài chục pha cầu. Bấm giờ 50 frame đầu để lập kế hoạch (ước lượng 20–40 giây mỗi frame khi có nhãn sơ bộ).

## Hiệu chỉnh và tinh chỉnh khung sân

Khung sân được dò một lần trên ảnh nền sạch của mỗi video, tinh chỉnh bằng vạch trắng, rồi kiểm tra hợp lệ trước khi dùng; người dùng luôn có thể kéo chỉnh.

**Hệ tọa độ sân** (gốc ở một góc sân; x theo chiều dài 0–13,40 m, y theo chiều rộng 0–6,10 m):

| Vạch theo chiều dài (x, m) | Vạch theo chiều rộng (y, m) |
| --- | --- |
| 0 và 13,40: biên cuối | 0 và 6,10: biên dọc đánh đôi |
| 0,76 và 12,64: giao cầu dài đánh đôi | 0,46 và 5,64: biên dọc đánh đơn |
| 4,72 và 8,68: giao cầu ngắn | 3,05: vạch giữa |
| 6,70: vị trí lưới (không có vạch) |  |

Hệ tọa độ nửa sân của người chơi: 0 ở biên cuối, 6,70 m ở lưới; khi vẽ, lưới luôn ở trên. Mô hình khai báo **tâm dải vạch** (vạch rộng \~4 cm, lệch 2 cm so với mép) và dò tâm dải trên ảnh.

**Các bước:**

1. **Hiệu chỉnh méo ống kính** bằng `cv2.calibrateCamera` từ ảnh bàn cờ; mọi bước sau làm trên ảnh đã hiệu chỉnh.
2. **Ảnh nền sạch:** trung vị của khoảng 60 frame rải khắp video để xóa người chơi.
3. **Homography thô:** chấm tay 4 góc nửa sân (prototype), sau thay bằng CourtKeyNet hoặc dò cổ điển (Hough + RANSAC) neo vào hình chữ V ở góc sân gần camera.
4. **Tách vạch trắng:** phép top-hat trên ảnh xám rồi ngưỡng Otsu; che mọi thứ ngoài nửa sân gần cộng khoảng đệm để loại vạch sân bên cạnh.
5. **Tinh chỉnh:** chuyển nửa sân gần sang ảnh nhìn từ trên xuống, dò vạch ngang và dọc bằng cộng dồn pixel trắng theo hàng và cột, tính homography hiệu chỉnh và nhân vào homography thô. Chỉ dùng vạch nửa sân gần.
6. **Góc sân ngoài khung:** khớp đường thẳng cho hai vạch biên bằng RANSAC, tính giao điểm bằng tích có hướng tọa độ thuần nhất; đưa góc ảo vào như điểm phụ. Hướng nâng cao: ước lượng homography trực tiếp từ các vạch (đường thẳng).
7. **Kiểm tra hợp lệ:** sai số tái chiếu trên giao điểm không dùng để tính, và tỉ lệ pixel vạch mô hình rơi đúng vào mặt nạ vạch trắng. Không đạt ngưỡng → Cảnh báo và mở màn hình kéo chỉnh. Chạy lại tỉ lệ trùng khớp vài phút một lần để phát hiện camera xê dịch.

Mô hình dò điểm mốc không tự biết khi góc quay nằm ngoài dữ liệu nó từng học, và vẫn trả về một tứ giác gọn gàng nhưng sai; vì vậy bước 7 là bắt buộc. Ngưỡng cụ thể chốt sau thí nghiệm tuần 5–6.

## Nhận diện, theo dõi và gán người chơi

YOLOv8 và ByteTrack phát hiện, theo dõi mọi người; bộ lọc vùng nửa sân và luật điểm người chơi chọn ra đúng một người chơi; điểm chân lấy từ điểm khớp cơ thể.

**Phát hiện và theo dõi:**

```python
model = YOLO("yolov8m-pose.pt")
results = model.track(video, tracker="bytetrack.yaml", stream=True, classes=[0], conf=0.3)
```

Thử xử lý 30 khung/giây trên video 60 fps; đo xem quãng đường có đổi đáng kể không.

**Điểm chân** (thứ tự ưu tiên):

1. **MediaPipe Pose Landmarker** (bản Full hoặc Heavy) chạy trên **vùng cắt quanh người chơi đã chọn**, không chạy trên cả khung hình. Lấy trung điểm gót (29, 30) và mũi bàn chân (31, 32) của chân đang chạm đất.
2. **YOLOv8-pose:** trung điểm hai cổ chân khi độ tin cậy > \~0,5.
3. **Dự phòng:** điểm giữa cạnh dưới bounding box, gắn cờ độ tin cậy thấp.

Chiếu điểm chân sang tọa độ sân bằng `cv2.perspectiveTransform`. Ba cách lấy điểm chân là ba cấu hình trong bảng ablation.

**Gán người chơi:**

- **Lọc vùng:** chỉ giữ người có điểm chân trong nửa sân gần, từ \~1 m sau biên cuối đến vạch lưới, khoảng đệm hông hẹp ở gần lưới để loại người đứng cạnh cột lưới.
- **Điểm người chơi** cho mỗi track: tỉ lệ thời gian trong sân trong cửa sổ 10–20 giây. Tối đa một người chơi mỗi nửa sân; nhiều ứng viên thì chọn điểm cao nhất.
- **Độ trễ trạng thái:** đã là người chơi thì giữ ID khi bước ra ngoài vạch ngắn; chỉ bỏ khi vắng lâu.
- **Ghép mảnh:** track mới xuất hiện cùng nửa sân sau khi mất dấu ngắn → gán lại ID cũ.
- **Đổi sân:** ghép số liệu theo tên người chơi và mốc đổi sân; người dùng xác nhận tên một lần.

**Fine-tune:** chỉ khi thử nghiệm cho thấy cần. Đo mAP và sai số cổ chân của mô hình pretrained trên 100–200 frame đầu; nếu sai số nhỏ so với sai số homography thì không fine-tune. Nếu fine-tune: giữ phần lớn mô hình, tốc độ học nhỏ, tăng cường dữ liệu (độ sáng, nhòe, lật ngang), trộn thêm ảnh COCO keypoints để tránh quên. Fine-tune chỉ cải thiện khả năng nhìn, không sửa được sai số do góc quay.

## Làm sạch quỹ đạo, chỉ số và heatmap

Quỹ đạo được làm sạch trước khi tính bất cứ chỉ số nào; mọi chỉ số tính trên hệ tọa độ nửa sân của người chơi và chỉ trong các đoạn hợp lệ.

**Làm sạch** (theo thứ tự):

1. Loại điểm ngoài sân quá khoảng đệm hoặc có vận tốc > \~7 m/s.
2. Khoảng mất dấu < 0,5 s: nội suy tuyến tính. Dài hơn: để trống, **không cộng quãng đường qua đó**.
3. Làm mượt bằng Savitzky–Golay (cửa sổ 0,3–0,5 s, bậc 2) hoặc Kalman vận tốc không đổi; so hai cách trong ablation.

**Chỉ số:**

| Nhóm | Chỉ số | Cách tính |
| --- | --- | --- |
| Khối lượng | Quãng đường | Tổng dịch chuyển giữa hai khung hợp lệ liên tiếp; tổng, trong lúc thi đấu, theo ván, trên mỗi phút |
| Cường độ | Vận tốc | Trung bình và phân vị 95 (không lấy giá trị lớn nhất) |
| Cường độ | Phân bố mức vận tốc | % thời gian đứng, đi, chạy, bứt tốc; ngưỡng 0,5 / 2 / 4 m/s là tham khảo ban đầu |
| Vùng hoạt động | Phân bố 6 vùng | Trước, giữa, cuối sân × phía tay thuận, phía trái tay |
| Vùng hoạt động | Mức bao quát | % ô có mặt trên một ngưỡng thời gian |
| Thói quen | Vị trí chờ | Trung tâm các vị trí gần như đứng yên trong pha cầu |
| Thói quen | Số lần và hướng di chuyển | Mỗi đoạn vận tốc vượt ngưỡng; phân loại lên lưới, lùi cuối, sang ngang |
| Thói quen | Thời gian trở về | Từ điểm xa nhất của một lần di chuyển đến khi về gần vị trí chờ |
| Độ tin cậy | Tỉ lệ dữ liệu hợp lệ | % thời gian có vị trí tin cậy; luôn hiển thị |

**Heatmap:**

- Vùng vẽ: sâu −1 → 6,70 m, ngang −0,5 → 6,60 m; ô 0,25 m; mỗi khung hợp lệ cộng 1/fps giây; chuẩn hóa thành % thời gian.
- Làm mờ Gaussian với bán kính xấp xỉ **sai số vị trí đo được** (\~0,2–0,3 m), để không thể hiện độ chi tiết hệ thống không đo được.
- Vẽ vạch sân chồng lên; bảng màu viridis hoặc inferno; gạch chéo vùng kém tin cậy (sát góc sân gần camera, vùng ít dữ liệu, góc sân ngoài khung).
- Hai phiên bản: toàn trận và chỉ trong lúc thi đấu.

**Thời gian nghỉ:** người chơi di chuyển < \~1 m/s liên tục vài giây, hoặc ra ngoài sân. Có hai camera thì kết hợp cả hai người. Kiểm chứng bằng vài đoạn gắn nhãn tay.

**Riêng góc chéo:** di chuyển dọc hướng nhìn của camera có sai số lớn hơn; báo cáo sai số quãng đường tách theo hướng. Không đưa "vị trí chờ lý tưởng" cố định để chấm điểm; so sánh giữa các trận của chính người chơi.

## Đánh giá

Mỗi khâu có một thước đo riêng trên tập kiểm tra; kết quả trình bày bằng bảng số kèm một hình và một đoạn nhận xét: sai số có chấp nhận được cho người chơi phong trào không, và ở vùng nào thì không.

| Khâu | Thước đo | Dữ liệu chuẩn |
| --- | --- | --- |
| Homography | Sai số tái chiếu (cm), trung bình và lớn nhất; trước và sau tinh chỉnh | Giao điểm vạch sân không dùng để tính |
| Dò khung sân tự động | Tỉ lệ video thành công, sai số pixel giao điểm | Nhãn tay giao điểm, nhiều sân |
| Phát hiện người, điểm khớp | mAP, sai số pixel cổ chân (hoặc gót, mũi chân) | Frame gắn nhãn |
| Vị trí người chơi | Trung vị và phân vị 95 (cm), tách theo vùng sân; vẽ bản đồ sai số trên sân | Điểm chân gắn nhãn tay |
| Quãng đường | Sai số tương đối (%), tách theo hướng di chuyển | Đoạn đi theo đường đo sẵn, kể cả dọc đường chéo |
| Theo dõi | Số lần đổi ID, IDF1 (`py-motmetrics` hoặc TrackEval) | Đoạn gắn nhãn ID |
| Gán người chơi | Độ chính xác, độ phủ phân loại track; số frame gán nhầm | Đoạn có người đi ngang, đứng sát biên |
| Kiểm chứng chéo | Độ lệch vị trí và quãng đường giữa hai camera | Hai camera quay cùng lúc |

**Bảng ablation** (chạy trên tập kiểm tra, đo sai số vị trí và quãng đường): cơ sở (cạnh dưới bounding box, không làm mượt) → + cổ chân YOLOv8-pose → + gót và mũi chân MediaPipe → + làm mượt → + lọc vận tốc và bỏ khoảng mất dấu → + lọc vùng sân.

**Thí nghiệm chuẩn quay:** so các cấu hình (góc chéo và giữa cạnh dọc; cao 1,5 / 2 / 2,5 m) theo sai số tái chiếu và sai số vị trí. Kết quả là căn cứ cho chuẩn quay bản cuối.

## Hệ thống, hiển thị và monitor BLV

Hệ thống phân tích sau trận dùng FastAPI, PostgreSQL và React; monitor trực tiếp cho BLV là luồng riêng, chỉ để xem, không dùng để phân tích.

**Kiến trúc:** FastAPI + hàng đợi xử lý nền; PostgreSQL lưu trận, nguồn video theo camera (kèm độ lệch đồng bộ và file hiệu chỉnh), người chơi, bảng vị trí theo frame, chỉ số, mốc đổi sân; object storage lưu video; React cho giao diện; FFmpeg cắt clip.

**Luồng người dùng:** upload (và ảnh thử) → kiểm tra đầu vào → xác nhận khung sân (kéo chỉnh nếu cần) → đặt tên người chơi, tay thuận → xem kết quả.

**Trang kết quả mỗi trận:**

1. Thẻ tóm tắt: quãng đường, vận tốc, số lần di chuyển, tỉ lệ dữ liệu hợp lệ, kèm biên sai số.
2. Heatmap nửa sân với vạch sân và vùng kém tin cậy.
3. Biểu đồ 6 vùng và biểu đồ hoa gió hướng di chuyển.
4. **Phát lại quỹ đạo 2D** trên sơ đồ sân, đồng bộ với video (animation mức A, không cần AI).
5. Xu hướng qua các trận.

**Animation:** mức A (quỹ đạo 2D) nằm trong prototype. Mức B (khung xương 3D từ tọa độ 3D của MediaPipe, hiển thị bằng three.js) là điểm cộng. Mức C (avatar 3D bằng FreeMoCap, WHAM, 4D-Humans hoặc dịch vụ thương mại) chỉ thử trên một pha cầu mẫu; lưu ý giấy phép SMPL giới hạn phi thương mại.

**Monitor BLV:** mỗi điện thoại mở VDO.Ninja trên trình duyệt (kết nối bằng mã QR), OBS Studio nhận các luồng làm nguồn Browser và xếp thành lưới hoặc dùng Multiview.

- Điện thoại **vẫn tự ghi video gốc chất lượng đầy đủ**; luồng qua Wi-Fi chỉ để xem.
- Các luồng trực tiếp lệch nhau vài trăm mili giây; không dùng để đồng bộ phân tích.
- Kiểm tra mạng tại sân trước; VDO.Ninja cần Internet để thiết lập kết nối.
- Chỉ số hiển thị sau mỗi pha cầu hoặc mỗi ván, không theo thời gian thực.

## Phụ lục: công cụ và repo tham khảo

Các repo cầu lông dưới đây phần lớn là dự án cá nhân hoặc hackathon: dùng làm tham khảo và baseline, kiểm tra giấy phép trước khi dùng mã (không ghi giấy phép thì không sao chép), và trích dẫn trong báo cáo.

| Công cụ hoặc repo | Khâu | Dùng để |
| --- | --- | --- |
| [badminton-pipeline-repro](https://github.com/ychenfen/badminton-pipeline-repro) | Toàn pipeline | Chạy thử ngay để có baseline (TrackNet, YOLOv8s-pose, ByteTrack, homography từ 4 điểm nhập tay) |
| [badminton-tracker](https://github.com/55555bbbbbbb/badminton-tracker) | Hiệu chỉnh sân, gán ID | Tham khảo quy trình tinh chỉnh homography bằng vạch và quản lý danh tính người chơi |
| [CourtKeyNet](https://github.com/adithyanraj03/CourtKeyNet) | Dò khung sân | Mô hình điểm mốc sân cầu lông, bộ dữ liệu và công cụ gắn nhãn điểm mốc |
| [badminton\_shot\_type](https://github.com/RichardPinter/badminton_shot_type) | Mở rộng | TrackNetV3, RTMPose, phân loại cú đánh |
| [amdhacks](https://github.com/thecodemonki/amdhacks) | Hệ thống | Tham khảo stack FastAPI + React, heatmap, tách lần đánh bằng âm thanh |
| [MediaPipe Pose Landmarker](https://ai.google.dev/mediapipe/solutions/vision/pose_landmarker) | Điểm khớp | 33 điểm mốc có gót và mũi bàn chân, tọa độ 3D |
| Ultralytics YOLOv8 + ByteTrack | Phát hiện, theo dõi | Mô hình chính của phần lõi |
| [audio-offset-finder](https://github.com/bbc/audio-offset-finder) | Đồng bộ | Độ lệch giữa các camera từ âm thanh, Apache-2.0 |
| [VDO.Ninja](https://github.com/steveseguin/vdo.ninja) + OBS Studio | Monitor BLV | Đưa camera điện thoại vào một màn hình |
| [FreeMoCap](https://freemocap.org/) | Animation mức C | Bắt chuyển động không marker, xuất FBX, .blend, CSV |
| Label Studio, CVAT | Gắn nhãn | Gắn nhãn bounding box, điểm khớp, ID |
| py-motmetrics, TrackEval | Đánh giá | IDF1, số lần đổi ID |
| TrackNetV3 | Mở rộng | Theo dõi quả cầu, có trọng số huấn luyện sẵn |

Repo và công cụ được tra cứu ngày 27/9/2026; nên kiểm tra lại tình trạng và giấy phép trước khi dùng.
