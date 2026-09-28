# Tối ưu hiệu năng pipeline (TrackNet / YOLO / Pose) – Nhật ký kỹ thuật

> Phạm vi: toàn bộ phiên làm việc điều tra & tối ưu hiệu năng pipeline tracking (shuttle + player) trên phần cứng tham chiếu: **NVIDIA T550 Laptop GPU (4GB VRAM)**, **Intel i7-1265U (10 nhân/12 luồng)**, **32GB RAM**, Windows.
>
> Video test dùng xuyên suốt: `data/cfr/tran04_cam1.mp4` (1800 frame), so sánh qua `backend/scripts/benchmark_pipeline.py`.

---

## 1. Bối cảnh ban đầu

Pipeline dùng YOLOv8n + ByteTrack (theo dõi người chơi), TrackNetV3 (định vị cầu, CNN heatmap regression), MediaPipe Pose (điểm chạm chân), Kalman filter + state machine (auto umpire). Vấn đề khởi phát: chạy `test_shuttle_tracker.py` báo lỗi thiếu `cv2`, và sau khi cài xong, phát hiện **torch đang chạy CPU thay vì GPU** dù máy có GPU rời.

### Fix 1 — torch chạy CPU thay vì GPU
- Nguyên nhân: `pip install -r requirements.txt` cài bản `torch` mặc định trên PyPI, vốn là **build CPU-only**. Ghi chú này đã có sẵn trong `backend/requirements.txt` (torch cố tình không liệt kê trong file để tránh bị ghi đè bản CUDA).
- Fix: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` (bắt buộc `--force-reinstall` vì pip coi version số giống nhau là "đã thỏa mãn" dù khác build tag `+cpu` / `+cu121`).
- Kết quả: torch từ `2.14.0+cpu` → `2.5.1+cu121`, `torch.cuda.is_available()` = True, GPU T550 Laptop được nhận diện.

**Đây chính là nguyên nhân gián tiếp gây ra bug lớn nhất phiên này — xem mục 4.**

---

## 2. Vì sao FPS thấp / vì sao GPU-Util hiển thị thấp

Trước khi tối ưu, đã điều tra kỹ nguyên nhân FPS thấp (7–16 fps tuỳ cấu hình) và vì sao Task Manager cho thấy GPU dùng "chưa tới 10%" dù rõ ràng đang là bottleneck.

- **Kiến trúc pipeline hoàn toàn tuần tự**: mỗi frame chạy lần lượt YOLO (GPU) → TrackNet shuttle-detect (GPU) → MediaPipe pose (CPU) → umpire/vẽ (CPU). Không có bước nào chạy song song với bước khác.
- **Task Manager mặc định xem sai engine**: tab Performance > GPU hiển thị engine **"3D"** theo mặc định, trong khi khối lượng tính toán CUDA (torch inference) chạy trên engine **"Compute_0"/"Cuda"** — một engine riêng không hiện trên graph mặc định. Đo lại bằng `nvidia-smi dmon -s u` cho thấy SM-util thực tế dao động **18–45%**, không phải <10%.
- **GPU T550 chạy đúng công suất, không throttle**: `nvidia-smi` cho thấy P-State P0, SM clock ~87% max boost, nhiệt độ 68°C (an toàn), không có throttle-reason nào được set. Nghĩa là tốc độ chậm là do bản chất con chip 30W yếu + pipeline batch=1 (latency-bound), không phải do cấu hình/driver/nhiệt độ.
- Batch=1 (xử lý từng frame một, không gộp batch) khiến GPU không bao giờ được "lấp đầy" song song dù nó là nơi tốn thời gian nhất — đặc điểm chung của mọi pipeline video real-time single-stream.

---

## 3. Thử nghiệm tối ưu #1 — Threading overlap (pose CPU // shuttle GPU)

**Ý tưởng**: MediaPipe pose (CPU, ~16ms/frame trung bình) đang chặn hoàn toàn trước khi TrackNet shuttle-detect (GPU, ~55ms/frame lúc đó) chạy. CPU có 10 core rảnh trong lúc GPU bận — dùng `ThreadPoolExecutor` chạy pose song song với shuttle-detect.

**Triển khai**: `PlayerTracker.update_async()`/`join()` (thread nền), shuttle detector dùng player-selection của **frame trước** (lệch 1 frame, chấp nhận được) làm exclude/prior boxes để tách phụ thuộc dữ liệu.

**Kết quả đo thật (1800 frame)**:

| | tuần tự (baseline) | thread overlap |
|---|---|---|
| wall_fps | 9.67 | 9.58 (**không cải thiện**) |
| shuttle ms/frame (gồm `join()`) | 55.11 | 74.68 |
| pose ms/frame | 16.40 | 14.31 |
| pct_frames_with_player / pose_calls | 90.1% / 547 | 90.1% / 547 (giống hệt — đúng logic) |

**Nguyên nhân thất bại**: Python GIL. Hai thread dùng chung 1 GIL — MediaPipe (C++) không nhả GIL liên tục, khiến main thread (đang chuẩn bị tensor/gọi CUDA/sync `.cpu()`) phải xếp hàng chờ GIL. Phần lợi từ "chạy song song" bị ăn hết bởi chi phí tranh chấp GIL.

**Kết luận**: revert hoàn toàn, không giữ trong code chính (đã git checkout về bản gốc).

---

## 4. Thử nghiệm tối ưu #2 — Multiprocessing overlap (mỗi process 1 GIL riêng)

**Ý tưởng tiếp theo**: chạy MediaPipe trong 1 **process** riêng (không phải thread) để có GIL độc lập, giải quyết triệt để vấn đề mục 3. Dùng `multiprocessing.shared_memory` để truyền frame qua lại giữa 2 process mà không tốn chi phí copy qua pickle mỗi lần.

**Triển khai**: tách `PoseUpdateEngine` (logic pose + player-selection) ra khỏi `PlayerTracker` để dùng lại được cả ở process chính lẫn worker process; `MultiprocessPoseEngine` (`core/pose_process.py`) quản lý shared-memory buffer + `Pipe` + worker process bền vững.

**Kết quả đo thật (1800 frame, chạy 2 lần mỗi cấu hình để loại nhiễu)**:

| | baseline | thread | **process** |
|---|---|---|---|
| wall_fps (2 lần) | 9.67 | 9.58 | **9.38, 8.44** (chậm hơn cả 2 cái trên) |
| pct_frames_with_player / pose_calls | 90.1% / 547 | 90.1% / 547 | **86.8% / 539** (2 lần chạy process giống hệt nhau) |
| distinct_player_ids | 3 | 3 | **2** |

Rerun `thread`/baseline xác nhận pipeline này **hoàn toàn deterministic** (không có nhiễu ngẫu nhiên tự nhiên) — nghĩa là sai lệch của "process" là **có hệ thống, lặp lại được**, không phải race condition ngẫu nhiên (2 lần chạy process cho cùng 1 con số y hệt).

**2 vấn đề của multiprocessing**:
1. **Không nhanh hơn** — chi phí spawn process, import mediapipe/numpy trong worker, và bắt tay qua shared-memory/pipe ăn hết lợi ích bỏ GIL, trong khi tần suất gọi mediapipe chỉ ~1 lần/3 frame.
2. **Kết quả detect lệch thật** dù `PoseUpdateEngine.update()` chạy đúng y hệt code, cùng input `(frame, boxes, ids, H_inv)` — nghi vấn (chưa xác nhận chắc chắn): có 1 process phụ đang chạy song song có thể làm nhiễu autotune cuDNN của YOLO (chọn thuật toán conv khác dưới điều kiện tải hệ thống khác → sai số làm tròn khác → lật vài quyết định biên).

**Kết luận**: revert hoàn toàn (đã git checkout về bản gốc). Cả threading lẫn multiprocessing đều **không đáng dùng** trên phần cứng này.

---

## 5. Fix 2 (quan trọng nhất phiên này) — TrackNet chạy FP16 bị lỗi nghiêm trọng

Trong lúc benchmark các thử nghiệm ở mục 3–4, phát hiện **mọi lần chạy full 1800-frame của session này đều cho `pct_frames_detected = 0.0%`** — trong khi file benchmark baseline có sẵn từ trước (`tracknet.json`) ghi nhận **3.2%**. Cùng 1 đoạn video, cùng code (đã revert y hệt) — nhưng TrackNet không detect được cầu lần nào.

### Điều tra & xác nhận root cause

| Cấu hình | % detect | non-finite batch |
|---|---|---|
| FP16 (mặc định cũ, sau khi fix torch ở mục 1) | 0% | **~81% batch** (30/37) |
| FP32 ép chạy trên cùng GPU | **3.6%** (khớp/nhỉnh hơn baseline 3.2%) | 0 |
| FP16 + tắt `cudnn.benchmark` (loại trừ do chọn sai thuật toán autotune) | vẫn 0% | vẫn ~81% |

→ **FP16 tự nó tràn số (NaN) trên chính model + GPU này**, không liên quan tới cuDNN autotune. Code sẵn có 1 guard "coi như không detect" khi gặp NaN (không crash, không báo lỗi rõ ràng) — khiến bug này **âm thầm tồn tại suốt session** mà pipeline vẫn chạy "bình thường" không lỗi.

### Phát hiện bất ngờ: FP32 còn NHANH HƠN FP16 trên GPU này

| | FP16 (hỏng) | FP32 (đúng) |
|---|---|---|
| ms/window (raw forward pass) | 364 | **103** (nhanh hơn 3.5x) |
| wall_fps (full pipeline) | 9.67 | **15.97** (+65%) |
| shuttle ms/frame | 55.11 | **21.25** |
| pct_frames_detected | 3.2% (số cũ, môi trường khác) | 3.6% |

T550 (GPU mobile giá rẻ) rõ ràng **không có phần cứng FP16/Tensor Core throughput tốt** — chạy FP16 vừa chậm hơn vừa tràn số. Đây không phải trade-off tốc độ/độ chính xác — FP32 thắng cả 2 mặt.

### Fix đã áp dụng
- `backend/core/tracknet.py`: `TrackNetCandidateSource.__init__` thêm tham số `use_half` (mặc định **`False`**), docstring ghi rõ lý do + số đo.
- `backend/scripts/bench_tracknet_forward.py`: đồng bộ mặc định `use_half = False`.
- Xoá benchmark cũ đo dưới điều kiện FP16 hỏng (`tracknet_stride4.json`) vì số liệu không phản ánh đúng bản chất (do bug, không phải do stride).

**Đây là fix mang lại lợi ích thật lớn nhất và duy nhất "miễn phí" (không đánh đổi gì) của cả phiên làm việc.**

---

## 6. Đánh giá 4 hướng tối ưu tiếp theo (do người dùng đề xuất)

Sau fix FP16, người dùng đề xuất 4 hướng tối ưu thêm (tham khảo từ nguồn ngoài). Đánh giá đối chiếu với code thật:

| # | Đề xuất | Đánh giá |
|---|---|---|
| 1 | TensorRT INT8 (dùng repo `nickluo/TrackNetV3`) | Repo vendor trong dự án là **`qaz812345/TrackNetV3`** — khác repo, script không tương thích thẳng. Bài học FP16 mục 5 cho thấy claim "precision thấp hơn = nhanh hơn" **không đáng tin trên GPU yếu này** mà chưa đo thật. Rủi ro cao, effort lớn (cài TensorRT SDK, bộ hiệu chỉnh INT8). |
| 2 | Bỏ module InpaintNet (Rectification) | **Không áp dụng được** — đọc `tracknet_model.py:6-11` xác nhận dự án **chưa từng vendor InpaintNet** (đã thay bằng Kalman filter + `fill_missing_trajectory` riêng). Tiền đề đề xuất sai. |
| 3 | Export ONNX Runtime | Khả thi nhất, rủi ro thấp nhất — xem mục 7. |
| 4 | Giảm `seq_len`, retrain | Repo **chưa có pipeline train TrackNet** (chỉ có inference code) — rào cản lớn. Tính lại: giảm seq_len chỉ giảm channel đầu vào lớp ĐẦU TIÊN, phần nặng nhất model (block 128/256/512 channel) không đổi — lợi ích nhỏ hơn nhiều so với mô tả "giảm gần một nửa FLOPs". |

**Xếp hạng ưu tiên đã chọn:** ONNX Runtime (3) > TensorRT (1, cần verify) > bỏ InpaintNet (2, moot) > giảm seq_len (4, đầu tư lớn nhất, lợi ích nghi ngờ).

---

## 7. Thử nghiệm tối ưu #3 — ONNX Runtime backend

**Ý tưởng**: `TrackNet` (`core/tracknet_model.py`) chỉ dùng ops chuẩn (Conv2d/BatchNorm2d/MaxPool2d/Upsample/Sigmoid) — export ONNX sạch, không đổi trọng số. PyTorch eager mode dispatch từng block qua Python; ONNX Runtime gộp cả graph thành 1 lệnh C++ — lợi thế tiềm năng ở batch=1 (overhead dispatch Python chiếm tỷ trọng đáng kể).

**Triển khai**:
- `backend/scripts/export_tracknet_onnx.py` — export + tự so sánh output PyTorch vs ONNX (đạt: lệch tối đa 1e-6).
- `backend/core/tracknet_onnx.py` — `TrackNetONNXCandidateSource` (kế thừa `TrackNetCandidateSource`, chỉ override phần đụng PyTorch).
- `ShuttleDetector(backend="tracknet-onnx")` (`shuttle_tracker.py`), `--shuttle-backend tracknet-onnx` (`benchmark_pipeline.py`).
- Trở ngại triển khai: `onnxruntime-gpu` bản mới nhất (1.30) yêu cầu CUDA 13.x/cuDNN 9 hệ thống — **âm thầm fallback về CPU** nếu thiếu (không báo lỗi rõ, chỉ chạy chậm). Đã phát hiện bằng cách tự check `session.get_providers()` sau khi khởi tạo, không tin "chạy không lỗi" là "chạy đúng GPU". Fix: pin `onnxruntime-gpu==1.19.2` + cài kèm `nvidia-cudnn-cu12`, `nvidia-cublas-cu12` (pip wheel cấp DLL CUDA/cuDNN, không cần cài CUDA toolkit hệ thống).

### Kết quả đo thật

**Đúng 100%**: `pct_frames_detected` khớp tuyệt đối PyTorch FP32 (3.6%/64 runs cả 2 bên) — cùng trọng số, cùng phép toán.

**Riêng forward pass (isolate, không video/YOLO/pose), 30 lần lặp**: ONNX nhanh hơn thật — **94.4ms/batch vs 107.0ms/batch PyTorch (~12%)**. Xác nhận lý thuyết dispatch-overhead đúng hướng.

**Nhưng full pipeline lại chậm hơn**, đo liền kề cùng điều kiện hệ thống (1800 frame):

| | PyTorch (`tracknet`) | ONNX Runtime (`tracknet-onnx`) |
|---|---|---|
| wall_fps | **17.57** | 11.52 |
| yolo ms/frame | 25.55 | 39.74 |
| shuttle ms/frame | 20.15 | 24.02 |
| pose ms/frame | 10.65 | 22.11 |
| pct_frames_detected / n_runs | 3.6% / 64 | 3.6% / 64 (giống hệt) |

**Mọi stage đều chậm đi** với backend ONNX, **kể cả YOLO và MediaPipe pose — code hoàn toàn không đổi** giữa 2 lần đo này. Vì forward pass riêng lẻ của ONNX nhanh hơn, nguyên nhân không thể là bản thân model ONNX. Giả thuyết (chưa xác nhận bằng profiler): **PyTorch và ONNX Runtime mỗi cái giữ 1 CUDA context/allocator riêng** trong cùng 1 process — `ShuttleDetector(backend="tracknet-onnx")` vẫn chạy YOLO qua PyTorch, TrackNet qua ONNX Runtime, nên mỗi frame phải chuyển đổi qua lại giữa 2 context độc lập trên GPU 4GB vốn đã chật — chi phí chuyển đổi này lớn hơn cả lợi ích 12% kia, và kéo chậm luôn cả các stage không liên quan.

**Kết luận**: giữ `backend="tracknet"` (PyTorch) làm mặc định ở mọi nơi (`ShuttleDetector`, `benchmark_pipeline.py`, `test_auto_umpire.py`). Code ONNX (`tracknet-onnx`) giữ lại — đúng, chạy được, có tài liệu đầy đủ — nhưng **không kích hoạt mặc định**. Muốn tận dụng lợi thế ONNX thật sự cần chuyển cả YOLO sang ONNX Runtime để loại bỏ việc mix 2 CUDA context — việc lớn hơn nhiều, ngoài phạm vi phiên này.

---

## 8. Tổng kết trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| torch | `2.5.1+cu121`, dùng đúng GPU |
| TrackNet backend mặc định | PyTorch, **FP32** (`use_half=False`) |
| Threading overlap (pose/shuttle) | Đã revert, không có trong code |
| Multiprocessing overlap | Đã revert, không có trong code |
| ONNX Runtime backend | Có sẵn (`backend="tracknet-onnx"`), **không mặc định** |
| `batch_stride`, `imgsz` (YOLO) | Giữ nguyên mặc định — chưa thử đổi (user chưa đồng ý đánh đổi accuracy) |
| Train pipeline cho TrackNet | Chưa tồn tại trong repo |

**Bài học chung xuyên suốt phiên**: mọi giả thuyết tối ưu "lý thuyết nên nhanh hơn" (FP16, threading, multiprocessing, ONNX) đều **phải đo thật trên đúng phần cứng này** trước khi tin — 3/4 hướng thất bại dù lý thuyết đều hợp lý, chỉ có đúng 1 bug fix thật (FP16→FP32) mang lại cải thiện rõ ràng, và bug đó chỉ lộ ra nhờ so sánh số liệu detect-rate trước/sau, không phải nhờ đọc code.

## 9. File liên quan

- Code: `backend/core/tracknet.py`, `backend/core/tracknet_onnx.py`, `backend/core/tracknet_model.py`, `backend/core/shuttle_tracker.py`, `backend/core/player_tracker.py`, `backend/core/pose_update.py`*, `backend/core/pose_process.py`*
  (*2 file này thuộc thử nghiệm multiprocessing đã revert — không còn tồn tại trong code hiện tại, chỉ nhắc lại cho lịch sử)
- Script: `backend/scripts/benchmark_pipeline.py`, `backend/scripts/bench_tracknet_forward.py`, `backend/scripts/export_tracknet_onnx.py`
- Tài liệu kỹ thuật đầy đủ (mục E, F): `backend/README.md`
- Số liệu benchmark thô: `data/benchmarks/tracknet.json` (baseline gốc), `tracknet_fp32fix.json`, `tracknet_fp32_recheck.json`, `tracknet_onnx.json`
