# CầuLôngStats – Quy tắc phát triển

Hệ thống phân tích cầu lông: theo dõi người chơi (YOLOv8 + ByteTrack + MediaPipe), theo dõi cầu (TrackNetV3 + Kalman), trọng tài tự động IN/OUT, phát hiện đập cầu. Backend Python (`backend/`), frontend React (`frontend/`, mới là scaffold).

Đọc file này trước khi sửa code. Rule nào code hiện tại chưa tuân thủ được ghi ở mục **Known issues**.

## 1. Bố cục & lệnh

| Thư mục | Vai trò |
|---|---|
| `backend/core/` | Logic thuần, không UI (`cv2.imshow`, `waitKey` bị cấm). Đây là nơi duy nhất chứa thuật toán. |
| `backend/scripts/` | Demo/tool tương tác (`demo_*.py`, `calibrate_court.py`), benchmark, tiện ích. **Không phải test.** |
| `backend/tests/` | pytest thật, dữ liệu tổng hợp, không cần video. |
| `backend/models/` | Weights (git-ignore). Cách tải: `QUICK_START.md`. |
| `data/` | `raw/`, `cfr/` (git-ignore), `calibration.json`, `metadata.csv`, `benchmarks/*.json`. |

```bash
cd backend
python -m pytest -q                         # chạy TẤT CẢ test (pytest.ini chỉ thu thập tests/)
python scripts/demo_auto_umpire.py [video]  # demo; chạy từ thư mục nào cũng được
python scripts/calibrate_court.py --video ../data/cfr/tran04_cam1.mp4
```

## 2. Hệ tọa độ & đơn vị (bất biến – sai là sai kết quả)

- **Ảnh**: pixel `(x, y)`, y hướng xuống, full-frame trừ khi ghi khác. ROI là `(x0, y0, x1, y1)`. Bbox luôn `(x1, y1, x2, y2)`.
- **Thế giới**: mét. `x` dọc sân (0 = vạch cuối gần, `NET_X` = 6.70 = lưới, 13.40 = vạch cuối xa), `y` ngang sân 0..6.10. Tọa độ vạch là **tâm** vạch; `judge()` cộng `LINE_HALF_WIDTH` để vạch tính là IN.
- `H`: world → image. `H_inv`: image → world. Không đổi chiều ngầm; đặt tên biến `*_px`, `*_m`, `img_*`, `world_*` để lộ hệ tọa độ.
- **Hậu tố đơn vị bắt buộc**: `_px`, `_m`, `_s`, `_frames`. Tốc độ cầu: px/frame. Tốc độ người: m/s.
- Video phải là **CFR 60 fps** (`process_all_videos.py`). Không đoán fps: lấy từ `ThreadedVideoReader.fps`; nếu bỏ frame (stride) thì truyền `fps / stride` cho `PlayerTracker`.
- TrackNet chỉ ra candidate mỗi `batch_stride` frame (mặc định `config.TRACKNET_BATCH_STRIDE` = 5; 8 = không chồng lấn), các frame còn lại là mảng rỗng. Code tiêu thụ phải chịu được `None`/rỗng; `max_coast` và `min_gate_px` scale theo stride.
- Calibration hiện chỉ khớp **nửa sân gần**; nửa xa được ngoại suy. `data/calibration.json` là một calibration toàn cục: đổi góc camera phải hiệu chỉnh lại.

## 3. Config & hằng số

- Mọi ngưỡng/tunable nằm trong `backend/core/config.py`: một định nghĩa duy nhất, có docstring ghi ý nghĩa + đơn vị, tên có prefix theo hệ con (`KALMAN_*`, `PHYSICS_*`, `UMPIRE_*`, `SMASH_*`, `SELECTOR_*`, `TRACKNET_*`, `CV_*`, `PLAYER_*`).
- `config.py` là lá: **không import module `core` khác**. Kích thước sân vật lý ở `court_model.py`.
- Cấm magic number trong `core/` và `scripts/`. Ngoại lệ được chấp nhận: hằng toán học/kernel nội bộ có comment, và default của tham số keyword đã có tên + docstring.
- Default argument bind lúc `def`: giá trị cần tune live phải được đọc lúc gọi (`config.X` trong thân hàm, hoặc thuộc tính instance). Ví dụ đúng: `SmashDetector.speed_threshold` đọc `config` mỗi lần gọi.
- Đổi giá trị mặc định = đổi hành vi. Chạy lại benchmark (mục 8) và ghi lý do trong commit.

## 4. Đường dẫn

- Mọi path dẫn xuất từ vị trí file: dùng `config.DATA_DIR`, `MODELS_DIR`, `CALIBRATION_PATH`, `METADATA_PATH`, `TRACKNET_WEIGHTS_PATH`, `PLAYER_YOLO_MODEL_PATH`, `DEFAULT_VIDEO`.
- Cấm path tương đối theo CWD (`"models/yolov8n.pt"`) và chuỗi có backslash (`r"..\data\..."`).
- Output sinh ra (`highlights.html`, `missed_bounces.txt`, video demo) ghi vào `data/` và phải nằm trong `.gitignore`. Không commit binary/generated.

## 5. Import & cấu trúc code

- `core/`: import tương đối (`from . import config`). Import ở đầu file, không import trong hàm (trừ dependency nặng/tùy chọn, có comment lý do).
- `scripts/`: dòng đầu tiên là `from _common import ...` (hoặc `import _common`) — nó thêm `backend/` vào `sys.path`. Không tự viết `sys.path.append/insert`.
- Logic dùng chung (smash, fallback detector, vẽ) đặt trong `core/` hoặc `scripts/_common.py`, không copy giữa các script.
- Hàm nên ≤ ~50 dòng; constructor > 10 tham số → gom vào dataclass khi chạm tới.
- Không gán thuộc tính ad hoc lên object của module khác (`detector.paused = ...`): dùng biến cục bộ.
- Không side-effect lúc import (đọc file, print, khởi tạo model). Ngoại lệ duy nhất, đã ghi chú: `court_model.py` đọc kích thước sân từ `calibration.json` một lần.
- Dữ liệu có cấu trúc dùng dataclass/NamedTuple (`Call`, `FlightPoint`, `HitMeasurement`, `PlayerObs`), không dùng dict/tuple ad hoc mới.

## 6. Style

- PEP 8, 4 space, không trailing whitespace, EOL LF (`.gitattributes`). Type hint cho hàm public mới.
- Comment/docstring trong code: **tiếng Anh**, giải thích *vì sao* chứ không kể lại *làm gì*. Chuỗi hiển thị cho người dùng (UI, HTML) và tài liệu trong `docs/`: tiếng Việt.
- Không emoji trong code/log.
- `core/` dùng `logging.getLogger(__name__)`, format lazy (`logger.info("x=%s", x)`), không `print`. `scripts/` được `print`, và gọi `setup_logging()` ở `__main__` để thấy log của `core/`.
- Tên script demo: `demo_*.py`; tool: động từ (`calibrate_court.py`, `process_all_videos.py`). **Không bắt đầu bằng `test_`.**

## 7. Xử lý lỗi

- File/weights thiếu → `FileNotFoundError` với hướng dẫn khắc phục (nói rõ path `backend/models/...`). Tham số sai → `ValueError`. Video không mở được → `IOError` (mọi nơi, không nơi trả `None`).
- Cấm `except Exception: pass`. Bắt loại lỗi cụ thể, và log (`logger.warning`) khi nuốt lỗi.
- Fallback (vd. `tracknet` → `cv`) phải log warning nói rõ đã fallback; đặt ở một chỗ (`_common.create_detector`).
- Bất kỳ fallback nào làm đổi kết quả nghiệp vụ (vd. match type mặc định `singles`) phải log warning.
- `torch.load`: dùng `core.tracknet.load_checkpoint` (`weights_only=True`, fallback có cảnh báo). Chỉ nạp checkpoint tin cậy.

## 8. Test & kiểm chứng

- Logic mới ở `core/` phải có test trong `backend/tests/` (dữ liệu tổng hợp, hoặc `skip` có lý do rõ nếu cần weights).
- Test truyền tham số tường minh, không phụ thuộc giá trị tuning trong `config.py`.
- Sửa bug: viết test tái hiện trước.
- Đổi hành vi tracker/umpire/smash: chạy benchmark trước & sau và so sánh:
  ```bash
  python scripts/benchmark_pipeline.py <video> --frames 1800 --out ../data/benchmarks/<tên>.json
  python scripts/eval_events.py <video> --tol 12 --out <tên>      # P/R của hit, smash, chạm đất so với nhãn
  ```
  Đọc `pct_frames_detected`, `n_runs`, `umpire_calls`. GPU laptop throttle: so sánh các lần chạy sát nhau.
- `RallyUmpire.update` luôn nhận `flight_id=detector.flight_id`, `SmashDetector.update` nhận `detector.start_source`: một cú đánh khởi động lại track ngay trong một lần `detect()` nên `track_active` không hề đổi.
- `LiveTuner` (`scripts/_common.py`): thuộc tính phải tồn tại (gõ sai → `AttributeError`) và code bị tune phải đọc giá trị lúc gọi.

## 9. Git & review

- Conventional Commits: `feat|fix|refactor|test|docs|chore: mô tả`. Một thay đổi logic một commit; không commit khi `pytest` đỏ.
- Không commit: weights, video, `data/highlights.html`, `__pycache__`, `.env`.
- Trước khi xong việc, tự kiểm tra:
  1. `python -m pytest -q` xanh, và `python -m pyflakes core scripts tests` sạch.
  2. Đơn vị & hệ tọa độ của mọi giá trị mới đi qua ranh giới hàm.
  3. Không magic number / path CWD / `except Exception: pass` mới (`rg` để kiểm).
  4. `backend/README.md`, `QUICK_START.md` và docstring còn khớp code (đổi tên file/tham số thì grep toàn repo).

## 10. Bảo mật

- Không commit secret; cấu hình qua `.env` (`DATABASE_URL`, `CORS_ORIGINS`).
- CORS: origin lấy từ `CORS_ORIGINS`; không dùng `allow_origins=["*"]` cùng `allow_credentials=True`.
- Không thực thi/unpickle dữ liệu không tin cậy.

## Known issues (còn lại sau đợt dọn dẹp)

- `ShuttleDetector.__init__` có ~20 tham số; cần dataclass cấu hình khi chạm tới.
- Chưa có type hint ở phần lớn API `core/` (chỉ code mới có).
- Code vẽ trùng lặp: `_common.draw_shuttle_trajectory`/`Minimap` vs `ShuttleDetector.draw_trajectory` vs `CourtCalibrator.draw_court_frame`.
- **Chưa có nhãn thật**: `data/events/tran04_cam1.events.json` là file mẫu rỗng; `PHYSICS_STOP_SPEED_PX`, `SMASH_SPEED_THRESHOLD`, `UMPIRE_MIN_DESCENT_PX` và luật chạm đất đều chưa được hiệu chỉnh bằng dữ liệu. Điền nhãn rồi chạy `scripts/eval_events.py`.
- **Chạm đất vs hit**: cú nảy sau khi chạm sàn là đảo chiều vận tốc nên physics filter coi là ranh giới flight; umpire xử lý bằng cách xét flight cũ + vài điểm sau ranh giới (`_resolve_pending`). Cầu rơi trong đoạn không detect rồi xuất hiện nằm yên được call theo quy tắc `resting` (kém chính xác hơn `contact`). Các ngưỡng (`UMPIRE_REST_*`, `PHYSICS_STOP_SPEED_PX`) chưa hiệu chỉnh bằng nhãn; một số bounce vẫn bị bỏ sót khi cầu chỉ hiện ở vị trí nhiễu (di chuyển 2–4 px/frame do nhiễu TrackNet) hoặc không được detect.
- Vùng sân xa rất thô (1 px ≈ 13 cm tại vạch cuối xa; nửa xa chỉ là ngoại suy homography): call ở nửa xa kém tin cậy; xem `Call.uncertainty_m`.
- `MAX_COAST_CV = 8` là giá trị hiệu lực trước đây; bản WIP từng đặt 5 nhưng hằng số không được nối vào code. Nối hằng số làm đổi số liệu benchmark CV, nên giữ 8 — nếu muốn 5, đổi và benchmark lại.
- `court_model.py` đọc `calibration.json` lúc import (ngoại lệ có ghi chú); tách thành loader lazy nếu cần nhiều calibration.
- `backend/third_party/CourtKeyNet` là gitlink không có `.gitmodules` (clone mới sẽ thiếu nội dung).
- `docs/Tối ưu hiệu năng pipeline – Nhật ký kỹ thuật.md` là nhật ký lịch sử, vẫn dùng tên script cũ (`test_*.py`).
- `main.py`/`database.py` là scaffold, chưa nối với pipeline; `alembic`, `psycopg2-binary`, `pydantic-settings`, `python-multipart` chưa được import.
- Calibration chỉ cho nửa sân gần; nửa xa ngoại suy.
- `ShuttleDetector.fill_gaps()` (lấp frame không detect) mới dùng để vẽ; umpire/smash vẫn chỉ dùng detection thật. Homography chỉ mô tả mặt sàn nên không suy được độ cao cầu: ràng buộc sân chỉ loại điểm nằm dưới sàn hoặc ngoài ROI.
