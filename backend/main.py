import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CầuLôngStats API", version="1.0")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to CầuLôngStats API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- ĐOẠN CODE MỚI THÊM VÀO DÀNH CHO FRONTEND ---
@app.get("/api/tracking/{video_name}")
def get_tracking_data(video_name: str):
    """
    Đọc file JSON chứa dữ liệu tracking (skeleton, tọa độ cầu, v.v.)
    """
    # Đường dẫn trỏ tới thư mục chứa kết quả của script benchmark_pipeline.py
    # Ví dụ url là /api/tracking/run -> nó sẽ tìm file ../data/benchmarks/run.json
    file_path = f"../data/benchmarks/{video_name}.json" 
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file: {file_path}")
# ------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)