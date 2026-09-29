import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CầuLôngStats API", version="1.0")

# Setup CORS. Origins come from CORS_ORIGINS (comma separated); the default is the Vite dev
# server. Never combine allow_origins=["*"] with allow_credentials=True.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
