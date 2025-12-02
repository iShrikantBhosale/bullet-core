from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import shutil
import os
import threading
from train_engine import train_manager

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TrainConfig(BaseModel):
    vocab_size: int = 1000
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 4

# ===== API ROUTES =====
# Using explicit /api prefix to avoid conflict with StaticFiles

@app.post("/api/train")
async def start_training(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vocab_size: int = Form(4000),
    dim: int = Form(256),
    num_heads: int = Form(4),
    num_layers: int = Form(8),
    max_seq_len: int = Form(128),
    learning_rate: float = Form(5e-4),
    epochs: int = Form(3),
    batch_size: int = Form(4),
    gradient_accumulation_steps: int = Form(16),
    weight_decay: float = Form(0.01),
    clip_grad_norm: float = Form(1.0)
):
    if train_manager.is_training:
        return JSONResponse(status_code=400, content={"message": "Training already in progress"})

    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    config = {
        "vocab_size": vocab_size,
        "dim": dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "max_seq_len": max_seq_len,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "weight_decay": weight_decay,
        "clip_grad_norm": clip_grad_norm
    }

    # Start training in background
    background_tasks.add_task(train_manager.train_and_export, config, file_path, OUTPUT_DIR)

    return {"message": "Training started", "config": config}

@app.post("/api/stop")
async def stop_training():
    if not train_manager.is_training:
        return JSONResponse(status_code=400, content={"message": "No training in progress"})
    
    train_manager.stop_training()
    return {"message": "Training stop requested"}

@app.get("/api/logs")
async def get_logs():
    logs = train_manager.get_logs()
    return {
        "logs": logs,
        "progress": train_manager.progress,
        "is_training": train_manager.is_training,
        "model_available": bool(train_manager.model_path) and train_manager.progress >= 1.0,
        "model_name": train_manager.model_path,
        "metrics": train_manager.metrics  # Production Dashboard: Include metrics for graphs
    }

@app.get("/api/download/{model_name}")
async def download_model(model_name: str):
    file_path = os.path.join(OUTPUT_DIR, model_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=model_name)
    raise HTTPException(status_code=404, detail="Model not found")

# ===== STATIC FILES (MUST BE LAST) =====
import pathlib
frontend_path = pathlib.Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
