"""
web/routers/training.py
-----------------------
Training studio route: hyperparameter configuration, start/abort triggers,
and WebSocket endpoint for live log streaming and loss graphs.
"""

import json
import time
from pathlib import Path
from fastapi import APIRouter, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline.common import PROJECT_ROOT, Config
from web.state import state, JobType
from web.services.process_manager import process_mgr
from web.templates_config import templates

router = APIRouter()

RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/training", response_class=HTMLResponse)
async def training_view(request: Request):
    cfg = Config.load()
    return templates.TemplateResponse(
        request=request,
        name="training.html",
        context={
            "page_title": "Training Studio",
            "active_page": "training",
            "config": cfg.to_dict(),
            "active_job": state.active_job.to_dict() if state.active_job else None,
            "loss_history": state.loss_history,
        },
    )


@router.post("/api/training/start")
async def start_training(
    method: str = Form("qlora"),
    model_id: str = Form("meta-llama/Meta-Llama-3-8B-Instruct"),
    epochs: int = Form(3),
    batch_size: int = Form(4),
    grad_accum: int = Form(4),
    lora_r: int = Form(16),
    lora_alpha: int = Form(32),
    lora_dropout: float = Form(0.05),
    learning_rate: float = Form(0.0002),
    max_seq_length: int = Form(1024),
):
    if state.active_job is not None:
        return JSONResponse({"status": "error", "message": "Another job is already running"}, status_code=409)

    # Enforce safe parameter boundaries
    epochs = max(1, min(int(epochs), 50))
    batch_size = max(1, min(int(batch_size), 32))
    grad_accum = max(1, min(int(grad_accum), 64))
    lora_r = max(2, min(int(lora_r), 256))
    lora_alpha = max(2, min(int(lora_alpha), 512))
    lora_dropout = max(0.0, min(float(lora_dropout), 0.50))
    learning_rate = max(1e-7, min(float(learning_rate), 0.1))
    max_seq_length = max(128, min(int(max_seq_length), 8192))

    cfg = Config(
        model_id=model_id,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate_lora=learning_rate,
        max_seq_length=max_seq_length,
    )

    # Save run config
    run_cfg_path = RUNS_DIR / f"config_{int(time.time())}.json"
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    method_map = {
        "qlora": (JobType.TRAINING_QLORA, "pipeline.train_qlora"),
        "lora": (JobType.TRAINING_LORA, "pipeline.train_lora"),
        "fft": (JobType.TRAINING_FFT, "pipeline.train_fft"),
    }

    job_type, script_module = method_map.get(method.lower(), (JobType.TRAINING_QLORA, "pipeline.train_qlora"))

    args = ["--config-json", str(run_cfg_path)]

    await process_mgr.run_pipeline_script(
        job_type=job_type,
        script_module=script_module,
        args=args,
        cwd=PROJECT_ROOT,
        metadata={"method": method, "model_id": model_id, "config_file": str(run_cfg_path)},
    )

    return JSONResponse({"status": "started", "job_id": state.active_job.job_id})


@router.post("/api/training/stop")
async def stop_training():
    aborted = await process_mgr.abort_current_job()
    if aborted:
        return JSONResponse({"status": "success", "message": "Job abort signal sent"})
    return JSONResponse({"status": "error", "message": "No active process to stop"}, status_code=400)


@router.websocket("/ws/stream")
@router.websocket("/ws/training")
async def websocket_stream_endpoint(websocket: WebSocket):
    await state.connect_ws(websocket)
    try:
        while True:
            # Keep connection open and receive potential client heartbeats/messages
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        state.disconnect_ws(websocket)
    except Exception:
        state.disconnect_ws(websocket)
