"""
web/routers/training.py
-----------------------
Training studio route: hyperparameter configuration, start/abort triggers,
and WebSocket endpoint for live log streaming and loss graphs.
"""

import json
import time
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Request, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline.common import PROJECT_ROOT, Config
from web.state import state, JobType, JobStatus
from web.services.process_manager import process_mgr
from web.services.queue_manager import queue_manager
from web.services.sweep_service import sweep_service
from web.services.dataset_version_service import dataset_version_service
from web.services.audit_service import audit_service
from web.templates_config import templates
from web.celery_app import is_celery_available, HAS_CELERY, app as celery_app

router = APIRouter()

RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/training", response_class=HTMLResponse)
async def training_view(request: Request, model_id: Optional[str] = Query(None)):
    cfg = Config.load()
    cfg_dict = cfg.to_dict()
    if model_id and model_id.strip():
        cfg_dict["model_id"] = model_id.strip()

    return templates.TemplateResponse(
        request=request,
        name="training.html",
        context={
            "page_title": "Training Studio",
            "active_page": "training",
            "config": cfg_dict,
            "active_job": state.active_job.to_dict() if state.active_job else None,
            "loss_history": state.loss_history,
            "queued_jobs": queue_manager.get_queue(),
            "resumable_checkpoints": queue_manager.discover_resumable_checkpoints(),
            "dataset_versions": dataset_version_service.list_versions(),
            "active_dataset_version": dataset_version_service.get_active_version(),
        },
    )


@router.post("/api/v1/training/start")
async def start_training(
    request: Request,
    method: str = Form("qlora"),
    model_id: str = Form("meta-llama/Meta-Llama-3-8B-Instruct"),
    epochs: int = Form(3),
    batch_size: int = Form(2),
    grad_accum: int = Form(8),
    lora_r: int = Form(16),
    lora_alpha: int = Form(32),
    lora_dropout: float = Form(0.05),
    learning_rate: float = Form(0.0002),
    max_seq_length: int = Form(1024),
    resume_checkpoint: Optional[str] = Form(None),
    dataset_version: Optional[str] = Form(None),
):
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

    # Lineage: Resolve custom dataset version if specified
    ds_entry = None
    if dataset_version and dataset_version.strip():
        ds_entry = dataset_version_service.get_version(dataset_version.strip())
        if ds_entry and Path(ds_entry["path"]).exists():
            cfg.dataset_path = Path(ds_entry["path"])

    # Save run config
    run_id = f"run_{method.lower()}_{int(time.time())}"
    run_cfg_path = RUNS_DIR / f"config_{run_id}.json"
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    method_map = {
        "qlora": (JobType.TRAINING_QLORA, "pipeline.train_qlora"),
        "lora": (JobType.TRAINING_LORA, "pipeline.train_lora"),
        "fft": (JobType.TRAINING_FFT, "pipeline.train_fft"),
    }
    job_type, script_module = method_map.get(method.lower(), (JobType.TRAINING_QLORA, "pipeline.train_qlora"))

    mode = "celery" if (is_celery_available() and celery_app) else "subprocess"

    # If another job is currently active, enqueue in FIFO queue
    if state.active_job is not None:
        job_spec = {
            "run_id": run_id,
            "job_type": job_type.value,
            "method": method.lower(),
            "model_id": model_id,
            "config": cfg.to_dict(),
            "config_file": str(run_cfg_path),
            "resume_checkpoint": resume_checkpoint,
            "dataset_version": ds_entry.get("version_tag") if ds_entry else None,
            "dataset_hash": ds_entry.get("sha256") if ds_entry else None,
            "mode": mode,
        }
        res = queue_manager.enqueue(job_spec)
        audit_service.log(
            action="RUN_START",
            resource_type="run",
            resource_id=run_id,
            request=request,
            status="QUEUED",
            details={"method": method, "model_id": model_id, "position": res["position"], "epochs": epochs, "lr": learning_rate},
        )
        return JSONResponse({
            "status": "queued",
            "message": f"Job queued at position #{res['position']}",
            "job_id": run_id,
            "position": res["position"],
        })

    # Record Run in DB
    try:
        from web.db.session import SessionLocal
        from web.db.models import Run
        if SessionLocal:
            db = SessionLocal()
            new_run = Run(
                id=run_id,
                job_type=job_type.value,
                model_id=model_id,
                method=method.upper(),
                dataset_path=str(cfg.dataset_path),
                dataset_version=ds_entry.get("version_tag") if ds_entry else None,
                dataset_hash=ds_entry.get("sha256") if ds_entry else None,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                status="RUNNING",
                config_json=cfg.to_dict(),
                mode=mode,
            )
            db.add(new_run)
            db.commit()
            db.close()
    except Exception:
        pass

    audit_service.log(
        action="RUN_START",
        resource_type="run",
        resource_id=run_id,
        request=request,
        status="RUNNING",
        details={"method": method, "model_id": model_id, "mode": mode, "epochs": epochs, "lr": learning_rate},
    )

    # Launch immediately via Celery or Subprocess
    if mode == "celery" and celery_app:
        job = state.set_job(
            job_type=job_type,
            metadata={
                "method": method,
                "model_id": model_id,
                "run_id": run_id,
                "mode": "celery",
                "config_file": str(run_cfg_path),
                "resume_checkpoint": resume_checkpoint,
            },
        )
        await state.add_log(f"--- [DISPATCH] Job {run_id} ({job_type.value}) dispatched to Celery ---")
        celery_task = celery_app.send_task(
            "web.celery_app.task_train_model",
            args=[run_id, method, model_id, cfg.to_dict()],
            queue="training_queue",
        )
        job.metadata["celery_task_id"] = celery_task.id
        return JSONResponse({"status": "started", "job_id": run_id})

    # Subprocess execution
    args = ["--config-json", str(run_cfg_path)]
    if resume_checkpoint and resume_checkpoint.strip():
        args.extend(["--resume-from-checkpoint", str(resume_checkpoint.strip())])

    await process_mgr.run_pipeline_script(
        job_type=job_type,
        script_module=script_module,
        args=args,
        cwd=PROJECT_ROOT,
        metadata={
            "method": method,
            "model_id": model_id,
            "run_id": run_id,
            "mode": "subprocess",
            "config_file": str(run_cfg_path),
            "resume_checkpoint": resume_checkpoint,
        },
    )

    job_id = state.active_job.job_id if state.active_job else run_id
    return JSONResponse({"status": "started", "job_id": job_id})


@router.get("/api/v1/training/queue", response_class=JSONResponse)
async def get_training_queue_api():
    """Return all currently queued training jobs."""
    return {"status": "success", "queue": queue_manager.get_queue()}


@router.delete("/api/v1/training/queue/{run_id}", response_class=JSONResponse)
async def remove_queued_job_api(request: Request, run_id: str):
    """Remove a job from the training queue."""
    removed = queue_manager.remove(run_id)
    audit_service.log(
        action="QUEUE_CANCEL",
        resource_type="queue_job",
        resource_id=run_id,
        request=request,
        status="SUCCESS" if removed else "FAILED",
    )
    if removed:
        return {"status": "success", "message": f"Job {run_id} removed from queue"}
    return JSONResponse({"status": "error", "message": f"Job {run_id} not found in queue"}, status_code=404)


@router.get("/api/v1/training/checkpoints/resumable", response_class=JSONResponse)
async def list_resumable_checkpoints_api():
    """Return list of existing checkpoint directories that can be resumed."""
    checkpoints = queue_manager.discover_resumable_checkpoints()
    return {"status": "success", "checkpoints": checkpoints}


@router.post("/api/v1/training/sweeps/generate", response_class=JSONResponse)
async def generate_sweep_api(request: Request):
    """Generate and queue a combinatorial hyperparameter sweep matrix."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    model_id = body.get("model_id", "meta-llama/Meta-Llama-3-8B-Instruct")
    method = body.get("method", "qlora")

    # Parse comma-separated or list values
    def parse_float_list(val, default):
        if not val:
            return default
        if isinstance(val, list):
            return [float(x) for x in val]
        return [float(x.strip()) for x in str(val).split(",") if x.strip()]

    def parse_int_list(val, default):
        if not val:
            return default
        if isinstance(val, list):
            return [int(x) for x in val]
        return [int(x.strip()) for x in str(val).split(",") if x.strip()]

    lrs = parse_float_list(body.get("learning_rates"), [0.0001, 0.0002])
    ranks = parse_int_list(body.get("lora_ranks"), [16, 32])
    epochs = parse_int_list(body.get("epochs_list"), [3])

    res = sweep_service.generate_grid_sweep(
        model_id=model_id,
        method=method,
        learning_rates=lrs,
        lora_ranks=ranks,
        epochs_list=epochs,
    )
    audit_service.log(
        action="SWEEP_START",
        resource_type="sweep",
        resource_id=res.get("sweep_id"),
        request=request,
        status="SUCCESS",
        details={"model_id": model_id, "method": method, "total_combinations": res.get("total_combinations")},
    )
    return JSONResponse(res)


@router.post("/api/v1/training/stop")
async def stop_training(request: Request):
    curr_job = state.active_job.job_id if state.active_job else None
    if state.active_job and state.active_job.metadata.get("mode") == "celery":
        celery_id = state.active_job.metadata.get("celery_task_id")
        if celery_id and celery_app:
            try:
                celery_app.control.revoke(celery_id, terminate=True, signal="SIGTERM")
            except Exception:
                pass
        state.finish_job(JobStatus.ABORTED)
        await state.add_log("--- [ABORTED] Celery task aborted by user ---")
        audit_service.log(
            action="RUN_CANCEL",
            resource_type="run",
            resource_id=curr_job,
            request=request,
            status="SUCCESS",
            details="Celery task aborted",
        )
        return JSONResponse({"status": "success", "message": "Celery job revoked"})

    aborted = await process_mgr.abort_current_job()
    audit_service.log(
        action="RUN_CANCEL",
        resource_type="run",
        resource_id=curr_job,
        request=request,
        status="SUCCESS" if aborted else "FAILED",
    )
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
