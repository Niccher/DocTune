"""
web/routers/models.py
---------------------
Model checkpoints manager and adapter merging utility.
"""

import json
import time
from pathlib import Path
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from pipeline.common import PROJECT_ROOT, Config
from web.state import state, JobType
from web.services.process_manager import process_mgr
from web.templates_config import templates
from web.celery_app import is_celery_available, app as celery_app
from web.db.session import get_db, SessionLocal, resolve_session
from web.db.models import Checkpoint

from web.services.hf_model_service import hf_model_service

router = APIRouter()



def get_checkpoint_details(dir_path: Path):
    if not dir_path.exists():
        return None

    files = list(dir_path.glob("*"))
    if not files:
        return None

    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    adapter_cfg_file = dir_path / "adapter_config.json"
    adapter_cfg = None
    if adapter_cfg_file.exists():
        try:
            with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
        except Exception:
            pass

    return {
        "path": str(dir_path),
        "name": dir_path.name,
        "size_mb": round(total_bytes / (1024 * 1024), 2),
        "files_count": len(files),
        "is_adapter": adapter_cfg_file.exists(),
        "adapter_config": adapter_cfg,
    }


@router.get("/models", response_class=HTMLResponse)
async def models_view(request: Request):
    cfg = Config.load()
    checkpoints = {
        "qlora": get_checkpoint_details(cfg.output_qlora),
        "lora": get_checkpoint_details(cfg.output_lora),
        "fft": get_checkpoint_details(cfg.output_fft),
        "merged": get_checkpoint_details(cfg.output_merged),
    }

    db_checkpoints = []
    try:
        from web.db.session import SessionLocal
        from web.db.models import Checkpoint
        if SessionLocal:
            db_s = SessionLocal()
            rows = db_s.query(Checkpoint).order_by(Checkpoint.created_at.desc()).all()
            for r in rows:
                db_checkpoints.append({
                    "id": r.id,
                    "name": r.name,
                    "run_id": r.run_id,
                    "adapter_type": r.adapter_type,
                    "step": r.step,
                    "val_loss": r.val_loss,
                    "disk_path": r.disk_path,
                    "s3_bucket": r.s3_bucket,
                    "s3_key": r.s3_key,
                    "size_mb": r.size_mb,
                    "base_model": r.base_model,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                })
            db_s.close()
    except Exception:
        pass

    hub_models = hf_model_service.list_curated_models()
    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={
            "page_title": "Checkpoints & Model Hub",
            "active_page": "models",
            "checkpoints": checkpoints,
            "db_checkpoints": db_checkpoints,
            "hub_models": hub_models,
            "active_default_model": cfg.model_id,
            "cfg": cfg.to_dict(),
            "active_job": state.active_job.to_dict() if state.active_job else None,
        },
    )


@router.get("/api/v1/models/checkpoints")
async def list_checkpoints(db: Session = Depends(get_db)):
    checkpoints = []
    active_db, should_close = resolve_session(db)
    try:
        if active_db:
            rows = active_db.query(Checkpoint).order_by(Checkpoint.created_at.desc()).all()
            for r in rows:
                checkpoints.append({
                    "id": r.id,
                    "name": r.name,
                    "run_id": r.run_id,
                    "adapter_type": r.adapter_type,
                    "step": r.step,
                    "val_loss": r.val_loss,
                    "disk_path": r.disk_path,
                    "s3_bucket": r.s3_bucket,
                    "s3_key": r.s3_key,
                    "size_mb": r.size_mb,
                    "base_model": r.base_model,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if should_close and active_db:
            active_db.close()
    return JSONResponse({"status": "success", "checkpoints": checkpoints})


@router.get("/api/v1/models/checkpoints/{checkpoint_id}/download-url")
async def get_checkpoint_download_url(checkpoint_id: str, db: Session = Depends(get_db)):
    """Generate presigned S3 download URLs for checkpoint files."""
    from web.services.storage_service import storage_service

    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=503)

    try:
        ckpt = active_db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
        if not ckpt:
            return JSONResponse({"status": "error", "message": "Checkpoint not found"}, status_code=404)

        if not ckpt.s3_bucket or not ckpt.s3_key:
            return JSONResponse({
                "status": "local_only",
                "disk_path": ckpt.disk_path,
                "message": "Checkpoint is stored only on local disk.",
            })

        objs = storage_service.list_objects(ckpt.s3_bucket, prefix=ckpt.s3_key)
        files = []
        for o in objs:
            url = storage_service.generate_presigned_download_url(
                bucket=ckpt.s3_bucket,
                s3_key=o["key"],
                expires_in=3600,
            )
            files.append({
                "key": o["key"],
                "filename": Path(o["key"]).name,
                "size": o["size"],
                "download_url": url,
            })

        return JSONResponse({
            "status": "success",
            "checkpoint_id": checkpoint_id,
            "bucket": ckpt.s3_bucket,
            "prefix": ckpt.s3_key,
            "files": files,
        })
    finally:
        if should_close and active_db:
            active_db.close()


@router.post("/api/v1/models/checkpoints/{checkpoint_id}/pull")
async def pull_checkpoint_local(checkpoint_id: str, db: Session = Depends(get_db)):
    """Download checkpoint files from MinIO/S3 down to the local worker/web node."""
    from web.services.storage_service import storage_service

    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=503)

    try:
        ckpt = active_db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
        if not ckpt:
            return JSONResponse({"status": "error", "message": "Checkpoint not found"}, status_code=404)

        if not ckpt.s3_bucket or not ckpt.s3_key:
            return JSONResponse({"status": "error", "message": "Checkpoint has no S3 storage location"}, status_code=400)

        target_dir = Path(ckpt.disk_path) if ckpt.disk_path else PROJECT_ROOT / f"checkpoints/{ckpt.id}"
        success = storage_service.download_dir(ckpt.s3_bucket, ckpt.s3_key, target_dir)
        if success:
            return JSONResponse({
                "status": "success",
                "message": f"Checkpoint pulled successfully to {target_dir}",
                "local_path": str(target_dir),
            })
        return JSONResponse({"status": "error", "message": "Failed to pull files from object storage"}, status_code=500)
    finally:
        if should_close and active_db:
            active_db.close()


@router.post("/api/v1/models/merge")
async def trigger_merge(
    adapter_dir: str = Form(...),
    export_dir: str = Form(...),
    model_id: str = Form("meta-llama/Meta-Llama-3-8B-Instruct"),
):
    if state.active_job is not None:
        return JSONResponse({"status": "error", "message": "Another job is already running"}, status_code=409)

    src_path = Path(adapter_dir)
    if not (src_path / "adapter_config.json").exists():
        return JSONResponse({"status": "error", "message": f"No adapter_config.json in {adapter_dir}"}, status_code=400)

    run_id = f"merge_{int(time.time())}"
    is_celery = is_celery_available() and celery_app

    # Persist Run entry in PostgreSQL
    try:
        from web.db.session import SessionLocal
        from web.db.models import Run
        if SessionLocal:
            db_m = SessionLocal()
            run_rec = Run(
                id=run_id,
                model_id=model_id,
                method="MERGE",
                status="RUNNING",
                mode="celery" if is_celery else "local",
            )
            db_m.add(run_rec)
            db_m.commit()
            db_m.close()
    except Exception:
        pass

    # Dual execution: Use Celery if worker cluster & Redis are available, else fallback to local subprocess
    if is_celery:
        job = state.set_job(
            job_type=JobType.MERGING,
            metadata={"adapter_dir": adapter_dir, "export_dir": export_dir, "run_id": run_id, "mode": "celery", "model_id": model_id},
        )
        await state.add_log(f"--- [DISPATCH] Model adapter merge {run_id} dispatched to Celery cluster ---")
        celery_task = celery_app.send_task(
            "web.celery_app.task_merge_adapters",
            args=[run_id, adapter_dir, export_dir, model_id],
            queue="training_queue",
        )
        job.metadata["celery_task_id"] = celery_task.id
        return JSONResponse({"status": "started", "job_id": run_id})

    args = [
        "--adapter-dir", adapter_dir,
        "--export-dir", export_dir,
        "--model-id", model_id,
    ]

    await process_mgr.run_pipeline_script(
        job_type=JobType.MERGING,
        script_module="pipeline.merge_adapters",
        args=args,
        cwd=PROJECT_ROOT,
        metadata={"adapter_dir": adapter_dir, "export_dir": export_dir, "run_id": run_id, "model_id": model_id},
    )

    job_id = state.active_job.job_id if state.active_job else run_id
    return JSONResponse({"status": "started", "job_id": job_id})


# ── Hugging Face Base Models Hub Endpoints ─────────────────────────

@router.get("/api/v1/models/hub")
async def get_hub_models():
    """List curated Hugging Face models decorated with local cache and default status."""
    catalog = hf_model_service.list_curated_models()
    cfg = Config.load()
    return JSONResponse({
        "status": "success",
        "models": catalog,
        "active_default": cfg.model_id,
    })


@router.get("/api/v1/models/hub/search")
async def search_hub_models(q: str = ""):
    """Search text-generation models live on Hugging Face Hub."""
    models = await hf_model_service.search_hf_hub(q)
    cfg = Config.load()
    return JSONResponse({
        "status": "success",
        "models": models,
        "active_default": cfg.model_id,
    })


@router.post("/api/v1/models/hub/set-default")
async def set_default_base_model(model_id: str = Form(...)):
    """Set the system-wide active base model in settings.json and database."""
    success = hf_model_service.set_active_base_model(model_id)
    if success:
        return JSONResponse({"status": "success", "default_model_id": model_id})
    return JSONResponse({"status": "error", "message": "Failed updating default model"}, status_code=500)


@router.post("/api/v1/models/hub/download")
async def trigger_model_download(model_id: str = Form(...)):
    """Trigger background download of base model weights and tokenizer into cache."""
    clean_id = model_id.strip()
    if is_celery_available() and celery_app:
        celery_task = celery_app.send_task(
            "web.celery_app.task_download_model",
            args=[clean_id],
            queue="data_prep_queue",
        )
        hf_model_service.register_download_task(clean_id, celery_task_id=celery_task.id)
        return JSONResponse({"status": "started", "task_id": celery_task.id, "model_id": clean_id})
    else:
        import asyncio
        asyncio.create_task(hf_model_service.start_background_download(clean_id))
        return JSONResponse({"status": "started", "model_id": clean_id})


@router.get("/api/v1/models/hub/download-status")
async def get_model_download_status(model_id: str):
    """Poll background download progress and local cache readiness."""
    status_data = hf_model_service.get_download_status(model_id)
    return JSONResponse({
        "status": "success",
        "model_id": model_id,
        "download_status": status_data.get("status", "IDLE"),
        "progress": status_data.get("progress", 0),
        "cached": status_data.get("cached", False),
        "size_gb": status_data.get("size_gb", 0.0),
        "path": status_data.get("path"),
    })

