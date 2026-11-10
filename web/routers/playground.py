import time
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from pipeline.common import PROJECT_ROOT, Config
from pipeline.evaluate import TEST_CASES
from web.state import state, JobType
from web.services.process_manager import process_mgr
from web.templates_config import templates
from web.celery_app import is_celery_available, app as celery_app
from web.db.session import SessionLocal, get_db, resolve_session
import random
from web.services.inference_service import inference_service
from web.services.deployment_service import deployment_service
from web.db.models import PlaygroundLog, EvaluationBenchmark, Checkpoint, Run
from web.routers.auth import require_project_role, get_project_id

router = APIRouter()




@router.get("/playground", response_class=HTMLResponse)
async def playground_view(request: Request):
    cfg = Config.load()
    available_adapters = []
    if (cfg.output_qlora / "adapter_config.json").exists():
        available_adapters.append({"name": "QLoRA Adapter", "path": str(cfg.output_qlora)})
    if (cfg.output_lora / "adapter_config.json").exists():
        available_adapters.append({"name": "LoRA Adapter", "path": str(cfg.output_lora)})
    if (cfg.output_merged / "config.json").exists():
        available_adapters.append({"name": "Merged Standalone Model", "path": str(cfg.output_merged)})

    # Also include any checkpoints registered in PostgreSQL that exist on disk
    if SessionLocal:
        try:
            db = SessionLocal()
            ckpts = db.query(Checkpoint).all()
            for c in ckpts:
                p = Path(c.disk_path) if c.disk_path else None
                if p and p.exists() and not any(a["path"] == str(p) for a in available_adapters):
                    available_adapters.append({"name": c.name, "path": str(p)})
            db.close()
        except Exception:
            pass

    return templates.TemplateResponse(
        request=request,
        name="playground.html",
        context={
            "page_title": "Evaluation & Playground",
            "active_page": "playground",
            "test_cases": TEST_CASES,
            "available_adapters": available_adapters,
            "cfg": cfg.to_dict(),
            "active_job": state.active_job.to_dict() if state.active_job else None,
        },
    )


@router.post("/api/v1/evaluate/run")
async def run_evaluation_job(
    adapter_path: str = Form(...),
    suite: Optional[str] = Form("regression_100"),
):
    if state.active_job is not None:
        return JSONResponse({"status": "error", "message": "Another job is already running"}, status_code=409)

    run_id = f"eval_{int(time.time())}"
    args = [
        "--suite", suite or "regression_100",
        "--target-model", adapter_path,
        "--save-db",
    ]

    await process_mgr.run_pipeline_script(
        job_type=JobType.EVAL,
        script_module="pipeline.eval.runner",
        args=args,
        cwd=PROJECT_ROOT,
        metadata={"adapter_path": adapter_path, "suite": suite, "run_id": run_id},
    )

    job_id = state.active_job.job_id if state.active_job else "job_eval"
    return JSONResponse({"status": "started", "job_id": job_id, "suite": suite})


@router.post("/api/v1/playground/chat")
async def playground_chat(
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    blind_ab: Optional[bool] = Form(False),
    temperature: Optional[float] = Form(0.7),
):
    """
    Interactive prompt comparison between base pre-trained model and
    fine-tuned adapter weights with optional blind A/B side-by-side randomization.
    """
    start_t = time.time()
    tuned_model_id = model or "default"

    # Query base model
    base_comp = await inference_service.generate_completion_async(
        messages=[{"role": "user", "content": prompt}],
        model_name="base",
        temperature=temperature or 0.7,
    )
    base_resp = base_comp.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Query fine-tuned model
    tuned_comp = await inference_service.generate_completion_async(
        messages=[{"role": "user", "content": prompt}],
        model_name=tuned_model_id,
        temperature=temperature or 0.7,
    )
    tuned_resp = tuned_comp.get("choices", [{}])[0].get("message", {}).get("content", "")

    latency_ms = round((time.time() - start_t) * 1000, 1)
    log_id = f"play_{int(time.time()*1000)}"

    # Blind A/B randomization to eliminate position bias
    swap = False
    if blind_ab:
        swap = random.choice([True, False])
        pane_a = tuned_resp if swap else base_resp
        pane_b = base_resp if swap else tuned_resp
    else:
        pane_a = base_resp
        pane_b = tuned_resp

    # Save prompt interaction to PostgreSQL if available
    if SessionLocal:
        try:
            db = SessionLocal()
            log_entry = PlaygroundLog(
                id=log_id,
                prompt=prompt,
                base_model_response=base_resp,
                tuned_model_response=tuned_resp,
                model_name=tuned_model_id,
                latency_ms=latency_ms,
                notes=json.dumps({"blind_ab": bool(blind_ab), "swapped": swap}),
            )
            db.add(log_entry)
            db.commit()
            db.close()
        except Exception:
            pass

    return JSONResponse({
        "status": "success",
        "log_id": log_id,
        "prompt": prompt,
        "base_response": base_resp,
        "tuned_response": tuned_resp,
        "blind_ab": bool(blind_ab),
        "pane_a": pane_a,
        "pane_b": pane_b,
        "latency_ms": latency_ms,
        "model_name": tuned_model_id,
    })


@router.get("/api/v1/deployments", response_class=JSONResponse)
def list_deployments():
    """List active and historical model checkpoint deployments."""
    return JSONResponse({"status": "success", "deployments": deployment_service.list_deployments()})


@router.post("/api/v1/deployments", response_class=JSONResponse)
def deploy_checkpoint_endpoint(
    checkpoint_id: str = Form(...),
    backend: Optional[str] = Form("llama.cpp"),
    port: Optional[int] = Form(8080),
    _role=Depends(require_project_role("operator")),
):
    """Deploy a model checkpoint for active serving."""
    try:
        dep = deployment_service.deploy_checkpoint(checkpoint_id, backend_type=backend or "llama.cpp", port=port or 8080)
        return JSONResponse({"status": "success", "deployment": dep})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@router.delete("/api/v1/deployments/{dep_id}", response_class=JSONResponse)
def undeploy_checkpoint_endpoint(dep_id: str, _role=Depends(require_project_role("operator"))):
    """Stop and undeploy an active serving instance."""
    success = deployment_service.undeploy_checkpoint(dep_id)
    return JSONResponse({"status": "success" if success else "not_found"})



@router.post("/api/v1/playground/feedback")
async def playground_feedback(request: Request, db: Session = Depends(get_db)):
    """Save human review rating and notes for a playground prompt comparison."""
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=503)

    log_id = ""
    rating = ""
    notes = ""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        log_id = str(body.get("log_id", ""))
        rating = str(body.get("rating", ""))
        notes = str(body.get("notes", ""))
    else:
        form = await request.form()
        log_id = str(form.get("log_id", ""))
        rating = str(form.get("rating", ""))
        notes = str(form.get("notes", ""))

    if not log_id or not rating:
        return JSONResponse({"status": "error", "message": "log_id and rating are required"}, status_code=400)

    try:
        entry = active_db.query(PlaygroundLog).filter(PlaygroundLog.id == log_id).first()
        if not entry:
            return JSONResponse({"status": "error", "message": "Log entry not found"}, status_code=404)
        entry.rating = rating.lower().strip()
        if notes:
            entry.notes = notes.strip()
        active_db.commit()
        return JSONResponse({"status": "success", "log_id": log_id, "rating": entry.rating})
    finally:
        if should_close and active_db:
            active_db.close()


@router.get("/api/v1/playground/history")
async def playground_history(limit: int = 20, db: Session = Depends(get_db)):
    """Retrieve recent prompt evaluation comparisons and human feedback."""
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "success", "total": 0, "history": []})

    try:
        logs = active_db.query(PlaygroundLog).order_by(PlaygroundLog.created_at.desc()).limit(limit).all()
        return JSONResponse({
            "status": "success",
            "total": len(logs),
            "history": [
                {
                    "id": l.id,
                    "prompt": l.prompt,
                    "base_response": l.base_model_response,
                    "tuned_response": l.tuned_model_response,
                    "rating": l.rating,
                    "notes": l.notes,
                    "latency_ms": l.latency_ms,
                    "created_at": l.created_at.isoformat() if l.created_at else None,
                }
                for l in logs
            ],
        })
    finally:
        if should_close and active_db:
            active_db.close()


@router.get("/api/v1/benchmarks")
async def get_benchmarks_api(db: Session = Depends(get_db)):
    """Return model evaluation benchmarks leaderboard."""
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "success", "benchmarks": []})

    try:
        results = active_db.query(EvaluationBenchmark).order_by(EvaluationBenchmark.evaluated_at.desc()).all()
        return JSONResponse({
            "status": "success",
            "benchmarks": [
                {
                    "id": b.id,
                    "checkpoint_id": b.checkpoint_id,
                    "run_id": b.run_id,
                    "test_suite": b.test_suite_name,
                    "bleu": b.bleu_score,
                    "rouge_1": b.rouge_1,
                    "rouge_2": b.rouge_2,
                    "rouge_l": b.rouge_l,
                    "perplexity": b.perplexity,
                    "avg_latency_ms": b.avg_latency_ms,
                    "evaluated_at": b.evaluated_at.isoformat() if b.evaluated_at else None,
                }
                for b in results
            ]
        })
    finally:
        if should_close and active_db:
            active_db.close()


# ── DPO / Preference Fine-Tuning Endpoints ──────────────────────────

@router.get("/api/v1/playground/dpo/stats")
async def get_dpo_stats(db: Session = Depends(get_db)):
    """Return preference dataset statistics from collected thumbs-up/down ratings."""
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "success", "total_rated": 0, "thumbs_up": 0, "thumbs_down": 0, "with_notes": 0})

    try:
        up_count = active_db.query(PlaygroundLog).filter(PlaygroundLog.rating == "thumbs_up").count()
        down_count = active_db.query(PlaygroundLog).filter(PlaygroundLog.rating == "thumbs_down").count()
        notes_count = active_db.query(PlaygroundLog).filter(PlaygroundLog.notes.isnot(None), PlaygroundLog.notes != "").count()
        total_rated = up_count + down_count

        pref_file = PROJECT_ROOT / "data" / "preference_pairs.jsonl"
        exported_count = 0
        if pref_file.exists():
            with open(pref_file, "r", encoding="utf-8") as f:
                exported_count = sum(1 for _ in f)

        return JSONResponse({
            "status": "success",
            "total_rated": total_rated,
            "thumbs_up": up_count,
            "thumbs_down": down_count,
            "with_notes": notes_count,
            "exported_count": exported_count,
            "ready_for_dpo": total_rated > 0,
        })
    finally:
        if should_close and active_db:
            active_db.close()


@router.post("/api/v1/playground/dpo/export")
async def export_dpo_preference_dataset(db: Session = Depends(get_db)):
    """
    Extract Playground human feedback pairs and build a DPO/ORPO JSONL dataset.
    Schema: {"prompt": str, "chosen": str, "rejected": str}
    """
    import json
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=503)

    try:
        logs = active_db.query(PlaygroundLog).filter(PlaygroundLog.rating.in_(["thumbs_up", "thumbs_down"])).all()
        if not logs:
            return JSONResponse({"status": "error", "message": "No rated comparisons found in Playground logs"}, status_code=400)

        out_path = PROJECT_ROOT / "data" / "preference_pairs.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pairs = []
        for l in logs:
            prompt = l.prompt.strip()
            if l.rating == "thumbs_up":
                chosen = l.tuned_model_response or ""
                rejected = l.base_model_response or ""
            else:  # thumbs_down
                # If operator provided corrected answer in notes, use that as chosen
                chosen = l.notes.strip() if (l.notes and l.notes.strip()) else (l.base_model_response or "")
                rejected = l.tuned_model_response or ""

            if prompt and chosen and rejected and chosen != rejected:
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        if not pairs:
            return JSONResponse({"status": "error", "message": "No valid distinct chosen/rejected pairs could be constructed"}, status_code=400)

        with open(out_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # Sync to MinIO S3 if available
        try:
            from web.services.storage_service import storage_service
            storage_service.upload_file(out_path, storage_service.bucket_datasets, "preference_pairs.jsonl")
        except Exception:
            pass

        return JSONResponse({
            "status": "success",
            "exported_count": len(pairs),
            "path": str(out_path),
            "message": f"Successfully exported {len(pairs)} preference pairs for DPO training",
        })
    finally:
        if should_close and active_db:
            active_db.close()


@router.post("/api/v1/playground/dpo/start")
async def start_dpo_training(
    model_id: str = Form("meta-llama/Meta-Llama-3-8B-Instruct"),
    beta: float = Form(0.1),
    learning_rate: float = Form(0.000005),
    epochs: int = Form(3),
):
    """Trigger Direct Preference Optimization (DPO) pipeline training."""
    if state.active_job is not None:
        return JSONResponse({"status": "error", "message": "Another job is already running"}, status_code=409)

    dataset_path = PROJECT_ROOT / "data" / "preference_pairs.jsonl"
    if not dataset_path.exists():
        return JSONResponse({
            "status": "error",
            "message": "Preference dataset not found. Export playground ratings first.",
        }, status_code=400)

    output_dir = PROJECT_ROOT / "output_dpo"
    run_id = f"dpo_{int(time.time())}"
    is_celery = is_celery_available() and celery_app

    # Persist Run entry in PostgreSQL
    try:
        db = SessionLocal() if SessionLocal else None
        if db:
            run_rec = Run(
                id=run_id,
                job_type=JobType.TRAINING_DPO.value,
                method="DPO",
                model_id=model_id,
                epochs=epochs,
                learning_rate=learning_rate,
                dataset_path=str(dataset_path),
                status="RUNNING",
                mode="celery" if is_celery else "subprocess",
            )
            db.add(run_rec)
            db.commit()
            db.close()
    except Exception:
        pass

    if is_celery:
        job = state.set_job(
            job_type=JobType.TRAINING_DPO,
            metadata={"run_id": run_id, "mode": "celery", "model_id": model_id, "beta": beta, "epochs": epochs},
        )
        await state.add_log(f"--- [DISPATCH] DPO training {run_id} dispatched to Celery cluster ---")
        celery_task = celery_app.send_task(
            "web.celery_app.task_train_dpo",
            args=[run_id, model_id, str(dataset_path), str(output_dir), beta, learning_rate, epochs],
            queue="training_queue",
        )
        job.metadata["celery_task_id"] = celery_task.id
        return JSONResponse({"status": "started", "job_id": run_id})

    args = [
        "--dataset-path", str(dataset_path),
        "--output-dir", str(output_dir),
        "--beta", str(beta),
        "--learning-rate", str(learning_rate),
        "--epochs", str(epochs),
    ]

    await process_mgr.run_pipeline_script(
        job_type=JobType.TRAINING_DPO,
        script_module="pipeline.train_dpo",
        args=args,
        cwd=PROJECT_ROOT,
        metadata={"run_id": run_id, "mode": "subprocess", "model_id": model_id, "beta": beta, "epochs": epochs},
    )

    job_id = state.active_job.job_id if state.active_job else run_id
    return JSONResponse({"status": "started", "job_id": job_id})

