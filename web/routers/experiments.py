"""
web/routers/experiments.py
--------------------------
Side-by-side run comparison studio.
Overlays multi-series training loss curves and benchmarks hyperparameter performance.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Request, Query, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from pipeline.common import PROJECT_ROOT
from web.templates_config import templates
from web.db.session import SessionLocal, get_db, resolve_session
from web.db.models import Run, MetricPoint, Checkpoint, EvaluationBenchmark
from web.routers.auth import get_project_id, require_project_role, scoped_query

router = APIRouter()


def get_all_runs_summary(project_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve runs from database scoped to project_id or fallback to .runs/ disk scan."""
    runs = []
    if SessionLocal:
        try:
            db = SessionLocal()
            try:
                query = db.query(Run)
                if project_id and hasattr(Run, "project_id"):
                    query = query.filter(Run.project_id == project_id)
                db_runs = query.order_by(Run.created_at.desc()).limit(limit).all()
                for r in db_runs:
                    runs.append({
                        "id": r.id,
                        "job_type": r.job_type,
                        "method": r.method,
                        "model_id": r.model_id,
                        "status": r.status,
                        "epochs": r.epochs,
                        "batch_size": r.batch_size,
                        "learning_rate": r.learning_rate,
                        "duration_sec": r.duration_sec,
                        "final_loss": r.final_loss,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "metrics_count": len(r.metrics) if r.metrics else 0,
                        "dataset_version": getattr(r, "dataset_version", None),
                    })
                return runs
            finally:
                db.close()
        except Exception:
            pass

    # Disk scan fallback from isolated artifacts/runs/
    artifacts_runs_dir = PROJECT_ROOT / "artifacts" / "runs"
    if artifacts_runs_dir.exists():
        for run_folder in sorted(artifacts_runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            if not run_folder.is_dir():
                continue
            manifest_file = run_folder / "manifest.json"
            config_file = run_folder / "config.json"
            meta = {}
            if manifest_file.exists():
                try:
                    import json
                    meta = json.loads(manifest_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif config_file.exists():
                try:
                    import json
                    meta = json.loads(config_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            runs.append({
                "id": run_folder.name,
                "job_type": meta.get("job_type", "TRAINING"),
                "method": meta.get("method", "QLORA"),
                "model_id": meta.get("model_id", "meta-llama/Meta-Llama-3-8B-Instruct"),
                "status": meta.get("status", "COMPLETED"),
                "epochs": meta.get("epochs", meta.get("num_train_epochs", 3)),
                "batch_size": meta.get("batch_size", meta.get("per_device_train_batch_size", 2)),
                "learning_rate": meta.get("learning_rate", meta.get("learning_rate_lora", 0.0002)),
                "created_at": meta.get("created_at"),
                "metrics_count": len(meta.get("metrics", [])) if "metrics" in meta else 0,
            })
    return runs


@router.get("/experiments", response_class=HTMLResponse)
async def experiments_view(request: Request):
    project_id = get_project_id(request)
    runs = get_all_runs_summary(project_id=project_id, limit=30)
    return templates.TemplateResponse(
        request=request,
        name="experiments.html",
        context={
            "page_title": "Run Comparison Studio",
            "active_page": "experiments",
            "runs": runs,
        },
    )


@router.get("/api/v1/experiments/runs", response_class=JSONResponse)
async def list_runs_api(request: Request, limit: int = Query(50, ge=1, le=100)):
    """Return recent training runs for multi-run comparison selection scoped to project."""
    project_id = get_project_id(request)
    return {"status": "success", "runs": get_all_runs_summary(project_id=project_id, limit=limit)}


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail_view(request: Request, run_id: str):
    """Render single run detail page with diagnostic card, telemetry, and deploy snippets."""
    project_id = get_project_id(request)
    run_data = None
    metrics_data = []
    diagnostic = None

    if SessionLocal:
        db = SessionLocal()
        try:
            r = db.query(Run).filter(Run.id == run_id).first()
            if r and (not hasattr(r, "project_id") or not r.project_id or r.project_id == project_id):
                run_data = {
                    "id": r.id,
                    "job_type": r.job_type,
                    "method": r.method,
                    "model_id": r.model_id,
                    "status": r.status,
                    "epochs": r.epochs,
                    "batch_size": r.batch_size,
                    "learning_rate": r.learning_rate,
                    "duration_sec": r.duration_sec,
                    "final_loss": r.final_loss,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "dataset_version": getattr(r, "dataset_version", None),
                    "dataset_hash": getattr(r, "dataset_hash", None),
                    "error_code": getattr(r, "error_code", None),
                    "error_details": getattr(r, "error_details", None),
                }

                if getattr(r, "error_code", None) or getattr(r, "error_details", None):
                    try:
                        from pipeline.errors import classify_training_error
                        fake_err = Exception(r.error_details or r.error_code or "Unknown training failure")
                        diagnostic = classify_training_error(fake_err, context={"model_id": r.model_id})
                        if r.error_code:
                            diagnostic["code"] = r.error_code
                    except Exception:
                        pass

                pts = (
                    db.query(MetricPoint)
                    .filter(MetricPoint.run_id == run_id)
                    .order_by(MetricPoint.step.asc())
                    .all()
                )
                metrics_data = [{"step": p.step, "loss": round(p.loss, 4)} for p in pts]
        finally:
            db.close()

    if not run_data:
        run_folder = PROJECT_ROOT / "artifacts" / "runs" / run_id
        if run_folder.exists() and run_folder.is_dir():
            manifest_file = run_folder / "manifest.json"
            meta = {}
            if manifest_file.exists():
                try:
                    import json
                    meta = json.loads(manifest_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            run_data = {
                "id": run_id,
                "job_type": meta.get("job_type", "TRAINING"),
                "method": meta.get("method", "QLORA"),
                "model_id": meta.get("model_id", "meta-llama/Meta-Llama-3-8B-Instruct"),
                "status": meta.get("status", "COMPLETED"),
                "epochs": meta.get("epochs", 3),
                "batch_size": meta.get("batch_size", 2),
                "learning_rate": meta.get("learning_rate", 0.0002),
                "duration_sec": meta.get("duration_sec", 0),
                "final_loss": meta.get("final_loss"),
                "dataset_version": meta.get("dataset_version_id"),
                "dataset_hash": meta.get("dataset_sha256"),
                "error_code": None,
                "error_details": None,
            }

    if not run_data:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in current project.")

    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "page_title": f"Run {run_id}",
            "active_page": "experiments",
            "run": run_data,
            "metrics": metrics_data,
            "diagnostic": diagnostic,
        },
    )


@router.get("/api/v1/experiments/compare", response_class=JSONResponse)
async def compare_runs_api(request: Request, run_ids: str = Query("", description="Comma-separated run IDs to compare")):
    """
    Return aligned telemetry curves (step vs loss) and hyperparameter benchmark tables
    for multiple runs overlaid on the same canvas, strictly scoped to current project.
    """
    project_id = get_project_id(request)
    selected_ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
    if not selected_ids:
        return JSONResponse({"status": "error", "message": "No run_ids provided"}, status_code=400)

    series = []
    table_rows = []

    if SessionLocal:
        db = SessionLocal()
        try:
            for rid in selected_ids:
                run = db.query(Run).filter(Run.id == rid).first()
                if not run:
                    continue
                # Cross-tenant boundary check
                if hasattr(run, "project_id") and run.project_id and run.project_id != project_id:
                    continue

                # Fetch metric points
                pts = (
                    db.query(MetricPoint)
                    .filter(MetricPoint.run_id == rid)
                    .order_by(MetricPoint.step.asc())
                    .all()
                )

                # Fetch benchmark evaluation if exists
                bench = db.query(EvaluationBenchmark).filter(EvaluationBenchmark.run_id == rid).first()

                data_points = [{"x": p.step, "y": round(p.loss, 4)} for p in pts]
                series_name = f"{rid} ({run.method or 'LoRA'}, lr={run.learning_rate})"
                series.append({
                    "name": series_name,
                    "run_id": rid,
                    "data": data_points,
                })

                table_rows.append({
                    "run_id": rid,
                    "method": run.method,
                    "model_id": run.model_id,
                    "status": run.status,
                    "epochs": run.epochs,
                    "batch_size": run.batch_size,
                    "learning_rate": run.learning_rate,
                    "final_loss": run.final_loss or (data_points[-1]["y"] if data_points else None),
                    "duration_sec": run.duration_sec,
                    "dataset_version": getattr(run, "dataset_version", None) or "Active",
                    "bleu_score": bench.bleu_score if bench else None,
                    "rouge_l": bench.rouge_l if bench else None,
                    "perplexity": bench.perplexity if bench else None,
                })
        finally:
            db.close()

    # Return real database series only. Do not fabricate mock metrics.
    return {
        "status": "success",
        "run_count": len(series),
        "series": series,
        "table": table_rows,
    }


@router.get("/api/v1/experiments/rag-benchmark", response_class=JSONResponse)
@router.post("/api/v1/experiments/rag-benchmark", response_class=JSONResponse)
async def trigger_rag_benchmark_api(_role=Depends(require_project_role("operator"))):
    """Run empirical 4-way RAG vs Fine-Tuning benchmark on enterprise documents."""
    from pipeline.rag_benchmark import run_rag_benchmark
    res = run_rag_benchmark()
    return JSONResponse(res)


