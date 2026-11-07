"""
web/routers/data_prep.py
------------------------
Document ingestion, file manager, and synthetic Q&A dataset generation studio.
"""

import os
import time
import json
import shutil
import hashlib
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File, Form, Query, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from pipeline.common import PROJECT_ROOT, RAW_DOCS_DIR, DEFAULT_JSONL
from web.state import state, JobType
from web.services.process_manager import process_mgr
from web.services.config_service import config_service
from web.services.teacher_service import teacher_service
from web.templates_config import templates
from web.celery_app import is_celery_available, app as celery_app
from web.db.session import SessionLocal, get_db, resolve_session
from web.db.models import UploadedDocument, DatasetSample
from web.services.storage_service import storage_service
from pydantic import BaseModel


router = APIRouter()



def list_raw_documents():
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Try PostgreSQL / SQLite query
    try:
        if SessionLocal:
            db = SessionLocal()
            try:
                docs = db.query(UploadedDocument).order_by(UploadedDocument.uploaded_at.desc()).all()
                if docs:
                    return [
                        {
                            "name": d.filename,
                            "size_kb": round((d.size_bytes or 0) / 1024, 1),
                            "modified": d.uploaded_at.timestamp() if d.uploaded_at else time.time(),
                            "extension": d.file_format or Path(d.filename).suffix.lower(),
                            "status": d.status,
                            "hash": d.file_hash,
                        }
                        for d in docs
                    ]
            finally:
                db.close()
    except Exception:
        pass

    # 2. Disk scan fallback
    files = []
    patterns = ["*.pdf", "*.docx", "*.csv", "*.PDF", "*.DOCX", "*.CSV"]
    found = []
    for pat in patterns:
        found.extend(RAW_DOCS_DIR.glob(pat))

    for p in sorted(set(found)):
        stat = p.stat()
        files.append({
            "name": p.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": stat.st_mtime,
            "extension": p.suffix.lower(),
            "status": "READY",
        })
    return files


def load_dataset_samples(offset: int = 0, limit: int = 25, status: Optional[str] = None):
    # 1. Try DB query
    try:
        if SessionLocal:
            db = SessionLocal()
            try:
                query = db.query(DatasetSample)
                if status:
                    query = query.filter(DatasetSample.status == status)
                total = query.count()
                if total > 0:
                    samples_orm = query.order_by(DatasetSample.created_at.asc()).offset(offset).limit(limit).all()
                    samples = [
                        {
                            "id": s.id,
                            "status": s.status,
                            "quality_score": s.quality_score,
                            "messages": [
                                {"role": "system", "content": s.system_prompt or ""},
                                {"role": "user", "content": s.user_instruction},
                                {"role": "assistant", "content": s.assistant_response},
                            ],
                        }
                        for s in samples_orm
                    ]
                    return {"total": total, "samples": samples, "offset": offset, "limit": limit}
            finally:
                db.close()
    except Exception:
        pass

    # 2. Disk fallback & seed DB
    if not DEFAULT_JSONL.exists():
        storage_service.download_file(storage_service.bucket_datasets, "private_docs.jsonl", DEFAULT_JSONL)

    if not DEFAULT_JSONL.exists():
        return {"total": 0, "samples": []}

    samples = []
    total = 0
    all_raw = []
    with open(DEFAULT_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total += 1
            try:
                data = json.loads(line.strip())
                all_raw.append(data)
                if offset <= idx < offset + limit:
                    samples.append(data)
            except Exception:
                pass

    # Seed DB if empty
    if total > 0 and SessionLocal:
        try:
            db = SessionLocal()
            try:
                if db.query(DatasetSample).count() == 0:
                    for i, item in enumerate(all_raw):
                        msgs = item.get("messages", [])
                        sys_p = next((m.get("content", "") for m in msgs if m.get("role") == "system"), None)
                        usr_p = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                        ast_p = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                        if usr_p:
                            sample_obj = DatasetSample(
                                id=f"sample_{i+1:05d}",
                                system_prompt=sys_p,
                                user_instruction=usr_p,
                                assistant_response=ast_p,
                                status="APPROVED",
                            )
                            db.add(sample_obj)
                    db.commit()
            finally:
                db.close()
        except Exception:
            pass

    return {"total": total, "samples": samples, "offset": offset, "limit": limit}


@router.get("/data-prep", response_class=HTMLResponse)
async def data_prep_view(request: Request, page: int = Query(1, ge=1)):
    limit = 15
    offset = (page - 1) * limit

    files = list_raw_documents()
    dataset_info = load_dataset_samples(offset=offset, limit=limit)
    total_pages = max(1, (dataset_info["total"] + limit - 1) // limit)
    settings = config_service.get_settings()
    presets = teacher_service.get_presets()

    return templates.TemplateResponse(
        request=request,
        name="data_prep.html",
        context={
            "page_title": "Document Ingestion & Dataset Studio",
            "active_page": "data_prep",
            "files": files,
            "samples": dataset_info["samples"],
            "total_samples": dataset_info["total"],
            "current_page": page,
            "total_pages": total_pages,
            "settings": settings,
            "presets": presets,
            "active_job": state.active_job.to_dict() if state.active_job else None,
        },
    )


@router.get("/api/v1/dataset/preview", response_class=JSONResponse)
async def dataset_preview_api(offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    """API endpoint to fetch paginated JSONL dataset samples."""
    return load_dataset_samples(offset=offset, limit=limit)


@router.get("/api/v1/dataset/samples", response_class=JSONResponse)
async def get_dataset_samples_api(
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    status: Optional[str] = Query(None),
):
    """API endpoint to query dataset samples with status filter and pagination."""
    data = load_dataset_samples(offset=offset, limit=limit, status=status)
    return JSONResponse({"status": "success", **data})



@router.get("/api/v1/teacher/presets", response_class=JSONResponse)
async def get_teacher_presets_api():
    """Return all provider presets with host/docker URL mappings."""
    return {"presets": teacher_service.get_presets()}


@router.get("/api/v1/teacher/models", response_class=JSONResponse)
async def get_teacher_models_api(
    provider: str = Query("ollama"),
    base_url: str = Query(""),
    api_key: str = Query(""),
):
    """Dynamically discover pulled/available models from the target runner or API."""
    if not base_url:
        preset = teacher_service.get_presets().get(provider, {})
        base_url = preset.get("effective_base_url", "http://localhost:11434/v1")
    models = await teacher_service.fetch_available_models(provider=provider, base_url=base_url, api_key=api_key)
    return {"status": "success", "models": models}


@router.post("/api/v1/teacher/test", response_class=JSONResponse)
async def test_teacher_connection_api(request: Request):
    """Live verification and latency benchmarking of the teacher LLM endpoint."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    provider = body.get("provider", "ollama")
    base_url = body.get("base_url") or body.get("teacher_url", "")
    api_key = body.get("api_key") or body.get("teacher_api_key", "")
    model = body.get("model") or body.get("teacher_model", "")

    if not base_url:
        preset = teacher_service.get_presets().get(provider, {})
        base_url = preset.get("effective_base_url", "http://localhost:11434/v1")

    result = await teacher_service.test_connection(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
    return JSONResponse(result)


@router.post("/api/v1/teacher/pull", response_class=JSONResponse)
async def pull_teacher_model_api(request: Request):
    """Trigger background download/pull of a teacher model into Ollama or llama.cpp."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    provider = body.get("provider", "ollama")
    base_url = body.get("base_url") or body.get("teacher_url", "")
    api_key = body.get("api_key") or body.get("teacher_api_key", "")
    model = body.get("model") or body.get("teacher_model", "")

    if not model or not model.strip():
        return JSONResponse({"status": "error", "message": "Model name is required to pull."}, status_code=400)

    if not base_url:
        preset = teacher_service.get_presets().get(provider, {})
        base_url = preset.get("effective_base_url", "http://localhost:11434/v1")

    res = teacher_service.start_pull_task(
        provider=provider,
        base_url=base_url,
        model=model.strip(),
        api_key=api_key.strip() if api_key else "",
    )
    return JSONResponse(res)


@router.get("/api/v1/teacher/pull-status", response_class=JSONResponse)
async def get_teacher_pull_status_api(
    provider: str = Query("ollama"),
    model: str = Query(...),
):
    """Poll live download/pull status of a model."""
    status_info = teacher_service.get_pull_status(provider=provider, model=model)
    return JSONResponse(status_info)


@router.post("/api/v1/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = []
    allowed_exts = {".pdf", ".docx", ".csv"}

    db = SessionLocal() if SessionLocal else None
    try:
        for f in files:
            ext = Path(f.filename).suffix.lower()
            if ext in allowed_exts:
                dest = RAW_DOCS_DIR / f.filename
                hasher = hashlib.sha256()
                with open(dest, "wb") as out:
                    while chunk := f.file.read(65536):
                        hasher.update(chunk)
                        out.write(chunk)
                file_hash = hasher.hexdigest()
                file_size = dest.stat().st_size
                uploaded.append(f.filename)

                # Object storage upload (MinIO S3 or transparent .storage fallback)
                s3_bucket = storage_service.bucket_docs
                s3_key = f.filename
                storage_service.upload_file(dest, s3_bucket, s3_key)

                # PostgreSQL metadata record
                if db:
                    doc_id = f"doc_{file_hash[:16]}"
                    existing = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id).first()
                    if existing:
                        existing.filename = f.filename
                        existing.disk_path = str(dest)
                        existing.size_bytes = file_size
                        existing.status = "READY"
                        existing.s3_bucket = s3_bucket
                        existing.s3_key = s3_key
                    else:
                        doc_rec = UploadedDocument(
                            id=doc_id,
                            filename=f.filename,
                            disk_path=str(dest),
                            file_hash=file_hash,
                            file_format=ext,
                            status="READY",
                            size_bytes=file_size,
                            s3_bucket=s3_bucket,
                            s3_key=s3_key,
                        )
                        db.add(doc_rec)
                    db.commit()
    finally:
        if db:
            db.close()

    return JSONResponse({"status": "success", "uploaded": uploaded})


@router.post("/api/v1/documents/delete")
async def delete_document(filename: str = Form(...)):
    target = (RAW_DOCS_DIR / filename).resolve()
    # Prevent directory traversal
    if str(target).startswith(str(RAW_DOCS_DIR.resolve())) and target.exists():
        target.unlink()

        # Delete from MinIO/S3 or local fallback
        storage_service.delete_file(storage_service.bucket_docs, filename)

        # Delete from PostgreSQL
        if SessionLocal:
            try:
                db = SessionLocal()
                db.query(UploadedDocument).filter(UploadedDocument.filename == filename).delete()
                db.commit()
                db.close()
            except Exception:
                pass

        return JSONResponse({"status": "deleted", "filename": filename})
    return JSONResponse({"status": "error", "message": "File not found or invalid path"}, status_code=400)


@router.get("/api/v1/documents/{filename}/download-url")
async def get_document_download_url(filename: str):
    """Generate a presigned download URL for a raw document or serve file."""
    if storage_service.is_s3_enabled():
        url = storage_service.generate_presigned_download_url(
            bucket=storage_service.bucket_docs,
            s3_key=filename,
            expires_in=3600,
        )
        if url:
            return JSONResponse({"status": "success", "url": url, "filename": filename})
    dest = RAW_DOCS_DIR / filename
    if dest.exists():
        return JSONResponse({"status": "local", "path": str(dest), "filename": filename})
    return JSONResponse({"status": "error", "message": "Document not found"}, status_code=404)


class SampleEditPayload(BaseModel):
    user_instruction: Optional[str] = None
    assistant_response: Optional[str] = None
    status: Optional[str] = None


@router.put("/api/v1/dataset/samples/{sample_id}")
async def edit_dataset_sample(sample_id: str, payload: SampleEditPayload, db: Session = Depends(get_db)):
    """Edit instruction, assistant response, or status in PostgreSQL."""
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=503)

    try:
        sample = active_db.query(DatasetSample).filter(DatasetSample.id == sample_id).first()
        if not sample:
            return JSONResponse({"status": "error", "message": "Sample not found"}, status_code=404)
        if payload.user_instruction is not None:
            sample.user_instruction = payload.user_instruction
            sample.status = "EDITED"
        if payload.assistant_response is not None:
            sample.assistant_response = payload.assistant_response
            sample.status = "EDITED"
        if payload.status is not None:
            sample.status = payload.status
        active_db.commit()
        return JSONResponse({
            "status": "success",
            "sample_id": sample_id,
            "sample": {
                "id": sample.id,
                "user_instruction": sample.user_instruction,
                "assistant_response": sample.assistant_response,
                "status": sample.status,
            },
        })
    finally:
        if should_close and active_db:
            active_db.close()


@router.post("/api/v1/dataset/samples/{sample_id}/status")
async def update_sample_status(request: Request, sample_id: str, db: Session = Depends(get_db)):
    """Approve or reject a synthetic instruction sample."""
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=503)

    status_val = ""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        status_val = body.get("status", "")
    else:
        form = await request.form()
        status_val = form.get("status", "")

    if not status_val:
        return JSONResponse({"status": "error", "message": "Missing status parameter"}, status_code=400)

    try:
        sample = active_db.query(DatasetSample).filter(DatasetSample.id == sample_id).first()
        if not sample:
            return JSONResponse({"status": "error", "message": "Sample not found"}, status_code=404)
        sample.status = status_val.upper()
        active_db.commit()
        return JSONResponse({"status": "success", "sample_id": sample_id, "sample_status": sample.status})
    finally:
        if should_close and active_db:
            active_db.close()



@router.post("/api/v1/dataset/export-jsonl")
async def export_curated_dataset(db: Session = Depends(get_db)):
    """Export all approved and edited samples from PostgreSQL to private_docs.jsonl."""
    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable"}, status_code=503)

    try:
        samples = active_db.query(DatasetSample).filter(DatasetSample.status != "REJECTED").order_by(DatasetSample.created_at.asc()).all()
        if not samples:
            return JSONResponse({"status": "error", "message": "No approved samples to export"}, status_code=400)

        DEFAULT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(DEFAULT_JSONL, "w", encoding="utf-8") as f:
            for s in samples:
                msgs = []
                if s.system_prompt:
                    msgs.append({"role": "system", "content": s.system_prompt})
                msgs.append({"role": "user", "content": s.user_instruction})
                msgs.append({"role": "assistant", "content": s.assistant_response})
                f.write(json.dumps({"messages": msgs}) + "\n")
                count += 1

        # Upload to MinIO/S3 datasets bucket
        storage_service.upload_file(DEFAULT_JSONL, storage_service.bucket_datasets, "private_docs.jsonl")

        return JSONResponse({
            "status": "success",
            "exported_count": count,
            "path": str(DEFAULT_JSONL),
            "s3_bucket": storage_service.bucket_datasets,
            "s3_key": "private_docs.jsonl",
        })
    finally:
        if should_close and active_db:
            active_db.close()


@router.get("/api/v1/dataset/download-url")
async def get_dataset_download_url():
    """Generate a presigned download URL for the instruction dataset."""
    if storage_service.is_s3_enabled():
        url = storage_service.generate_presigned_download_url(
            bucket=storage_service.bucket_datasets,
            s3_key="private_docs.jsonl",
            expires_in=3600,
        )
        if url:
            return JSONResponse({"status": "success", "url": url, "filename": "private_docs.jsonl"})
    if DEFAULT_JSONL.exists():
        return JSONResponse({"status": "local", "path": str(DEFAULT_JSONL), "filename": "private_docs.jsonl"})
    return JSONResponse({"status": "error", "message": "Dataset not found"}, status_code=404)


@router.post("/api/v1/data-prep/start")
async def start_data_prep(
    teacher_provider: str = Form(""),
    teacher_url: str = Form(""),
    teacher_model: str = Form(""),
    teacher_api_key: str = Form(""),
    max_chunks: int = Form(3),
    chunk_size: int = Form(1000),
    overlap: int = Form(150),
    workers: int = Form(12),
):
    if state.active_job is not None:
        return JSONResponse({"status": "error", "message": "Another job is already running"}, status_code=409)

    # Fallback to configured settings if parameters are omitted or blank
    settings = config_service.get_settings()
    qa_cfg = settings.get("data_prep", {}).get("qa_generator", {})
    provider = teacher_provider or qa_cfg.get("provider", "ollama")
    preset = teacher_service.get_presets().get(provider, {})

    raw_url = teacher_url or qa_cfg.get("base_url") or preset.get("effective_base_url", "http://localhost:11434/v1")
    final_url = teacher_service.normalize_base_url(provider, raw_url)
    final_model = teacher_model or qa_cfg.get("model") or preset.get("default_model", "llama3.1:latest")
    final_api_key = (teacher_api_key.strip() if teacher_api_key else "") or (qa_cfg.get("api_key", "").strip() if qa_cfg.get("api_key") else "") or "not-needed"


    args = [
        "--teacher-url", final_url,
        "--teacher-model", final_model,
        "--teacher-api-key", final_api_key,
        "--max-chunks", str(max_chunks),
        "--chunk-size", str(chunk_size),
        "--overlap", str(overlap),
        "--workers", str(workers),
    ]

    if is_celery_available() and celery_app:
        run_id = f"prep_{int(time.time())}"
        job = state.set_job(
            job_type=JobType.DATA_PREP,
            metadata={"teacher_model": final_model, "teacher_url": final_url, "provider": provider, "run_id": run_id, "mode": "celery"},
        )
        await state.add_log(f"--- [DISPATCH] Data preparation job {run_id} dispatched to Celery cluster ---")
        celery_task = celery_app.send_task(
            "web.celery_app.task_prepare_data",
            args=[run_id, final_url, final_model, final_api_key, max_chunks, chunk_size, overlap, workers],
            queue="data_prep_queue",
        )
        job.metadata["celery_task_id"] = celery_task.id
        return JSONResponse({"status": "started", "job_id": run_id})

    await process_mgr.run_pipeline_script(
        job_type=JobType.DATA_PREP,
        script_module="pipeline.prepare_data",
        args=args,
        cwd=PROJECT_ROOT,
        metadata={"teacher_model": final_model, "teacher_url": final_url, "provider": provider},
    )

    job_id = state.active_job.job_id if state.active_job else "job_data_prep"
    return JSONResponse({"status": "started", "job_id": job_id})


