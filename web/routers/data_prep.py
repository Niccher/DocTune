"""
web/routers/data_prep.py
------------------------
Document ingestion, file manager, and synthetic Q&A dataset generation studio.
"""

import os
import json
import shutil
from typing import List
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from pipeline.common import PROJECT_ROOT, RAW_DOCS_DIR, DEFAULT_JSONL
from web.state import state, JobType
from web.services.process_manager import process_mgr
from web.services.config_service import config_service
from web.services.teacher_service import teacher_service
from web.templates_config import templates
from pydantic import BaseModel


router = APIRouter()



def list_raw_documents():
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
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
        })
    return files


def load_dataset_samples(offset: int = 0, limit: int = 25):
    if not DEFAULT_JSONL.exists():
        return {"total": 0, "samples": []}

    samples = []
    total = 0
    with open(DEFAULT_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            total += 1
            if offset <= idx < offset + limit:
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
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


@router.post("/api/v1/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = []
    allowed_exts = {".pdf", ".docx", ".csv"}

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext in allowed_exts:
            dest = RAW_DOCS_DIR / f.filename
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            uploaded.append(f.filename)

    return JSONResponse({"status": "success", "uploaded": uploaded})


@router.post("/api/v1/documents/delete")
async def delete_document(filename: str = Form(...)):
    target = (RAW_DOCS_DIR / filename).resolve()
    # Prevent directory traversal
    if str(target).startswith(str(RAW_DOCS_DIR.resolve())) and target.exists():
        target.unlink()
        return JSONResponse({"status": "deleted", "filename": filename})
    return JSONResponse({"status": "error", "message": "File not found or invalid path"}, status_code=400)


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

    await process_mgr.run_pipeline_script(
        job_type=JobType.DATA_PREP,
        script_module="pipeline.prepare_data",
        args=args,
        cwd=PROJECT_ROOT,
        metadata={"teacher_model": final_model, "teacher_url": final_url, "provider": provider},
    )

    job_id = state.active_job.job_id if state.active_job else "job_data_prep"
    return JSONResponse({"status": "started", "job_id": job_id})

