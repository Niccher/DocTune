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
from web.templates_config import templates

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
            "active_job": state.active_job.to_dict() if state.active_job else None,
        },
    )


@router.get("/api/dataset/preview", response_class=JSONResponse)
async def dataset_preview_api(offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    """API endpoint to fetch paginated JSONL dataset samples."""
    return load_dataset_samples(offset=offset, limit=limit)


@router.post("/api/documents/upload")
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


@router.post("/api/documents/delete")
async def delete_document(filename: str = Form(...)):
    target = (RAW_DOCS_DIR / filename).resolve()
    # Prevent directory traversal
    if str(target).startswith(str(RAW_DOCS_DIR.resolve())) and target.exists():
        target.unlink()
        return JSONResponse({"status": "deleted", "filename": filename})
    return JSONResponse({"status": "error", "message": "File not found or invalid path"}, status_code=400)


@router.post("/api/data-prep/start")
async def start_data_prep(
    teacher_url: str = Form("http://localhost:8000/v1"),
    teacher_model: str = Form("Qwen/Qwen2.5-72B-Instruct"),
    teacher_api_key: str = Form("not-needed"),
    max_chunks: int = Form(3),
    chunk_size: int = Form(1000),
    overlap: int = Form(150),
    workers: int = Form(12),
):
    if state.active_job is not None:
        return JSONResponse({"status": "error", "message": "Another job is already running"}, status_code=409)

    args = [
        "--teacher-url", teacher_url,
        "--teacher-model", teacher_model,
        "--teacher-api-key", teacher_api_key,
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
        metadata={"teacher_model": teacher_model, "teacher_url": teacher_url},
    )

    return JSONResponse({"status": "started", "job_id": state.active_job.job_id})
