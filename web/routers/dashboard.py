"""
web/routers/dashboard.py
------------------------
Overview dashboard route displaying rich system hardware telemetry,
pipeline health, and command center navigation.
"""

from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from pipeline.common import PROJECT_ROOT, RAW_DOCS_DIR, DEFAULT_JSONL, Config
from web.state import state
from web.services.system_service import get_system_telemetry
from web.templates_config import templates

router = APIRouter()



def get_pipeline_details():
    pdf_count = 0
    docx_count = 0
    csv_count = 0
    total_doc_bytes = 0

    if RAW_DOCS_DIR.exists():
        for p in RAW_DOCS_DIR.iterdir():
            if p.is_file():
                ext = p.suffix.lower()
                size = p.stat().st_size
                total_doc_bytes += size
                if ext == ".pdf":
                    pdf_count += 1
                elif ext == ".docx":
                    docx_count += 1
                elif ext == ".csv":
                    csv_count += 1

    total_docs = pdf_count + docx_count + csv_count

    jsonl_count = 0
    jsonl_bytes = 0
    if DEFAULT_JSONL.exists():
        try:
            jsonl_bytes = DEFAULT_JSONL.stat().st_size
            with open(DEFAULT_JSONL, "r", encoding="utf-8") as f:
                jsonl_count = sum(1 for _ in f)
        except Exception:
            jsonl_count = 0

    cfg = Config.load()

    def get_ckpt_meta(path: Path, is_peft: bool, action_label: str, action_url: str):
        exists = (path / "adapter_config.json").exists() if is_peft else (path / "config.json").exists()
        size_mb = 0
        modified = None
        if exists:
            try:
                files = list(path.glob("*"))
                size_mb = round(sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024), 1)
                modified = max(f.stat().st_mtime for f in files if f.is_file())
            except Exception:
                pass
        return {
            "name": path.name,
            "path": str(path),
            "exists": exists,
            "size_mb": size_mb,
            "modified": modified,
            "action_label": action_label,
            "action_url": action_url,
        }

    checkpoints_summary = [
        get_ckpt_meta(cfg.output_qlora, True, "Test in Playground", "/playground"),
        get_ckpt_meta(cfg.output_merged, False, "Serve with vLLM", "/models"),
        get_ckpt_meta(cfg.output_lora, True, "Merge Weights", "/models"),
        get_ckpt_meta(cfg.output_fft, False, "Inspect Weights", "/models"),
    ]

    return {
        "raw_docs_count": total_docs,
        "pdf_count": pdf_count,
        "docx_count": docx_count,
        "csv_count": csv_count,
        "raw_docs_size_kb": round(total_doc_bytes / 1024, 1),
        "qa_pairs_count": jsonl_count,
        "dataset_size_kb": round(jsonl_bytes / 1024, 1),
        "checkpoints": checkpoints_summary,
    }


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("/studio", response_class=HTMLResponse)
async def dashboard_view(request: Request):
    telemetry = get_system_telemetry()
    pipeline_data = get_pipeline_details()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "page_title": "Overview Dashboard",
            "active_page": "dashboard",
            "system": telemetry,
            "pipeline": pipeline_data,
            "active_job": state.active_job.to_dict() if state.active_job else None,
            "recent_jobs": state.recent_jobs,
        },
    )


@router.get("/api/v1/telemetry", response_class=JSONResponse)
async def telemetry_api():
    telemetry = get_system_telemetry()
    pipeline_data = get_pipeline_details()
    return {
        "system": telemetry,
        "pipeline": pipeline_data,
        "active_job": state.active_job.to_dict() if state.active_job else None,
    }
