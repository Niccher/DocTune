"""
web/routers/models.py
---------------------
Model checkpoints manager and adapter merging utility.
"""

import json
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from pipeline.common import PROJECT_ROOT, Config
from web.state import state, JobType
from web.services.process_manager import process_mgr

router = APIRouter()
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))


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
    cfg = Config()
    checkpoints = {
        "qlora": get_checkpoint_details(cfg.output_qlora),
        "lora": get_checkpoint_details(cfg.output_lora),
        "fft": get_checkpoint_details(cfg.output_fft),
        "merged": get_checkpoint_details(cfg.output_merged),
    }

    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={
            "page_title": "Checkpoints & Model Hub",
            "active_page": "models",
            "checkpoints": checkpoints,
            "cfg": cfg.to_dict(),
            "active_job": state.active_job.to_dict() if state.active_job else None,
        },
    )


@router.post("/api/models/merge")
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
        metadata={"adapter_dir": adapter_dir, "export_dir": export_dir},
    )

    return JSONResponse({"status": "started", "job_id": state.active_job.job_id})
