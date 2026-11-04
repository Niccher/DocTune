"""
web/routers/admin.py
--------------------
FastAPI endpoints for the System Administration dashboard,
settings persistence, credential verification, and path creation.
"""

from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from pipeline.common import PROJECT_ROOT, CONFIG_FILE
from web.services.config_service import config_service
from web.services.system_service import get_hardware_bounds
from web.templates_config import templates

router = APIRouter(tags=["admin"])



@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Render the System Administration page."""
    settings = config_service.get_settings()
    hardware_bounds = get_hardware_bounds()
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "active_page": "admin",
            "page_title": "System Administration & Configuration",
            "settings": settings,
            "hardware_bounds": hardware_bounds,
        },
    )


@router.get("/api/admin/hardware-bounds")
async def get_hardware_bounds_api():
    """Return physical host boundaries and safe ceilings."""
    return get_hardware_bounds()


@router.get("/api/admin/settings")

async def get_settings_api():
    """Return the active system configuration."""
    return config_service.get_settings()


@router.post("/api/admin/settings")
async def save_settings_api(payload: Dict[str, Any]):
    """Update and persist system configuration."""
    success, message, updated = config_service.save_settings(payload)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "success", "message": message, "settings": updated}


@router.post("/api/admin/reset")
async def reset_settings_api():
    """Reset configuration to factory defaults."""
    restored = config_service.reset_to_defaults()
    return {
        "status": "success",
        "message": "Configuration successfully restored to factory defaults.",
        "settings": restored,
    }


@router.get("/api/admin/export")
async def export_settings_api():
    """Download settings.json file."""
    if not CONFIG_FILE.exists():
        config_service.reset_to_defaults()
    return FileResponse(
        path=str(CONFIG_FILE),
        filename="fine_tuning_settings.json",
        media_type="application/json",
    )


@router.post("/api/admin/test-hf")
async def test_hf_token_api(payload: Dict[str, str]):
    """Validate a Hugging Face token."""
    token = payload.get("token", "")
    res = config_service.test_hf_token(token)
    return res


@router.post("/api/admin/check-paths")
async def check_paths_api(payload: Dict[str, Any] = None):
    """Verify and create storage directories."""
    storage_paths = payload.get("storage_paths") if payload else None
    res = config_service.verify_and_create_paths(storage_paths)
    return res


@router.post("/api/admin/test-webhook")
async def test_webhook_api(payload: Dict[str, str]):
    """Send test alert to webhook."""
    url = payload.get("url", "")
    res = config_service.test_webhook(url)
    return res
