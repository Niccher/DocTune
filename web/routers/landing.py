"""
web/routers/landing.py
----------------------
Public landing page suite for DocTune Studio:
- /: Primary Product Landing Page
- /features: Pipeline & Technical Features Deep Dive
- /security: Air-Gap Compliance & Data Privacy Overview
- /docs: Quickstart & REST API Reference Hub
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from web.templates_config import templates
from web.services.config_service import config_service

router = APIRouter(tags=["landing"])


@router.get("/", response_class=HTMLResponse)
async def landing_index(request: Request):
    """Render the primary DocTune Studio product landing page."""
    return templates.TemplateResponse(
        request=request,
        name="landing/index.html",
        context={
            "page_title": "Turn Private Documents into Production LLMs",
            "active_page": "landing",
            "app_name": config_service.get_app_name(),
        },
    )


@router.get("/features", response_class=HTMLResponse)
async def landing_features(request: Request):
    """Render the technical features and pipeline deep dive page."""
    return templates.TemplateResponse(
        request=request,
        name="landing/features.html",
        context={
            "page_title": "Pipeline Architecture & Features",
            "active_page": "features",
            "app_name": config_service.get_app_name(),
        },
    )


@router.get("/security", response_class=HTMLResponse)
async def landing_security(request: Request):
    """Render the air-gap compliance and data security page."""
    return templates.TemplateResponse(
        request=request,
        name="landing/security.html",
        context={
            "page_title": "Security & Air-Gap Compliance",
            "active_page": "security",
            "app_name": config_service.get_app_name(),
        },
    )


@router.get("/docs", response_class=HTMLResponse)
@router.get("/documentation", response_class=HTMLResponse)
async def landing_docs(request: Request):
    """Render the quickstart guide and API documentation page."""
    return templates.TemplateResponse(
        request=request,
        name="landing/docs.html",
        context={
            "page_title": "Quickstart & API Documentation",
            "active_page": "docs",
            "app_name": config_service.get_app_name(),
        },
    )
