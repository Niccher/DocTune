"""
web/templates_config.py
-----------------------
Centralized Jinja2 templates manager with global context helpers.
"""

from fastapi.templating import Jinja2Templates
from pipeline.common import PROJECT_ROOT
from web.services.config_service import config_service

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))

# Register dynamic global helpers available across all Jinja2 templates
templates.env.globals["get_app_name"] = config_service.get_app_name
templates.env.globals["get_app_version"] = config_service.get_app_version
templates.env.globals["get_settings"] = config_service.get_settings
templates.env.globals["get_current_user"] = lambda request: getattr(request.state, "user", None) if request and hasattr(request, "state") else None
templates.env.globals["is_auth_enabled"] = lambda: bool(config_service.get_auth_config().get("enabled", False))

