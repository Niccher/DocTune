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
templates.env.globals["get_settings"] = config_service.get_settings
