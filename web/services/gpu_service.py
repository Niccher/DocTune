"""
web/services/gpu_service.py
---------------------------
Compatibility wrapper forwarding GPU queries to unified system_service.
"""

from web.services.system_service import get_gpu_info as get_gpu_telemetry, get_system_telemetry

__all__ = ["get_gpu_telemetry", "get_system_telemetry"]
