"""
web/services/system_service.py
------------------------------
Hardware and system telemetry service querying CPU, RAM, Disk, and GPU.
"""

import os
import sys
import shutil
import platform
import warnings
from typing import Dict, Any, List
from pathlib import Path

from pipeline.common import PROJECT_ROOT

# GPU telemetry using nvidia-ml-py with fallback
HAS_PYNVML = False
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    try:
        import pynvml
        HAS_PYNVML = True
    except ImportError:
        try:
            import nvidia_ml_py as pynvml
            HAS_PYNVML = True
        except ImportError:
            HAS_PYNVML = False


def get_cpu_info() -> Dict[str, Any]:
    """Extract CPU model, cores, architecture, and load average."""
    model_name = platform.processor() or "Unknown CPU"

    # On Linux, inspect /proc/cpuinfo for accurate marketing brand name
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            with open(cpuinfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        model_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    cores = os.cpu_count() or 1
    arch = platform.machine() or "x86_64"

    try:
        load_1, load_5, load_15 = os.getloadavg()
    except Exception:
        load_1, load_5, load_15 = 0.0, 0.0, 0.0

    return {
        "model_name": model_name,
        "cores": cores,
        "architecture": arch,
        "load_avg_1m": round(load_1, 2),
        "load_avg_5m": round(load_5, 2),
        "load_avg_15m": round(load_15, 2),
    }


def get_ram_info() -> Dict[str, Any]:
    """Extract System Memory (RAM) usage from /proc/meminfo."""
    mem_total_kb = 0
    mem_avail_kb = 0

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            with open(meminfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_avail_kb = int(line.split()[1])
        except Exception:
            pass

    if mem_total_kb > 0:
        total_gb = round(mem_total_kb / (1024 ** 2), 2)
        avail_gb = round(mem_avail_kb / (1024 ** 2), 2)
        used_gb = round((mem_total_kb - mem_avail_kb) / (1024 ** 2), 2)
        pct = round((used_gb / total_gb) * 100, 1)
    else:
        total_gb, avail_gb, used_gb, pct = 0.0, 0.0, 0.0, 0.0

    return {
        "total_gb": total_gb,
        "used_gb": used_gb,
        "avail_gb": avail_gb,
        "used_pct": pct,
    }


def get_disk_info() -> Dict[str, Any]:
    """Extract disk space metrics for the project storage drive."""
    try:
        total_bytes, used_bytes, free_bytes = shutil.disk_usage(PROJECT_ROOT)
        total_gb = round(total_bytes / (1024 ** 3), 1)
        used_gb = round(used_bytes / (1024 ** 3), 1)
        free_gb = round(free_bytes / (1024 ** 3), 1)
        used_pct = round((used_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0.0
    except Exception:
        total_gb, used_gb, free_gb, used_pct = 0.0, 0.0, 0.0, 0.0

    return {
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "used_pct": used_pct,
        "is_low": free_gb < 20.0,
    }


def get_gpu_info() -> Dict[str, Any]:
    """Extract GPU telemetry with CPU/fallback safety."""
    if not HAS_PYNVML:
        return {
            "has_gpu": False,
            "device_count": 0,
            "devices": [],
            "primary": None,
            "message": "NVIDIA driver not detected or running in CPU mode.",
        }

    devices: List[Dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = round(mem_info.total / (1024 ** 3), 2)
            used_gb = round(mem_info.used / (1024 ** 3), 2)
            free_gb = round(mem_info.free / (1024 ** 3), 2)
            mem_pct = round((mem_info.used / mem_info.total) * 100, 1) if mem_info.total > 0 else 0

            try:
                temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp_c = None

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util_pct = util.gpu
            except Exception:
                gpu_util_pct = None

            devices.append({
                "index": i,
                "name": name,
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "memory_util_pct": mem_pct,
                "gpu_util_pct": gpu_util_pct,
                "temperature_c": temp_c,
            })

        pynvml.nvmlShutdown()
        return {
            "has_gpu": len(devices) > 0,
            "device_count": len(devices),
            "devices": devices,
            "primary": devices[0] if devices else None,
        }
    except Exception as e:
        return {
            "has_gpu": False,
            "device_count": 0,
            "devices": [],
            "primary": None,
            "message": f"GPU query error: {str(e)}",
        }


def get_system_telemetry() -> Dict[str, Any]:
    """Unified system inspection aggregating CPU, RAM, Disk, GPU, and Host."""
    return {
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "disk": get_disk_info(),
        "gpu": get_gpu_info(),
        "host": {
            "os": platform.system(),
            "kernel": platform.release(),
            "python": sys.version.split()[0],
            "hostname": platform.node(),
        },
    }


def get_hardware_bounds() -> Dict[str, Any]:
    """
    Calculate safe operational upper boundaries based on live physical host inspection.
    Used for backend sanitization, frontend input limits, and pre-flight checks.
    """
    cpu = get_cpu_info()
    ram = get_ram_info()
    disk = get_disk_info()
    gpu = get_gpu_info()

    host_cores = int(cpu.get("cores", 1))
    max_safe_cpu = max(1, host_cores - 1) if host_cores > 1 else 1

    total_ram_gb = float(ram.get("total_gb", 0.0))
    max_safe_ram = max(2, int(total_ram_gb - 2.0)) if total_ram_gb > 4.0 else max(1, int(total_ram_gb))

    gpu_count = int(gpu.get("device_count", 0))
    max_safe_gpus = max(0, gpu_count)

    free_disk_gb = float(disk.get("free_gb", 0.0))
    min_required_disk_gb = 15.0

    return {
        "host_cores": host_cores,
        "max_safe_cpu": max_safe_cpu,
        "total_ram_gb": total_ram_gb,
        "max_safe_ram": max_safe_ram,
        "detected_gpus": gpu_count,
        "max_safe_gpus": max_safe_gpus,
        "free_disk_gb": free_disk_gb,
        "min_required_disk_gb": min_required_disk_gb,
        "has_sufficient_disk": free_disk_gb >= min_required_disk_gb,
    }

