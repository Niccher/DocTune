"""
web/services/metrics_service.py
-------------------------------
High-performance Prometheus 0.0.4 metrics collector and exporter.
Exposes real-time GPU thermals/VRAM, training telemetry, queue backlog,
host resource utilization, and HTTP request metrics for Grafana.
"""

import time
from typing import Dict, Any, List
from collections import defaultdict
from pathlib import Path

from pipeline.common import PROJECT_ROOT
from web.state import state
from web.services.system_service import get_cpu_info, get_ram_info, get_disk_info, get_gpu_info
from web.services.queue_manager import queue_manager
from web.services.config_service import config_service
from web.db.session import SessionLocal, HAS_SQLALCHEMY
from web.db.models import UploadedDocument, DatasetVersion, Run, Checkpoint, AuditLog


class MetricsService:
    """Zero-dependency Prometheus metric collector formatting output to OpenMetrics standard."""

    def __init__(self):
        self.start_time = time.time()
        self._http_requests: Dict[str, int] = defaultdict(int)

    def record_http_request(self, method: str, status_code: int):
        """Track endpoint invocations by method and HTTP status code."""
        key = f'{method.upper()}:{status_code}'
        self._http_requests[key] += 1

    def generate_prometheus_metrics(self) -> str:
        """Collect all active subsystem telemetry and compile into Prometheus 0.0.4 text format."""
        lines: List[str] = []

        # ── 1. Application & Uptime ─────────────────────────────────────────
        version = config_service.get_app_version()
        uptime_seconds = round(time.time() - self.start_time, 2)

        lines.append("# HELP doctune_app_info Application metadata and version.")
        lines.append("# TYPE doctune_app_info gauge")
        lines.append(f'doctune_app_info{{app="DocTune Studio",version="{version}"}} 1')

        lines.append("# HELP doctune_uptime_seconds Seconds since the DocTune web service started.")
        lines.append("# TYPE doctune_uptime_seconds counter")
        lines.append(f"doctune_uptime_seconds {uptime_seconds}")

        # ── 2. GPU Telemetry ────────────────────────────────────────────────
        gpu_info = get_gpu_info()
        gpu_count = int(gpu_info.get("device_count") or gpu_info.get("gpu_count") or len(gpu_info.get("devices", [])) or 0)
        if not gpu_info.get("has_gpu", False) and gpu_count == 0:
            gpu_count = 0

        lines.append("# HELP doctune_gpu_count Total physical GPUs detected on host.")
        lines.append("# TYPE doctune_gpu_count gauge")
        lines.append(f"doctune_gpu_count {gpu_count}")

        if gpu_count > 0:
            lines.append("# HELP doctune_gpu_memory_used_bytes GPU VRAM currently allocated in bytes.")
            lines.append("# TYPE doctune_gpu_memory_used_bytes gauge")
            lines.append("# HELP doctune_gpu_memory_total_bytes GPU total physical VRAM capacity in bytes.")
            lines.append("# TYPE doctune_gpu_memory_total_bytes gauge")
            lines.append("# HELP doctune_gpu_utilization_percent Core GPU compute utilization percentage.")
            lines.append("# TYPE doctune_gpu_utilization_percent gauge")
            lines.append("# HELP doctune_gpu_temperature_celsius Current GPU die temperature in Celsius.")
            lines.append("# TYPE doctune_gpu_temperature_celsius gauge")

            for dev_idx, dev in enumerate(gpu_info.get("devices", [])):
                name = str(dev.get("name") or f"GPU-{dev_idx}").replace('"', '\\"')
                if "used_gb" in dev:
                    used_bytes = int(float(dev.get("used_gb") or 0.0) * (1024 ** 3))
                else:
                    used_bytes = int(float(dev.get("memory_used_mb") or 0.0) * (1024 * 1024))

                if "total_gb" in dev:
                    total_bytes = int(float(dev.get("total_gb") or 0.0) * (1024 ** 3))
                else:
                    total_bytes = int(float(dev.get("memory_total_mb") or 0.0) * (1024 * 1024))

                util_pct = dev.get("gpu_util_pct") if dev.get("gpu_util_pct") is not None else dev.get("utilization_pct", 0)
                temp_c = dev.get("temperature_c")

                labels = f'device="{dev_idx}",gpu="{name}"'
                lines.append(f"doctune_gpu_memory_used_bytes{{{labels}}} {used_bytes}")
                lines.append(f"doctune_gpu_memory_total_bytes{{{labels}}} {total_bytes}")
                lines.append(f"doctune_gpu_utilization_percent{{{labels}}} {util_pct if util_pct is not None else 0}")
                if temp_c is not None:
                    lines.append(f"doctune_gpu_temperature_celsius{{{labels}}} {temp_c}")

        # ── 3. Host System (CPU, RAM, Disk) ──────────────────────────────────
        cpu = get_cpu_info()
        lines.append("# HELP doctune_cpu_cores Number of available CPU hardware threads.")
        lines.append("# TYPE doctune_cpu_cores gauge")
        lines.append(f"doctune_cpu_cores {cpu.get('cores', 1)}")

        lines.append("# HELP doctune_cpu_load System load average over 1m, 5m, and 15m.")
        lines.append("# TYPE doctune_cpu_load gauge")
        lines.append(f'doctune_cpu_load{{interval="1m"}} {cpu.get("load_avg_1m", 0.0)}')
        lines.append(f'doctune_cpu_load{{interval="5m"}} {cpu.get("load_avg_5m", 0.0)}')
        lines.append(f'doctune_cpu_load{{interval="15m"}} {cpu.get("load_avg_15m", 0.0)}')

        ram = get_ram_info()
        ram_total_bytes = int(ram.get("total_gb", 0) * (1024 ** 3))
        ram_used_bytes = int(ram.get("used_gb", 0) * (1024 ** 3))

        lines.append("# HELP doctune_ram_memory_used_bytes Host RAM memory allocated in bytes.")
        lines.append("# TYPE doctune_ram_memory_used_bytes gauge")
        lines.append(f"doctune_ram_memory_used_bytes {ram_used_bytes}")

        lines.append("# HELP doctune_ram_memory_total_bytes Host RAM total capacity in bytes.")
        lines.append("# TYPE doctune_ram_memory_total_bytes gauge")
        lines.append(f"doctune_ram_memory_total_bytes {ram_total_bytes}")

        lines.append("# HELP doctune_ram_memory_used_percent Host RAM utilization percentage.")
        lines.append("# TYPE doctune_ram_memory_used_percent gauge")
        lines.append(f"doctune_ram_memory_used_percent {ram.get('used_pct', 0.0)}")

        disk = get_disk_info()
        disk_total_bytes = int(disk.get("total_gb", 0) * (1024 ** 3))
        disk_used_bytes = int(disk.get("used_gb", 0) * (1024 ** 3))
        disk_free_bytes = int(disk.get("free_gb", 0) * (1024 ** 3))

        lines.append("# HELP doctune_disk_storage_used_bytes Storage volume bytes used.")
        lines.append("# TYPE doctune_disk_storage_used_bytes gauge")
        lines.append(f"doctune_disk_storage_used_bytes {disk_used_bytes}")

        lines.append("# HELP doctune_disk_storage_free_bytes Storage volume bytes free.")
        lines.append("# TYPE doctune_disk_storage_free_bytes gauge")
        lines.append(f"doctune_disk_storage_free_bytes {disk_free_bytes}")

        lines.append("# HELP doctune_disk_storage_used_percent Storage volume utilization percentage.")
        lines.append("# TYPE doctune_disk_storage_used_percent gauge")
        lines.append(f"doctune_disk_storage_used_percent {disk.get('used_pct', 0.0)}")

        # ── 4. Training Queue & Pipeline ─────────────────────────────────────
        is_active = 1 if state.active_job is not None else 0
        queue_items = queue_manager.get_queue()
        queue_depth = len(queue_items)

        lines.append("# HELP doctune_pipeline_active_jobs Indicator whether a training job is currently running (1 or 0).")
        lines.append("# TYPE doctune_pipeline_active_jobs gauge")
        lines.append(f"doctune_pipeline_active_jobs {is_active}")

        lines.append("# HELP doctune_queue_pending_jobs Number of pending training jobs waiting in FIFO queue.")
        lines.append("# TYPE doctune_queue_pending_jobs gauge")
        lines.append(f"doctune_queue_pending_jobs {queue_depth}")

        # Active Run Telemetry
        if state.active_job:
            run_id = (state.active_job.metadata.get("run_id") or state.active_job.job_id or "unknown").replace('"', '\\"')
            job_type = (state.active_job.job_type.value if hasattr(state.active_job.job_type, "value") else str(state.active_job.job_type)).replace('"', '\\"')

            latest_metric = state.loss_history[-1] if state.loss_history else {}
            step = latest_metric.get("step", 0)
            total_steps = latest_metric.get("total_steps", 0)
            loss = latest_metric.get("loss", 0.0)
            lr = latest_metric.get("learning_rate", 0.0)

            lines.append("# HELP doctune_training_step Current training step of active job.")
            lines.append("# TYPE doctune_training_step counter")
            lines.append(f'doctune_training_step{{run_id="{run_id}",type="{job_type}"}} {step}')

            lines.append("# HELP doctune_training_total_steps Target total training steps for active job.")
            lines.append("# TYPE doctune_training_total_steps gauge")
            lines.append(f'doctune_training_total_steps{{run_id="{run_id}",type="{job_type}"}} {total_steps}')

            lines.append("# HELP doctune_training_loss Most recent loss metric recorded.")
            lines.append("# TYPE doctune_training_loss gauge")
            lines.append(f'doctune_training_loss{{run_id="{run_id}",type="{job_type}"}} {loss}')

            lines.append("# HELP doctune_training_learning_rate Current active learning rate.")
            lines.append("# TYPE doctune_training_learning_rate gauge")
            lines.append(f'doctune_training_learning_rate{{run_id="{run_id}",type="{job_type}"}} {lr}')

        # ── 5. Database Assets & Audit Records ──────────────────────────────
        if HAS_SQLALCHEMY and SessionLocal:
            try:
                with SessionLocal() as db:
                    docs_count = db.query(UploadedDocument).count()
                    versions_count = db.query(DatasetVersion).count()
                    runs_count = db.query(Run).count()
                    checkpoints_count = db.query(Checkpoint).count()
                    audit_count = db.query(AuditLog).count()

                    lines.append("# HELP doctune_uploaded_documents_total Total private documents in repository.")
                    lines.append("# TYPE doctune_uploaded_documents_total gauge")
                    lines.append(f"doctune_uploaded_documents_total {docs_count}")

                    lines.append("# HELP doctune_dataset_versions_total Total immutable dataset version snapshots.")
                    lines.append("# TYPE doctune_dataset_versions_total gauge")
                    lines.append(f"doctune_dataset_versions_total {versions_count}")

                    lines.append("# HELP doctune_runs_history_total Total training runs recorded in database.")
                    lines.append("# TYPE doctune_runs_history_total gauge")
                    lines.append(f"doctune_runs_history_total {runs_count}")

                    lines.append("# HELP doctune_checkpoints_total Total saved model checkpoints.")
                    lines.append("# TYPE doctune_checkpoints_total gauge")
                    lines.append(f"doctune_checkpoints_total {checkpoints_count}")

                    lines.append("# HELP doctune_audit_events_total Total operational events logged in audit trail.")
                    lines.append("# TYPE doctune_audit_events_total counter")
                    lines.append(f"doctune_audit_events_total {audit_count}")
            except Exception:
                pass

        # ── 6. HTTP API Metrics ─────────────────────────────────────────────
        lines.append("# HELP doctune_http_requests_total Total HTTP API calls serviced since startup.")
        lines.append("# TYPE doctune_http_requests_total counter")
        if not self._http_requests:
            lines.append('doctune_http_requests_total{method="GET",status="200"} 0')
        else:
            for k, count in sorted(self._http_requests.items()):
                m, s = k.split(":")
                lines.append(f'doctune_http_requests_total{{method="{m}",status="{s}"}} {count}')

        return "\n".join(lines) + "\n"


metrics_service = MetricsService()
