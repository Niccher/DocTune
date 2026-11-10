"""
web/state.py
------------
In-memory and database-synchronized application state.
Provides WebSocket broadcasting, real-time log streaming, and dynamic
job querying across Celery workers without single-process concurrency gating.
"""

import time
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Set
from fastapi import WebSocket

from web.db.session import SessionLocal, HAS_SQLALCHEMY
from web.db.models import Job, Run, MetricPoint, now_utc

logger = logging.getLogger("doctune.state")


class JobType(str, Enum):
    DATA_PREP = "DATA_PREP"
    TRAINING = "TRAINING"
    TRAINING_QLORA = "TRAINING_QLORA"
    TRAINING_LORA = "TRAINING_LORA"
    TRAINING_FFT = "TRAINING_FFT"
    EVAL = "EVAL"
    MERGE = "MERGE"
    GGUF_EXPORT = "GGUF_EXPORT"
    EXPORT_GGUF = "EXPORT_GGUF"
    DEPLOY = "DEPLOY"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    CANCELLED = "CANCELLED"


@dataclass
class JobInfo:
    job_id: str
    job_type: JobType
    status: JobStatus = JobStatus.RUNNING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    pid: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        duration = (self.end_time or time.time()) - self.start_time
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": round(duration, 1),
            "pid": self.pid,
            "metadata": self.metadata,
        }


class AppState:
    def __init__(self):
        self._local_job: Optional[JobInfo] = None
        self.gpu_lock = asyncio.Lock()
        self.log_buffer: deque = deque(maxlen=1000)
        self.loss_history: List[Dict[str, Any]] = []
        self.ws_clients: Set[WebSocket] = set()
        self.recent_jobs: List[Dict[str, Any]] = []

    @property
    def active_job(self) -> Optional[JobInfo]:
        """
        Dynamically query active jobs from PostgreSQL to ensure multi-worker
        and reboot awareness. Falls back to local subprocess tracker.
        """
        if self._local_job and self._local_job.status == JobStatus.RUNNING:
            return self._local_job

        if HAS_SQLALCHEMY and SessionLocal:
            try:
                db = SessionLocal()
                active = db.query(Job).filter(Job.status.in_(["RUNNING", "ASSIGNED"])).order_by(Job.started_at.desc()).first()
                if active:
                    start_ts = active.started_at.timestamp() if active.started_at else active.queued_at.timestamp()
                    # Resolve JobType enum
                    try:
                        j_type = JobType(active.type)
                    except Exception:
                        j_type = JobType.TRAINING_QLORA

                    job_info = JobInfo(
                        job_id=active.id,
                        job_type=j_type,
                        status=JobStatus.RUNNING,
                        start_time=start_ts,
                        metadata=active.spec_json or {},
                    )
                    db.close()
                    return job_info
                db.close()
            except Exception:
                pass

        return None

    @active_job.setter
    def active_job(self, value: Optional[JobInfo]):
        self._local_job = value

    def set_job(self, job_type: JobType, pid: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> JobInfo:
        """Register a local job execution."""
        meta = metadata or {}
        job_id = meta.get("job_id") or meta.get("run_id") or f"{job_type.value.lower()}_{int(time.time())}"
        job = JobInfo(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.RUNNING,
            pid=pid,
            metadata=meta,
        )
        self._local_job = job
        self.log_buffer.clear()
        self.loss_history.clear()
        return job

    def finish_job(self, status: JobStatus):
        """Mark local job finished."""
        if self._local_job:
            self._local_job.status = status
            self._local_job.end_time = time.time()
            self.recent_jobs.insert(0, self._local_job.to_dict())
            if len(self.recent_jobs) > 20:
                self.recent_jobs.pop()
            self._local_job = None

    async def connect_ws(self, ws: WebSocket):
        await ws.accept()
        self.ws_clients.add(ws)

        init_payload = {
            "type": "init",
            "active_job": self.active_job.to_dict() if self.active_job else None,
            "logs": list(self.log_buffer),
            "loss_history": self.loss_history,
        }
        await ws.send_json(init_payload)

    def disconnect_ws(self, ws: WebSocket):
        self.ws_clients.discard(ws)

    async def broadcast_ws(self, payload: Dict[str, Any]):
        if not self.ws_clients:
            return

        dead_clients = set()
        for ws in self.ws_clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead_clients.add(ws)

        for ws in dead_clients:
            self.ws_clients.discard(ws)

    async def add_log(self, line: str):
        self.log_buffer.append(line)
        await self.broadcast_ws({"type": "log", "line": line})

    def rehydrate_from_db(self):
        """Rehydrate state on system startup and reconcile orphaned tasks."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return

        try:
            from web.services.queue_manager import queue_manager
            # Reconcile any orphan jobs from earlier worker crashes
            queue_manager.reconcile_orphan_jobs(timeout_sec=0)

            db = SessionLocal()
            try:
                runs = db.query(Run).order_by(Run.created_at.desc()).limit(20).all()
                rehydrated = []
                for r in runs:
                    start_ts = r.started_at.timestamp() if r.started_at else (r.created_at.timestamp() if r.created_at else time.time())
                    end_ts = r.completed_at.timestamp() if r.completed_at else None
                    duration = r.duration_sec if r.duration_sec is not None else ((end_ts or time.time()) - start_ts)
                    rehydrated.append({
                        "job_id": r.id,
                        "run_id": r.id,
                        "job_type": r.job_type,
                        "status": r.status,
                        "start_time": start_ts,
                        "end_time": end_ts,
                        "duration_sec": round(duration, 1),
                        "pid": r.pid,
                        "metadata": {
                            "method": r.method,
                            "model_id": r.model_id,
                            "error_message": r.error_message,
                            "mode": r.mode,
                        },
                    })
                self.recent_jobs = rehydrated

                latest_training = db.query(Run).filter(
                    Run.job_type.in_(["TRAINING", "TRAINING_QLORA", "TRAINING_LORA", "TRAINING_FFT"])
                ).order_by(Run.created_at.desc()).first()

                if latest_training:
                    metrics = db.query(MetricPoint).filter(
                        MetricPoint.run_id == latest_training.id
                    ).order_by(MetricPoint.step.asc()).all()
                    self.loss_history = [
                        {"step": m.step, "loss": round(m.loss, 4), "epoch": round(m.epoch or 0.0, 4)}
                        for m in metrics
                    ]
            finally:
                db.close()
        except Exception as ex:
            logger.debug("State rehydration exception: %s", ex)


# Global singleton instance
state = AppState()
