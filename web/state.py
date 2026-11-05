"""
web/state.py
------------
Global application state, execution lock, and WebSocket connection manager.
"""

import asyncio
import time
from collections import deque
from enum import Enum
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from fastapi import WebSocket


class JobType(str, Enum):
    IDLE = "IDLE"
    DATA_PREP = "DATA_PREP"
    TRAINING_QLORA = "TRAINING_QLORA"
    TRAINING_LORA = "TRAINING_LORA"
    TRAINING_FFT = "TRAINING_FFT"
    MERGING = "MERGING"
    EVAL = "EVAL"


class JobStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


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
        self.active_job: Optional[JobInfo] = None
        self.gpu_lock = asyncio.Lock()
        self.log_buffer: deque = deque(maxlen=1000)
        self.loss_history: List[Dict[str, Any]] = []
        self.ws_clients: Set[WebSocket] = set()
        self.recent_jobs: List[Dict[str, Any]] = []

    def set_job(self, job_type: JobType, pid: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> JobInfo:
        job = JobInfo(
            job_id=f"{job_type.value.lower()}_{int(time.time())}",
            job_type=job_type,
            status=JobStatus.RUNNING,
            pid=pid,
            metadata=metadata or {},
        )
        self.active_job = job
        self.log_buffer.clear()
        self.loss_history.clear()
        return job

    def finish_job(self, status: JobStatus):
        if self.active_job:
            self.active_job.status = status
            self.active_job.end_time = time.time()
            self.recent_jobs.insert(0, self.active_job.to_dict())
            if len(self.recent_jobs) > 20:
                self.recent_jobs.pop()
            self.active_job = None

    async def connect_ws(self, ws: WebSocket):
        await ws.accept()
        self.ws_clients.add(ws)

        # Rehydrate client with existing buffer and loss metrics
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
        """Rehydrate recent_jobs and loss_history from PostgreSQL/SQLite on startup."""
        try:
            from datetime import datetime
            from web.db.session import SessionLocal
            from web.db.models import Run, MetricPoint
            if not SessionLocal:
                return

            db = SessionLocal()
            try:
                # 1. Clean up stale zombie jobs left RUNNING across restart
                stale_runs = db.query(Run).filter(Run.status.in_(["RUNNING", "PENDING"])).all()
                for r in stale_runs:
                    r.status = "ABORTED"
                    r.error_message = "Server rebooted while task was active."
                    r.completed_at = datetime.utcnow()
                if stale_runs:
                    db.commit()

                # 2. Rehydrate recent_jobs
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

                # 3. Rehydrate loss_history for latest training run
                latest_training = db.query(Run).filter(
                    Run.job_type.in_(["TRAINING_QLORA", "TRAINING_LORA", "TRAINING_FFT"])
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
        except Exception:
            # Rehydration notice or table absent on initial cold boot
            pass


# Global singleton instance
state = AppState()
