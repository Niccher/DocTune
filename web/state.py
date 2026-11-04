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

    async def add_loss_point(self, step: int, loss: float, epoch: float):
        point = {"step": step, "loss": round(loss, 4), "epoch": epoch}
        self.loss_history.append(point)
        await self.broadcast_ws({"type": "metric", "data": point})


# Global singleton instance
state = AppState()
