"""
pipeline/trainer/callbacks.py
-----------------------------
Custom Trainer callbacks for live metric streaming, early stopping,
database heartbeat updates, and cooperative task cancellation.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("doctune.trainer.callbacks")

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object


class MetricLoggingCallback(TrainerCallback):
    """Outputs real-time step and evaluation metrics as structured JSON lines."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            payload = {
                "step": int(state.global_step),
                "epoch": round(float(state.epoch or 0.0), 4),
                **logs,
            }
            print(f"[METRIC_JSON] {json.dumps(payload)}", flush=True)


class HeartbeatAndCancelCallback(TrainerCallback):
    """
    Updates the database job heartbeat every 30 seconds and checks Redis
    for an asynchronous cancellation signal (cancel:<run_id>).
    """
    def __init__(self, run_id: str, job_id: Optional[str] = None):
        self.run_id = run_id
        self.job_id = job_id
        self.last_heartbeat = time.time()
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        # 1. Update worker heartbeat every 30s
        if now - self.last_heartbeat >= 30:
            self.last_heartbeat = now
            if self.job_id:
                try:
                    from web.services.queue_manager import queue_manager
                    queue_manager.update_heartbeat(self.job_id)
                except Exception:
                    pass

        # 2. Check cooperative cancellation signal
        try:
            import redis
            r = redis.Redis.from_url(self.redis_url, socket_timeout=0.3)
            if r.get(f"cancel:{self.run_id}"):
                logger.warning("Cancellation signal detected for run '%s'. Halting training.", self.run_id)
                control.should_training_stop = True
                control.should_save = True
        except Exception:
            pass
