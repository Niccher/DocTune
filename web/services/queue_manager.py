"""
web/services/queue_manager.py
-----------------------------
Sequential FIFO training job queue and resumable checkpoint discovery.
Supports queuing multiple fine-tuning runs and auto-advancing when jobs finish.
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from pipeline.common import PROJECT_ROOT, Config
from web.state import state, JobType, JobStatus

logger = logging.getLogger("doctune.queue")

QUEUE_FILE = PROJECT_ROOT / ".runs" / "job_queue.json"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueueManager:
    def __init__(self):
        self.queue: List[Dict[str, Any]] = []
        self._load_queue()

    def _load_queue(self):
        if QUEUE_FILE.exists():
            try:
                with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                    self.queue = json.load(f)
            except Exception as e:
                logger.error("Failed to load job queue from disk: %s", e)
                self.queue = []

    def _save_queue(self):
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.queue, f, indent=2)
        except Exception as e:
            logger.error("Failed to persist job queue: %s", e)

    def enqueue(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Add a job to the FIFO queue."""
        job_spec["status"] = "QUEUED"
        job_spec["queued_at"] = now_utc_iso()
        self.queue.append(job_spec)
        self._save_queue()

        # Update DB status to QUEUED if DB exists
        try:
            from web.db.session import SessionLocal
            from web.db.models import Run
            if SessionLocal:
                db = SessionLocal()
                try:
                    run = db.query(Run).filter(Run.id == job_spec.get("run_id")).first()
                    if run:
                        run.status = "QUEUED"
                        db.commit()
                except Exception:
                    pass
                finally:
                    db.close()
        except Exception:
            pass

        return {
            "status": "queued",
            "position": len(self.queue),
            "run_id": job_spec.get("run_id"),
            "job_spec": job_spec,
        }

    def get_queue(self) -> List[Dict[str, Any]]:
        return list(self.queue)

    def remove(self, run_id: str) -> bool:
        initial_len = len(self.queue)
        self.queue = [j for j in self.queue if j.get("run_id") != run_id]
        if len(self.queue) < initial_len:
            self._save_queue()
            try:
                from web.db.session import SessionLocal
                from web.db.models import Run
                if SessionLocal:
                    db = SessionLocal()
                    try:
                        run = db.query(Run).filter(Run.id == run_id).first()
                        if run and run.status == "QUEUED":
                            run.status = "CANCELLED"
                            db.commit()
                    except Exception:
                        pass
                    finally:
                        db.close()
            except Exception:
                pass
            return True
        return False

    def clear(self) -> int:
        count = len(self.queue)
        self.queue.clear()
        self._save_queue()
        return count

    async def dispatch_next(self) -> Optional[Dict[str, Any]]:
        """If no job is currently running, pops and starts the next job in the queue."""
        if state.active_job is not None:
            logger.info("Cannot dispatch next queued job: active job '%s' is running", state.active_job.job_id)
            return None

        if not self.queue:
            return None

        next_job = self.queue.pop(0)
        self._save_queue()

        logger.info("Auto-dispatching queued job '%s'", next_job.get("run_id"))
        from web.services.process_manager import process_mgr
        from web.celery_app import is_celery_available, app as celery_app

        run_id = next_job.get("run_id")
        method = next_job.get("method", "qlora").lower()
        model_id = next_job.get("model_id")
        cfg_dict = next_job.get("config", {})
        config_file = next_job.get("config_file")
        resume_checkpoint = next_job.get("resume_checkpoint")
        mode = next_job.get("mode", "subprocess")

        method_map = {
            "qlora": (JobType.TRAINING_QLORA, "pipeline.train_qlora"),
            "lora": (JobType.TRAINING_LORA, "pipeline.train_lora"),
            "fft": (JobType.TRAINING_FFT, "pipeline.train_fft"),
        }
        job_type, script_module = method_map.get(method, (JobType.TRAINING_QLORA, "pipeline.train_qlora"))

        # Update DB Run status to RUNNING
        try:
            from web.db.session import SessionLocal
            from web.db.models import Run
            if SessionLocal:
                db = SessionLocal()
                try:
                    run = db.query(Run).filter(Run.id == run_id).first()
                    if run:
                        run.status = "RUNNING"
                        run.started_at = datetime.now(timezone.utc)
                        db.commit()
                except Exception:
                    pass
                finally:
                    db.close()
        except Exception:
            pass

        # Dispatch via Celery or Subprocess
        if mode == "celery" and is_celery_available() and celery_app:
            job = state.set_job(
                job_type=job_type,
                metadata={
                    "method": method,
                    "model_id": model_id,
                    "run_id": run_id,
                    "mode": "celery",
                    "config_file": config_file,
                    "resume_checkpoint": resume_checkpoint,
                },
            )
            await state.add_log(f"--- [QUEUE DISPATCH] Job {run_id} ({job_type.value}) dispatched to Celery ---")
            task = celery_app.send_task(
                "web.celery_app.task_train_model",
                args=[run_id, method, model_id, cfg_dict],
                queue="training_queue",
            )
            job.metadata["celery_task_id"] = task.id
            return next_job

        # Subprocess dispatch
        args = ["--config-json", str(config_file)]
        if resume_checkpoint:
            args.extend(["--resume-from-checkpoint", str(resume_checkpoint)])

        await process_mgr.run_pipeline_script(
            job_type=job_type,
            script_module=script_module,
            args=args,
            cwd=PROJECT_ROOT,
            metadata={
                "method": method,
                "model_id": model_id,
                "run_id": run_id,
                "mode": "subprocess",
                "config_file": config_file,
                "resume_checkpoint": resume_checkpoint,
            },
        )
        await state.add_log(f"--- [QUEUE DISPATCH] Job {run_id} started via subprocess ---")
        return next_job

    def discover_resumable_checkpoints(self) -> List[Dict[str, Any]]:
        """Scan output directories for saved checkpoints that can be resumed."""
        cfg = Config()
        candidate_dirs = [
            ("QLoRA", cfg.output_qlora),
            ("LoRA", cfg.output_lora),
            ("FFT", cfg.output_fft),
        ]
        results = []
        for method, out_dir in candidate_dirs:
            if not out_dir.exists():
                continue

            for chk in sorted(out_dir.glob("checkpoint-*")):
                if not chk.is_dir():
                    continue

                step = None
                try:
                    step = int(chk.name.split("-")[-1])
                except Exception:
                    pass

                state_file = chk / "trainer_state.json"
                epoch = None
                last_loss = None
                if state_file.exists():
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            st = json.load(f)
                            epoch = st.get("epoch")
                            log_history = st.get("log_history", [])
                            if log_history:
                                last_loss = log_history[-1].get("loss")
                    except Exception:
                        pass

                results.append({
                    "method": method,
                    "name": chk.name,
                    "path": str(chk),
                    "step": step,
                    "epoch": epoch,
                    "last_loss": last_loss,
                    "modified": chk.stat().st_mtime,
                })

        results.sort(key=lambda x: x["modified"], reverse=True)
        return results


queue_manager = QueueManager()
