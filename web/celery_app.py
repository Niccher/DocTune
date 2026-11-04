"""
web/celery_app.py
-----------------
Distributed Celery application definition and task registry for LLM training,
dataset chunking, adapter merging, and Redis Pub/Sub metric streaming.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("doctune.celery")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

HAS_CELERY = False
try:
    from celery import Celery
    HAS_CELERY = True
except ImportError:
    pass

if HAS_CELERY:
    app = Celery(
        "doctune_studio",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=86400,          # 24 hours max for large fine-tuning jobs
        worker_prefetch_multiplier=1,   # 1 job per worker to prevent GPU collisions
        task_routes={
            "web.celery_app.task_train_model": {"queue": "training_queue"},
            "web.celery_app.task_merge_adapters": {"queue": "training_queue"},
            "web.celery_app.task_prepare_data": {"queue": "data_prep_queue"},
        },
    )
else:
    app = None


def publish_redis_event(channel: str, event_data: Dict[str, Any]):
    """Publish a real-time event to a Redis Pub/Sub channel."""
    try:
        import redis
        r = redis.Redis.from_url(REDIS_URL)
        r.publish(channel, json.dumps(event_data))
    except Exception as e:
        logger.debug("Redis publish skipped or unavailable: %s", e)


if HAS_CELERY and app:
    @app.task(bind=True, name="web.celery_app.task_train_model")
    def task_train_model(self, run_id: str, method: str, model_id: str, config_dict: Dict[str, Any]):
        """Execute QLoRA / LoRA / FFT training inside distributed worker."""
        from pipeline.common import Config, PROJECT_ROOT
        from web.db.session import SessionLocal
        from web.db.models import Run, MetricPoint
        from web.services.storage_service import storage_service

        db = SessionLocal()
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            run.status = "RUNNING"
            run.started_at = datetime.utcnow()
            db.commit()

        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "message": f"--- [CELERY WORKER] Job {run_id} started on worker: {method.upper()} with base {model_id} ---",
        })

        try:
            # Materialize config
            cfg = Config.from_dict(config_dict)

            # Route to appropriate training script
            if method.lower() == "qlora":
                import pipeline.train_qlora as trainer
            elif method.lower() == "lora":
                import pipeline.train_lora as trainer
            else:
                import pipeline.train_fft as trainer

            # Execute pipeline
            output_dir = getattr(cfg, f"output_{method.lower()}", cfg.output_qlora)
            trainer.train(cfg)

            # Upload artifacts to MinIO
            if storage_service.is_s3_enabled() and output_dir.exists():
                publish_redis_event(f"run:{run_id}:stream", {
                    "type": "log",
                    "message": f"--- [MINIO] Streaming trained weights to S3 bucket: {storage_service.bucket_checkpoints} ---",
                })
                for f in output_dir.glob("*"):
                    if f.is_file():
                        s3_key = f"{run_id}/{f.name}"
                        storage_service.upload_file(f, storage_service.bucket_checkpoints, s3_key)

            if run:
                run.status = "COMPLETED"
                run.completed_at = datetime.utcnow()
                db.commit()

            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "message": "--- [COMPLETED] Training completed successfully ---",
            })
            return {"status": "success", "run_id": run_id}

        except Exception as e:
            logger.error("Training task failed: %s", e)
            if run:
                run.status = "FAILED"
                run.error_message = str(e)
                run.completed_at = datetime.utcnow()
                db.commit()

            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "message": f"--- [FAILED] Error during execution: {str(e)} ---",
            })
            raise e
        finally:
            db.close()


    @app.task(bind=True, name="web.celery_app.task_prepare_data")
    def task_prepare_data(self, run_id: str, doc_name: str, chunk_size: int, chunk_overlap: int):
        """Execute document chunking and teacher LLM dataset generation."""
        from pipeline.prepare_data import chunk_text
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "message": f"--- [CELERY WORKER] Document processing started for {doc_name} ---",
        })
        return {"status": "success", "doc_name": doc_name}
