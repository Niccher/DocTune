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


def is_celery_available() -> bool:
    """Check if Celery app is initialized and Redis broker is reachable."""
    if not HAS_CELERY or not app:
        return False
    try:
        import redis
        r = redis.Redis.from_url(REDIS_URL, socket_timeout=1.0)
        return bool(r.ping())
    except Exception:
        return False


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

        start_msg = f"--- [CELERY WORKER] Job {run_id} started on worker: {method.upper()} with base {model_id} ---"
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "line": start_msg,
            "message": start_msg,
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

            # Define callback to publish metrics and write to DB
            cb_list = []
            try:
                from transformers import TrainerCallback

                class CeleryMetricCallback(TrainerCallback):
                    def on_log(self, args, state_hf, control, logs=None, **kwargs):
                        if logs and "loss" in logs:
                            try:
                                step = int(state_hf.global_step)
                                loss = float(logs["loss"])
                                epoch = float(state_hf.epoch or 0.0)
                                publish_redis_event(f"run:{run_id}:stream", {
                                    "type": "metric",
                                    "data": {"step": step, "loss": round(loss, 4), "epoch": round(epoch, 4)},
                                })
                                mp = MetricPoint(run_id=run_id, step=step, epoch=epoch, loss=loss)
                                db.add(mp)
                                db.commit()
                            except Exception as ex:
                                logger.debug("Failed saving metric point: %s", ex)

                cb_list.append(CeleryMetricCallback())
            except Exception:
                pass

            # Execute pipeline
            output_dir = getattr(cfg, f"output_{method.lower()}", cfg.output_qlora)
            trainer.train(cfg, callbacks=cb_list)

            # Upload artifacts to MinIO
            if storage_service.is_s3_enabled() and output_dir.exists():
                upload_msg = f"--- [MINIO] Streaming trained weights to S3 bucket: {storage_service.bucket_checkpoints} ---"
                publish_redis_event(f"run:{run_id}:stream", {
                    "type": "log",
                    "line": upload_msg,
                    "message": upload_msg,
                })
                for f in output_dir.glob("*"):
                    if f.is_file():
                        s3_key = f"{run_id}/{f.name}"
                        storage_service.upload_file(f, storage_service.bucket_checkpoints, s3_key)

            if run:
                run.status = "COMPLETED"
                run.completed_at = datetime.utcnow()
                db.commit()

            done_msg = "--- [COMPLETED] Training completed successfully ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "line": done_msg,
                "message": done_msg,
            })
            return {"status": "success", "run_id": run_id}

        except Exception as e:
            logger.error("Training task failed: %s", e)
            if run:
                run.status = "FAILED"
                run.error_message = str(e)
                run.completed_at = datetime.utcnow()
                db.commit()

            err_msg = f"--- [FAILED] Error during execution: {str(e)} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "line": err_msg,
                "message": err_msg,
            })
            raise e
        finally:
            db.close()


    @app.task(bind=True, name="web.celery_app.task_prepare_data")
    def task_prepare_data(
        self,
        run_id: str,
        teacher_url: str = "",
        teacher_model: str = "",
        teacher_api_key: str = "",
        max_chunks: int = 3,
        chunk_size: int = 1000,
        overlap: int = 150,
        workers: int = 12,
    ):
        """Execute document chunking and teacher LLM dataset generation inside distributed worker."""
        from pipeline.prepare_data import run_preparation, RAW_DOCS_DIR, DEFAULT_JSONL
        start_msg = f"--- [CELERY WORKER] Document processing started for job {run_id} ---"
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "line": start_msg,
            "message": start_msg,
        })
        try:
            total_pairs = run_preparation(
                docs_dir=RAW_DOCS_DIR,
                output_file=DEFAULT_JSONL,
                teacher_url=teacher_url,
                teacher_model=teacher_model,
                teacher_api_key=teacher_api_key,
                max_chunks=max_chunks,
                chunk_size=chunk_size,
                overlap=overlap,
                max_workers=workers,
            )
            done_msg = f"--- [COMPLETED] Wrote {total_pairs} instruction pairs ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "line": done_msg,
                "message": done_msg,
            })
            return {"status": "success", "run_id": run_id, "total_pairs": total_pairs}
        except Exception as e:
            logger.error("Data prep task failed: %s", e)
            err_msg = f"--- [FAILED] Data preparation error: {str(e)} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "line": err_msg,
                "message": err_msg,
            })
            raise e


    @app.task(bind=True, name="web.celery_app.task_merge_adapters")
    def task_merge_adapters(self, run_id: str, adapter_dir: str, export_dir: str, model_id: str):
        """Execute adapter merge into base model inside distributed worker."""
        from pipeline.merge_adapters import run_merge
        from pathlib import Path
        start_msg = f"--- [CELERY WORKER] Merging adapters from {adapter_dir} to {export_dir} ---"
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "line": start_msg,
            "message": start_msg,
        })
        try:
            run_merge(Path(adapter_dir), Path(export_dir), model_id)
            done_msg = f"--- [COMPLETED] Adapters merged successfully to {export_dir} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "line": done_msg,
                "message": done_msg,
            })
            return {"status": "success", "run_id": run_id, "export_dir": export_dir}
        except Exception as e:
            logger.error("Merge task failed: %s", e)
            err_msg = f"--- [FAILED] Adapter merge error: {str(e)} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "line": err_msg,
                "message": err_msg,
            })
            raise e
