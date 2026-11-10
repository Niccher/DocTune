"""
web/celery_app.py
-----------------
Distributed Celery application definition and task registry for LLM training,
dataset chunking, adapter merging, and Redis Pub/Sub metric streaming.
"""

import os
import json
import logging
from datetime import datetime, timezone
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
            "web.celery_app.task_export_gguf": {"queue": "training_queue"},
            "web.celery_app.task_train_dpo": {"queue": "training_queue"},
            "web.celery_app.task_prepare_data": {"queue": "data_prep_queue"},
            "web.celery_app.task_download_model": {"queue": "data_prep_queue"},
            "web.celery_app.task_reconcile_orphan_jobs": {"queue": "training_queue"},
        },
        beat_schedule={
            "reconcile-orphan-jobs-every-60s": {
                "task": "web.celery_app.task_reconcile_orphan_jobs",
                "schedule": 60.0,
            },
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
    @app.task(name="web.celery_app.task_reconcile_orphan_jobs")
    def task_reconcile_orphan_jobs():
        """Periodic reaper task marking jobs without heartbeats as FAILED (lost)."""
        from web.services.queue_manager import queue_manager
        count = queue_manager.reconcile_orphan_jobs(timeout_sec=300)
        return {"reconciled": count}

    @app.task(bind=True, name="web.celery_app.task_train_model")
    def task_train_model(self, run_id_or_spec: Any, method: Optional[str] = None, model_id: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """Execute QLoRA / LoRA / FFT training inside distributed worker with full artifact and run isolation."""
        import time
        import hashlib
        from pipeline.common import Config, PROJECT_ROOT, get_run_artifacts_dir, ARTIFACTS_ROOT
        from web.db.session import SessionLocal
        from web.db.models import Run, Job, MetricPoint, Checkpoint, now_utc
        from web.services.storage_service import storage_service
        from web.services.queue_manager import queue_manager

        # Normalize task arguments (supports single JobSpec dict or positional args)
        if isinstance(run_id_or_spec, dict):
            spec = run_id_or_spec
            run_id = spec.get("run_id")
            method = spec.get("method", "qlora")
            model_id = spec.get("model_id", "meta-llama/Meta-Llama-3-8B-Instruct")
            cfg_dict = spec.get("config", {})
            resume_ckpt = spec.get("resume_checkpoint")
            dataset_version_id = spec.get("dataset_version_id")
            expected_dataset_sha256 = spec.get("dataset_sha256")
            job_id = spec.get("job_id")
            project_id = spec.get("project_id", "proj_default")
        else:
            run_id = str(run_id_or_spec)
            method = method or "qlora"
            model_id = model_id or "meta-llama/Meta-Llama-3-8B-Instruct"
            cfg_dict = config_dict or {}
            resume_ckpt = cfg_dict.get("resume_checkpoint")
            dataset_version_id = cfg_dict.get("dataset_version_id")
            expected_dataset_sha256 = cfg_dict.get("dataset_sha256")
            job_id = None
            project_id = cfg_dict.get("project_id", "proj_default")

        worker_hostname = self.request.hostname or "celery-worker"
        db = SessionLocal()

        # Update Run and Job records to RUNNING
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            run.status = "RUNNING"
            run.started_at = now_utc()

        job = None
        if job_id:
            job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            job = db.query(Job).filter(Job.run_id == run_id).first()

        if job:
            job.status = "RUNNING"
            job.worker_id = worker_hostname
            job.celery_task_id = self.request.id
            job.started_at = now_utc()
            job.heartbeat_at = now_utc()

        db.commit()

        start_msg = f"--- [CELERY WORKER] Job {run_id} started on worker: {method.upper()} with base {model_id} (Host: {worker_hostname}) ---"
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "line": start_msg,
            "message": start_msg,
        })

        try:
            # 1. Prepare isolated per-run workspace directory (B-01)
            run_artifacts_dir = get_run_artifacts_dir(run_id)
            output_dir = run_artifacts_dir / "adapter"

            # Materialize config with isolated paths
            cfg = Config.from_dict(cfg_dict)
            cfg.run_id = run_id
            cfg.output_dir = run_artifacts_dir

            # 2. Resolve dataset version and verify SHA-256 integrity (B-02)
            ds_path = Path(cfg.dataset_path)
            if dataset_version_id or not ds_path.exists():
                s3_key = f"datasets/{dataset_version_id}.jsonl" if dataset_version_id else "private_docs.jsonl"
                local_target = ds_path if ds_path.parent.exists() else (run_artifacts_dir / "dataset.jsonl")
                sync_msg = f"--- [STORAGE] Pulling dataset version from S3: {storage_service.bucket_datasets}/{s3_key} ---"
                publish_redis_event(f"run:{run_id}:stream", {"type": "log", "line": sync_msg, "message": sync_msg})
                storage_service.download_file(storage_service.bucket_datasets, s3_key, local_target)
                cfg.dataset_path = local_target
                ds_path = local_target

            # Verify SHA-256 hash if specified
            if ds_path.exists() and expected_dataset_sha256:
                h = hashlib.sha256()
                with open(ds_path, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                actual_sha = h.hexdigest()
                if actual_sha != expected_dataset_sha256:
                    raise ValueError(
                        f"Dataset integrity failure for {ds_path.name}: "
                        f"expected SHA-256 {expected_dataset_sha256}, got {actual_sha}"
                    )
                publish_redis_event(f"run:{run_id}:stream", {
                    "type": "log",
                    "line": f"--- [SECURITY] Dataset SHA-256 verified: {actual_sha[:16]}... ---",
                })

            # 3. Route to unified training module (C-03)
            from pipeline.trainer.runner import run_training

            # 4. Celery Metric & Heartbeat & Cancellation Callback (B-08)
            cb_list = []
            try:
                from transformers import TrainerCallback

                class CeleryRunCallback(TrainerCallback):
                    def __init__(self, run_id_cb: str, job_id_cb: Optional[str]):
                        self.run_id = run_id_cb
                        self.job_id = job_id_cb
                        self.last_heartbeat = time.time()

                    def on_log(self, args, state_hf, control, logs=None, **kwargs):
                        if logs and "loss" in logs:
                            try:
                                step = int(state_hf.global_step)
                                loss = float(logs["loss"])
                                epoch = float(state_hf.epoch or 0.0)
                                eval_loss = float(logs["eval_loss"]) if "eval_loss" in logs else None

                                publish_redis_event(f"run:{self.run_id}:stream", {
                                    "type": "metric",
                                    "data": {
                                        "step": step,
                                        "loss": round(loss, 4),
                                        "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
                                        "epoch": round(epoch, 4),
                                    },
                                })
                                mp = MetricPoint(run_id=self.run_id, step=step, epoch=epoch, loss=loss, eval_loss=eval_loss)
                                db.add(mp)
                                db.commit()
                            except Exception as ex:
                                logger.debug("Failed saving metric point: %s", ex)

                    def on_step_end(self, args, state_hf, control, **kwargs):
                        # Periodic worker heartbeat every 30s
                        now = time.time()
                        if now - self.last_heartbeat >= 30:
                            self.last_heartbeat = now
                            if self.job_id:
                                queue_manager.update_heartbeat(self.job_id, worker_id=worker_hostname)

                        # Check cooperative cancellation signal from Redis (B-08)
                        try:
                            import redis
                            r = redis.Redis.from_url(REDIS_URL, socket_timeout=0.5)
                            if r.get(f"cancel:{self.run_id}"):
                                logger.warning("Cancellation signal received for run '%s'. Halting training.", self.run_id)
                                control.should_training_stop = True
                                control.should_save = True
                        except Exception:
                            pass

                cb_list.append(CeleryRunCallback(run_id, job.id if job else None))
            except Exception as cb_err:
                logger.debug("Failed attaching Celery callback: %s", cb_err)

            # 5. Execute unified training
            run_training(
                cfg=cfg,
                method=method,
                callbacks=cb_list,
                resume_from_checkpoint=resume_from,
                job_id=job.id if job else None,
            )

            # 6. Upload isolated artifacts to S3 under runs/<run_id>/ (B-01)
            if output_dir.exists():
                upload_msg = f"--- [STORAGE] Uploading trained weights to S3: {storage_service.bucket_checkpoints}/runs/{run_id}/adapter ---"
                publish_redis_event(f"run:{run_id}:stream", {"type": "log", "line": upload_msg, "message": upload_msg})
                storage_service.upload_dir(output_dir, storage_service.bucket_checkpoints, s3_prefix=f"runs/{run_id}/adapter")

            # Finalize Run and Job status
            if run:
                run.status = "COMPLETED"
                run.completed_at = now_utc()
                if run.started_at:
                    st = run.started_at
                    ct = run.completed_at
                    if st.tzinfo is None and ct.tzinfo is not None:
                        st = st.replace(tzinfo=timezone.utc)
                    elif st.tzinfo is not None and ct.tzinfo is None:
                        ct = ct.replace(tzinfo=timezone.utc)
                    run.duration_sec = round((ct - st).total_seconds(), 1)

                if output_dir.exists():
                    sz = round(sum(f.stat().st_size for f in output_dir.glob("*") if f.is_file()) / (1024 * 1024), 2)
                    ckpt = Checkpoint(
                        id=f"ckpt_{run_id}",
                        project_id=project_id,
                        run_id=run_id,
                        name=f"{model_id.split('/')[-1]}-{method.upper()}",
                        adapter_type=method.lower(),
                        base_model=model_id,
                        disk_path=str(output_dir),
                        s3_bucket=storage_service.bucket_checkpoints,
                        s3_key=f"runs/{run_id}/adapter",
                        size_mb=sz,
                    )
                    db.add(ckpt)

            if job:
                job.status = "SUCCEEDED"
                job.finished_at = now_utc()

            db.commit()

            # Dispatch success webhook notification (H-05)
            try:
                from web.services.notification_service import notification_service
                import asyncio
                asyncio.run(notification_service.notify_job_status(
                    run_id=run_id,
                    project_id=project_id,
                    job_type=f"TRAINING_{method.upper()}",
                    status="COMPLETED",
                    details={"model_id": model_id, "duration_sec": run.duration_sec if run else None},
                ))
            except Exception as notif_ex:
                logger.debug("Failed sending success notification: %s", notif_ex)

            # Auto-eval follow-up job (D-03)
            auto_eval = True
            if job and job.spec_json:
                auto_eval = job.spec_json.get("auto_eval", True)
            if auto_eval:
                try:
                    queue_manager.enqueue({
                        "job_id": f"eval_{run_id}",
                        "run_id": f"eval_{run_id}",
                        "project_id": project_id,
                        "type": "EVAL",
                        "model_id": model_id,
                        "checkpoint_id": f"ckpt_{run_id}",
                        "auto_eval": True,
                    }, project_id=project_id)
                    logger.info("Auto-eval job enqueued for run %s", run_id)
                except Exception as eval_ex:
                    logger.warning("Failed enqueueing auto_eval: %s", eval_ex)

            done_msg = f"--- [COMPLETED] Training run {run_id} completed successfully in isolated directory: {output_dir} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "line": done_msg,
                "message": done_msg,
            })
            return {"status": "success", "run_id": run_id, "output_dir": str(output_dir)}

        except Exception as e:
            logger.error("Training task failed for run '%s': %s", run_id, e)
            from pipeline.errors import classify_training_error
            diag = classify_training_error(e, {"model_id": model_id})
            err_code = diag.get("code", "EXECUTION_ERROR")
            err_title = diag.get("title", str(e))
            err_hint = diag.get("hint", "")
            err_full = f"[{err_code}] {err_title}: {err_hint}" if err_hint else f"[{err_code}] {err_title}"

            if run:
                run.status = "FAILED"
                run.error_message = err_full
                run.completed_at = now_utc()
            if job:
                job.status = "FAILED"
                job.error_code = err_code
                job.error_details = diag
                job.error_message = err_full
                job.finished_at = now_utc()
            db.commit()

            # Dispatch failure webhook notification (H-05)
            try:
                from web.services.notification_service import notification_service
                import asyncio
                asyncio.run(notification_service.notify_job_status(
                    run_id=run_id,
                    project_id=project_id,
                    job_type=f"TRAINING_{method.upper()}",
                    status="FAILED",
                    details=diag,
                ))
            except Exception as notif_ex:
                logger.debug("Failed sending failure notification: %s", notif_ex)

            err_msg = f"--- [FAILED] {err_full} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "error_code": err_code,
                "error_details": diag,
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
        redact_pii: bool = True,
        pii_mode: str = "placeholder",
    ):
        """Execute document chunking and teacher LLM dataset generation inside distributed worker."""
        from pipeline.prepare_data import run_preparation, RAW_DOCS_DIR, DEFAULT_JSONL
        start_msg = f"--- [CELERY WORKER] Document processing started for job {run_id} (PII Redaction: {redact_pii}) ---"
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
                redact_pii=redact_pii,
                pii_mode=pii_mode,
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
        from datetime import datetime, timezone
        from web.db.session import SessionLocal
        from web.db.models import Run, Checkpoint, now_utc
        from web.services.storage_service import storage_service

        db = SessionLocal() if SessionLocal else None
        run = db.query(Run).filter(Run.id == run_id).first() if db else None
        if run:
            run.status = "RUNNING"
            run.started_at = now_utc()
            db.commit()

        start_msg = f"--- [CELERY WORKER] Merging adapters from {adapter_dir} to {export_dir} ---"
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "line": start_msg,
            "message": start_msg,
        })
        try:
            ad_path = Path(adapter_dir)
            if not (ad_path / "adapter_config.json").exists():
                sync_msg = f"--- [STORAGE] Pulling adapter weights from S3: {storage_service.bucket_checkpoints}/{ad_path.name} ---"
                publish_redis_event(f"run:{run_id}:stream", {
                    "type": "log",
                    "line": sync_msg,
                    "message": sync_msg,
                })
                storage_service.download_dir(storage_service.bucket_checkpoints, ad_path.name, ad_path)

            run_merge(ad_path, Path(export_dir), model_id)
            done_msg = f"--- [COMPLETED] Adapters merged successfully to {export_dir} ---"

            exp_path = Path(export_dir)
            if exp_path.exists():
                upload_msg = f"--- [STORAGE] Uploading merged model to S3 bucket: {storage_service.bucket_merged}/merged_{run_id} ---"
                publish_redis_event(f"run:{run_id}:stream", {
                    "type": "log",
                    "line": upload_msg,
                    "message": upload_msg,
                })
                storage_service.upload_dir(exp_path, storage_service.bucket_merged, s3_prefix=f"merged_{run_id}")

            if db:
                if run:
                    run.status = "COMPLETED"
                    run.completed_at = now_utc()
                    if run.started_at:
                        st = run.started_at
                        ct = run.completed_at
                        if st.tzinfo is None and ct.tzinfo is not None:
                            st = st.replace(tzinfo=timezone.utc)
                        elif st.tzinfo is not None and ct.tzinfo is None:
                            ct = ct.replace(tzinfo=timezone.utc)
                        run.duration_sec = round((ct - st).total_seconds(), 1)

                sz = round(sum(f.stat().st_size for f in exp_path.glob("*") if f.is_file()) / (1024 * 1024), 2) if exp_path.exists() else 0.0
                ckpt = Checkpoint(
                    id=f"ckpt_{run_id}",
                    run_id=run_id,
                    name=f"merged-{exp_path.name}",
                    adapter_type="merged",
                    base_model=model_id,
                    disk_path=str(export_dir),
                    s3_bucket=storage_service.bucket_merged,
                    s3_key=f"merged_{run_id}",
                    size_mb=sz,
                )
                db.add(ckpt)
                db.commit()

            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "line": done_msg,
                "message": done_msg,
            })
            return {"status": "success", "run_id": run_id, "export_dir": export_dir}
        except Exception as e:
            logger.error("Merge task failed: %s", e)
            if db and run:
                run.status = "FAILED"
                run.error_message = str(e)
                run.completed_at = now_utc()
                db.commit()

            err_msg = f"--- [FAILED] Adapter merge error: {str(e)} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "line": err_msg,
                "message": err_msg,
            })
            raise e
        finally:
            if db:
                db.close()


    @app.task(bind=True, name="web.celery_app.task_export_gguf")
    def task_export_gguf(self, run_id: str, model_dir: str, export_dir: str, model_name: str, quantization: str):
        """Execute GGUF model conversion and Ollama Modelfile generation inside distributed worker."""
        from pipeline.export_gguf import run_conversion
        from pathlib import Path
        from web.services.storage_service import storage_service

        start_msg = f"--- [CELERY WORKER] GGUF conversion started for {model_name} ({quantization}) ---"
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "line": start_msg,
            "message": start_msg,
        })
        try:
            m_path = Path(model_dir)
            e_path = Path(export_dir)
            res = run_conversion(m_path, e_path, model_name, quantization)

            # Upload to MinIO merged / checkpoints bucket if storage enabled
            if e_path.exists():
                upload_msg = f"--- [STORAGE] Uploading GGUF package to S3: {storage_service.bucket_merged}/gguf_{run_id} ---"
                publish_redis_event(f"run:{run_id}:stream", {
                    "type": "log",
                    "line": upload_msg,
                    "message": upload_msg,
                })
                storage_service.upload_dir(e_path, storage_service.bucket_merged, s3_prefix=f"gguf_{run_id}")

            done_msg = f"--- [COMPLETED] GGUF model & Modelfile exported to {export_dir} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "line": done_msg,
                "message": done_msg,
            })
            return res
        except Exception as e:
            logger.error("GGUF export failed: %s", e)
            err_msg = f"--- [FAILED] GGUF export error: {str(e)} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "line": err_msg,
                "message": err_msg,
            })
            raise e


    @app.task(bind=True, name="web.celery_app.task_train_dpo")
    def task_train_dpo(self, run_id: str, model_id: str, dataset_path: str, output_dir: str, beta: float, lr: float, epochs: int):
        """Execute DPO preference alignment fine-tuning inside distributed worker."""
        from pipeline.common import Config
        from pipeline.train_dpo import train_dpo
        from pathlib import Path
        from web.db.session import SessionLocal
        from web.db.models import Run, MetricPoint, Checkpoint, now_utc
        from web.services.storage_service import storage_service

        db = SessionLocal() if SessionLocal else None
        run = db.query(Run).filter(Run.id == run_id).first() if db else None
        if run:
            run.status = "RUNNING"
            run.started_at = now_utc()
            db.commit()

        start_msg = f"--- [CELERY WORKER] DPO preference training {run_id} started (Base: {model_id}, Beta: {beta}) ---"
        publish_redis_event(f"run:{run_id}:stream", {
            "type": "log",
            "line": start_msg,
            "message": start_msg,
        })
        try:
            cfg = Config.load()
            cfg.model_id = model_id
            cfg.learning_rate_lora = lr
            cfg.num_train_epochs = epochs

            cb_list = []
            try:
                from transformers import TrainerCallback

                class CeleryDPOCallback(TrainerCallback):
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
                                if db:
                                    mp = MetricPoint(run_id=run_id, step=step, epoch=epoch, loss=loss)
                                    db.add(mp)
                                    db.commit()
                            except Exception:
                                pass
                cb_list.append(CeleryDPOCallback())
            except Exception:
                pass

            out_p = train_dpo(
                cfg=cfg,
                dataset_path=Path(dataset_path),
                output_dir=Path(output_dir),
                beta=beta,
                learning_rate=lr,
                num_train_epochs=epochs,
                callbacks=cb_list,
            )

            # Upload to S3 checkpoints
            if out_p.exists():
                storage_service.upload_dir(out_p, storage_service.bucket_checkpoints, s3_prefix=f"dpo_{run_id}")

            if db and run:
                run.status = "COMPLETED"
                run.completed_at = now_utc()
                db.commit()

            done_msg = f"--- [COMPLETED] DPO preference training finished successfully ({out_p}) ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "COMPLETED",
                "line": done_msg,
                "message": done_msg,
            })
            return {"status": "success", "run_id": run_id, "output_dir": str(out_p)}
        except Exception as e:
            logger.error("DPO training task failed: %s", e)
            if db and run:
                run.status = "FAILED"
                run.error_message = str(e)
                run.completed_at = now_utc()
                db.commit()
            err_msg = f"--- [FAILED] DPO training error: {str(e)} ---"
            publish_redis_event(f"run:{run_id}:stream", {
                "type": "status",
                "status": "FAILED",
                "line": err_msg,
                "message": err_msg,
            })
            raise e
        finally:
            if db:
                db.close()


    @app.task(bind=True, name="web.celery_app.task_download_model")
    def task_download_model(self, model_id: str):
        """Pre-download model weights and tokenizer into the shared Hugging Face cache."""
        import os
        from web.services.config_service import config_service

        settings = config_service.get_settings()
        token = settings.get("huggingface", {}).get("token") or os.getenv("HF_TOKEN")

        start_msg = f"--- [CELERY WORKER] Initiating pre-download of base model: {model_id} ---"
        publish_redis_event(f"download:{model_id}", {
            "type": "log",
            "status": "DOWNLOADING",
            "progress": 20,
            "line": start_msg,
            "message": start_msg,
        })

        try:
            try:
                from huggingface_hub import snapshot_download
                publish_redis_event(f"download:{model_id}", {
                    "type": "log",
                    "status": "DOWNLOADING",
                    "progress": 40,
                    "line": f"Downloading full snapshot of {model_id} via huggingface_hub...",
                    "message": f"Downloading full snapshot of {model_id} via huggingface_hub...",
                })
                snapshot_download(repo_id=model_id, token=token)
            except ImportError:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
                publish_redis_event(f"download:{model_id}", {
                    "type": "log",
                    "status": "DOWNLOADING",
                    "progress": 60,
                    "line": "Tokenizer cached successfully. Downloading model weights...",
                    "message": "Tokenizer cached successfully. Downloading model weights...",
                })
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=token,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                del model
                import gc; gc.collect()

            done_msg = f"--- [COMPLETED] Model {model_id} downloaded and cached on disk ---"
            publish_redis_event(f"download:{model_id}", {
                "type": "status",
                "status": "COMPLETED",
                "progress": 100,
                "line": done_msg,
                "message": done_msg,
            })
            return {"status": "success", "model_id": model_id}
        except Exception as e:
            logger.error("Model download task failed for %s: %s", model_id, e)
            err_msg = f"--- [FAILED] Model download failed: {str(e)} ---"
            publish_redis_event(f"download:{model_id}", {
                "type": "status",
                "status": "FAILED",
                "progress": 0,
                "error": str(e),
                "line": err_msg,
            })
            raise e


