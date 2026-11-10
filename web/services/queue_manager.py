"""
web/services/queue_manager.py
-----------------------------
Transactional PostgreSQL-backed job queue and checkpoint discovery service.
Eliminates in-memory single-slot gating and flat JSON queue files.
Supports priority queuing, worker heartbeats, and orphan reconciliation.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from pipeline.common import PROJECT_ROOT, ARTIFACTS_ROOT, Config
from web.core.ids import generate_id
from web.db.session import SessionLocal, HAS_SQLALCHEMY
from web.db.models import Job, Run, Checkpoint, Project


class StorageQuotaExceededException(Exception):
    """Raised when a project exceeds its allocated storage quota in bytes."""
    pass


logger = logging.getLogger("doctune.queue")


def now_utc():
    return datetime.now(timezone.utc)


class QueueManager:
    """Database-backed transactional job queue manager."""

    def enqueue(
        self,
        job_spec: Dict[str, Any],
        project_id: str = "proj_default",
        user_id: Optional[int] = None,
        priority: int = 10,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        # Normalize job_spec if passed as Pydantic model or dict
        if hasattr(job_spec, "model_dump"):
            spec = job_spec.model_dump()
        elif hasattr(job_spec, "dict"):
            spec = job_spec.dict()
        elif isinstance(job_spec, dict):
            spec = dict(job_spec)
        else:
            spec = dict(job_spec)

        job_id = spec.get("job_id") or generate_id("job")
        run_id = spec.get("run_id") or generate_id("run")
        job_type = spec.get("job_type", "TRAINING")
        gpu_req = int(spec.get("gpu_count", 1))

        close_session = False
        if db is None and HAS_SQLALCHEMY and SessionLocal:
            db = SessionLocal()
            close_session = True

        if not db:
            return {"status": "error", "message": "Database service unavailable"}

        try:
            # 1. Enforce storage quota per project (I-05)
            project = db.query(Project).filter(Project.id == project_id).first()
            if project and project.storage_quota_bytes and project.storage_quota_bytes > 0:
                used_bytes = 0
                runs = db.query(Run).filter(Run.project_id == project_id).all()
                for r in runs:
                    r_dir = ARTIFACTS_ROOT / "runs" / r.id
                    if r_dir.exists():
                        try:
                            used_bytes += sum(f.stat().st_size for f in r_dir.glob("**/*") if f.is_file())
                        except Exception:
                            pass
                if used_bytes >= project.storage_quota_bytes:
                    msg = f"Project '{project_id}' exceeded storage quota of {project.storage_quota_bytes} bytes (currently used: {used_bytes} bytes)."
                    logger.warning(msg)
                    raise StorageQuotaExceededException(msg)

            # 2. Check per-project concurrency limit (I-07, default 2)
            max_concurrent = getattr(project, "max_concurrent_jobs", 2) or 2
            active_count = db.query(Job).filter(
                Job.project_id == project_id,
                Job.status.in_(["RUNNING", "ASSIGNED"]),
            ).count()
            if active_count >= max_concurrent:
                logger.info("Project '%s' is at concurrency limit (%d/%d active). Job %s queued.", project_id, active_count, max_concurrent, job_id)

            job = Job(
                id=job_id,
                project_id=project_id,
                owner_id=user_id,
                run_id=run_id,
                type=job_type,
                status="QUEUED",
                priority=priority,
                spec_json=spec,
                gpu_req=gpu_req,
                queued_at=now_utc(),
            )
            db.add(job)

            # Ensure corresponding Run record exists in QUEUED status
            run = db.query(Run).filter(Run.id == run_id).first()
            if not run:
                run = Run(
                    id=run_id,
                    project_id=project_id,
                    created_by=user_id,
                    job_type=job_type,
                    method=spec.get("method", "qlora"),
                    model_id=spec.get("model_id", "meta-llama/Meta-Llama-3-8B-Instruct"),
                    status="QUEUED",
                    dataset_path=spec.get("dataset_path"),
                    dataset_version=spec.get("dataset_version_id"),
                    dataset_hash=spec.get("dataset_sha256"),
                    epochs=spec.get("epochs", 3),
                    batch_size=spec.get("batch_size", 4),
                    learning_rate=spec.get("learning_rate", 2e-4),
                    config_json=spec,
                    created_at=now_utc(),
                )
                db.add(run)
            else:
                run.status = "QUEUED"

            db.commit()
            db.refresh(job)

            # Count pending queue items ahead of this one
            position = db.query(Job).filter(
                Job.project_id == project_id,
                Job.status == "QUEUED",
                Job.queued_at <= job.queued_at,
            ).count()

            logger.info("Enqueued job '%s' (run '%s') at position %d", job_id, run_id, position)
            return {
                "status": "queued",
                "job_id": job_id,
                "run_id": run_id,
                "position": position,
                "job_spec": spec,
            }
        except Exception as ex:
            logger.error("Failed to enqueue job: %s", ex)
            db.rollback()
            return {"status": "error", "message": str(ex)}
        finally:
            if close_session and db:
                db.close()

    def list_queue(self, project_id: Optional[str] = None, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Alias for get_queue for backward compatibility."""
        return self.get_queue(project_id=project_id, db=db)

    def get_queue(self, project_id: Optional[str] = None, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Return list of pending jobs in queue ordered by priority and queue time."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return []

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            query = db.query(Job).filter(Job.status == "QUEUED")
            if project_id:
                query = query.filter(Job.project_id == project_id)
            jobs = query.order_by(Job.priority.desc(), Job.queued_at.asc()).all()

            return [
                {
                    "job_id": j.id,
                    "run_id": j.run_id,
                    "project_id": j.project_id,
                    "type": j.type,
                    "status": j.status,
                    "priority": j.priority,
                    "queued_at": j.queued_at.isoformat() if j.queued_at else None,
                    "spec": j.spec_json or {},
                }
                for j in jobs
            ]
        except Exception as ex:
            logger.error("Failed querying job queue: %s", ex)
            return []
        finally:
            if close_session and db:
                db.close()

    def get_active_jobs(self, project_id: Optional[str] = None, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Return all currently executing jobs across all or specified project."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return []

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            query = db.query(Job).filter(Job.status.in_(["RUNNING", "ASSIGNED"]))
            if project_id:
                query = query.filter(Job.project_id == project_id)
            jobs = query.order_by(Job.started_at.desc()).all()

            return [
                {
                    "job_id": j.id,
                    "run_id": j.run_id,
                    "project_id": j.project_id,
                    "type": j.type,
                    "status": j.status,
                    "worker_id": j.worker_id,
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "heartbeat_at": j.heartbeat_at.isoformat() if j.heartbeat_at else None,
                    "spec": j.spec_json or {},
                }
                for j in jobs
            ]
        except Exception as ex:
            logger.error("Failed querying active jobs: %s", ex)
            return []
        finally:
            if close_session and db:
                db.close()

    def remove(self, run_or_job_id: str, db: Optional[Session] = None) -> bool:
        """Cancel a queued or running job."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            job = db.query(Job).filter(
                (Job.id == run_or_job_id) | (Job.run_id == run_or_job_id)
            ).first()

            if not job:
                return False

            job.status = "CANCELLED"
            job.finished_at = now_utc()

            run = db.query(Run).filter(Run.id == job.run_id).first()
            if run and run.status in ["QUEUED", "RUNNING", "PENDING"]:
                run.status = "ABORTED"
                run.completed_at = now_utc()

            db.commit()

            # Set Redis cancellation signal flag for active worker
            if job.run_id:
                try:
                    import redis
                    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
                    r.set(f"cancel:{job.run_id}", "1", ex=3600)
                except Exception:
                    pass

            return True
        except Exception as ex:
            logger.error("Failed removing/cancelling job '%s': %s", run_or_job_id, ex)
            db.rollback()
            return False
        finally:
            if close_session and db:
                db.close()

    cancel = remove

    def get_job(self, run_or_job_id: str, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """Retrieve job record by run_id or job_id."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return None

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            job = db.query(Job).filter(
                (Job.id == run_or_job_id) | (Job.run_id == run_or_job_id) | (Job.celery_task_id == run_or_job_id)
            ).first()
            if not job:
                return None
            return {
                "job_id": job.id,
                "run_id": job.run_id,
                "project_id": job.project_id,
                "type": job.type,
                "status": job.status,
                "state": job.status,
                "priority": job.priority,
                "spec": job.spec_json,
                "queued_at": job.queued_at.isoformat() if job.queued_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "error_code": job.error_code,
                "error_message": job.error_message,
            }
        finally:
            if close_session and db:
                db.close()

    def get_status(self, run_or_job_id: str, db: Optional[Session] = None) -> Dict[str, Any]:
        """Return simplified status dictionary for a job."""
        j = self.get_job(run_or_job_id, db=db)
        if j:
            return {"status": j["status"], "state": j["status"], "run_id": j["run_id"], "job_id": j["job_id"]}
        return {"status": "NOT_FOUND", "state": "NOT_FOUND"}

    def mark_completed(
        self,
        run_or_job_id: str,
        results: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None,
    ) -> bool:
        """Mark job and corresponding run as COMPLETED with optional result summary."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            job = db.query(Job).filter((Job.id == run_or_job_id) | (Job.run_id == run_or_job_id)).first()
            if job:
                job.status = "COMPLETED"
                job.finished_at = now_utc()
            run = db.query(Run).filter((Run.id == run_or_job_id) | (Run.id == (job.run_id if job else ""))).first()
            if run:
                run.status = "COMPLETED"
                if results:
                    if "final_loss" in results:
                        run.final_loss = float(results["final_loss"])
                    if "duration_sec" in results:
                        run.duration_sec = float(results["duration_sec"])
            db.commit()
            return True
        except Exception as ex:
            logger.error("Failed marking job completed: %s", ex)
            db.rollback()
            return False
        finally:
            if close_session and db:
                db.close()

    def mark_failed(
        self,
        run_or_job_id: str,
        error_message: str,
        error_diagnostic: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None,
    ) -> bool:
        """Mark job and corresponding run as FAILED with error diagnostic."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            job = db.query(Job).filter((Job.id == run_or_job_id) | (Job.run_id == run_or_job_id)).first()
            if job:
                job.status = "FAILED"
                job.finished_at = now_utc()
                job.error_message = error_message
                if error_diagnostic:
                    job.error_code = error_diagnostic.get("code")
                    job.error_details = error_diagnostic
            run = db.query(Run).filter((Run.id == run_or_job_id) | (Run.id == (job.run_id if job else ""))).first()
            if run:
                run.status = "FAILED"
            db.commit()
            return True
        except Exception as ex:
            logger.error("Failed marking job failed: %s", ex)
            db.rollback()
            return False
        finally:
            if close_session and db:
                db.close()

    def update_heartbeat(self, job_id: str, worker_id: Optional[str] = None, db: Optional[Session] = None) -> bool:
        """Update worker heartbeat timestamp for an active job."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.heartbeat_at = now_utc()
                if worker_id:
                    job.worker_id = worker_id
                db.commit()
                return True
            return False
        except Exception as ex:
            logger.debug("Failed updating heartbeat: %s", ex)
            db.rollback()
            return False
        finally:
            if close_session and db:
                db.close()

    def reconcile_orphan_jobs(self, timeout_sec: int = 300, db: Optional[Session] = None) -> int:
        """
        Reconcile abandoned jobs where worker died (OOM or crash).
        Marks jobs with no heartbeat for > timeout_sec as FAILED (lost).
        """
        if not HAS_SQLALCHEMY or not SessionLocal:
            return 0

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            if timeout_sec <= 0:
                stale_jobs = db.query(Job).filter(
                    Job.status.in_(["RUNNING", "ASSIGNED"]),
                ).all()
            else:
                threshold = now_utc() - timedelta(seconds=timeout_sec)
                stale_jobs = db.query(Job).filter(
                    Job.status.in_(["RUNNING", "ASSIGNED"]),
                    (Job.heartbeat_at < threshold) | (Job.heartbeat_at.is_(None) & (Job.started_at < threshold)),
                ).all()

            count = len(stale_jobs)
            for j in stale_jobs:
                j.status = "FAILED"
                j.error_code = "WORKER_HEARTBEAT_TIMEOUT"
                j.error_message = f"Worker heartbeat lapsed over {timeout_sec}s. Presumed terminated or OOM killed."
                j.finished_at = now_utc()

                run = db.query(Run).filter(Run.id == j.run_id).first()
                if run and run.status == "RUNNING":
                    run.status = "FAILED"
                    run.error_message = j.error_message
                    run.completed_at = now_utc()

            if count > 0:
                db.commit()
                logger.warning("Reconciled %d orphaned / abandoned jobs to FAILED.", count)
            return count
        except Exception as ex:
            logger.error("Error reconciling orphan jobs: %s", ex)
            db.rollback()
            return 0
        finally:
            if close_session and db:
                db.close()

    def discover_resumable_checkpoints(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scan database checkpoints and isolated artifact paths for checkpoints
        that can be resumed.
        """
        results: List[Dict[str, Any]] = []

        # 1. Query registered checkpoints from DB
        if HAS_SQLALCHEMY and SessionLocal:
            try:
                db = SessionLocal()
                query = db.query(Checkpoint)
                if project_id:
                    query = query.filter(Checkpoint.project_id == project_id)
                ckpts = query.order_by(Checkpoint.created_at.desc()).limit(50).all()
                for c in ckpts:
                    results.append({
                        "name": c.name,
                        "method": c.adapter_type or "qlora",
                        "path": c.disk_path,
                        "step": c.step,
                        "run_id": c.run_id,
                        "val_loss": c.val_loss,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                    })
                db.close()
            except Exception:
                pass

        # 2. Also scan ARTIFACTS_ROOT/runs/*/checkpoints/ for intermediate checkpoints
        runs_dir = ARTIFACTS_ROOT / "runs"
        if runs_dir.exists():
            for run_path in runs_dir.iterdir():
                if not run_path.is_dir():
                    continue
                ckpt_dir = run_path / "checkpoints"
                if not ckpt_dir.exists():
                    continue
                for step_dir in sorted(ckpt_dir.glob("checkpoint-*")):
                    if not step_dir.is_dir():
                        continue
                    step = None
                    try:
                        step = int(step_dir.name.split("-")[-1])
                    except Exception:
                        pass
                    if not any(r.get("path") == str(step_dir) for r in results):
                        results.append({
                            "name": f"{run_path.name} ({step_dir.name})",
                            "method": "qlora",
                            "path": str(step_dir),
                            "step": step,
                            "run_id": run_path.name,
                            "modified": step_dir.stat().st_mtime,
                        })

        return results


queue_manager = QueueManager()
