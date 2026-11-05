"""
web/services/process_manager.py
-------------------------------
Asynchronous subprocess supervisor streaming real-time stdout to WebSockets
and parsing metrics for live loss graphing.
"""

import sys
import os
import re
import json
import signal
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

from web.state import state, JobType, JobStatus
from web.services.config_service import config_service
from web.services.system_service import get_hardware_bounds


class ProcessManager:
    def __init__(self):
        self.current_process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None

    async def run_pipeline_script(
        self,
        job_type: JobType,
        script_module: str,
        args: List[str],
        cwd: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Launch a pipeline script asynchronously with live output capture and hardware constraints.
        """
        if state.active_job is not None:
            raise RuntimeError(f"A job is already running: {state.active_job.job_type.value}")

        # Pre-flight hardware & storage safety check
        bounds = get_hardware_bounds()
        if not bounds.get("has_sufficient_disk", True):
            free_d = bounds.get("free_disk_gb", 0)
            req_d = bounds.get("min_required_disk_gb", 15.0)
            raise RuntimeError(
                f"Insufficient disk space to start pipeline: {free_d} GB free, "
                f"minimum {req_d} GB required to safely store checkpoints without disk exhaustion."
            )

        python_bin = sys.executable

        # Command: python -u -m <script_module> <args...>
        cmd = [python_bin, "-u", "-m", script_module] + args


        # Query compute resources configuration
        resources = config_service.get_compute_resources()
        cpu_threads = str(resources.get("cpu_threads", 4))
        gpu_device = str(resources.get("gpu_device", "0")).strip()
        gpu_count = int(resources.get("gpu_count", 1))
        max_ram_gb = resources.get("max_ram_gb", 10)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(cwd)
        env["OMP_NUM_THREADS"] = cpu_threads
        env["MKL_NUM_THREADS"] = cpu_threads
        env["TORCH_NUM_THREADS"] = cpu_threads

        if gpu_device.lower() in ["cpu", "none", "-1"]:
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif gpu_device:
            env["CUDA_VISIBLE_DEVICES"] = gpu_device

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
        )

        self.current_process = process
        job = state.set_job(job_type=job_type, pid=process.pid, metadata=metadata or {})

        await state.add_log(f"--- [START] Job {job.job_id} ({job_type.value}) ---")
        await state.add_log(f"--- [HARDWARE BUDGET] CPU Threads: {cpu_threads} | GPU: {env.get('CUDA_VISIBLE_DEVICES', 'Default')} (Count: {gpu_count}) | RAM Limit: {max_ram_gb} GB ---")
        await state.add_log(f"Command: {' '.join(cmd)}")


        self._reader_task = asyncio.create_task(self._read_stream(process))
        return True

    async def _read_stream(self, process: asyncio.subprocess.Process):
        assert process.stdout is not None
        metric_pattern = re.compile(r"'loss':\s*([0-9\.]+)")
        step_pattern = re.compile(r"'step':\s*([0-9]+)")
        epoch_pattern = re.compile(r"'epoch':\s*([0-9\.]+)")

        try:
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="replace").rstrip()

                # Check for structured metric tag
                # Check for structured metric tag
                parsed_metric = False
                step_val, loss_val, epoch_val = 0, 0.0, 0.0
                if line.startswith("[METRIC_JSON]"):
                    try:
                        metric_data = json.loads(line.replace("[METRIC_JSON]", "").strip())
                        step_val = int(metric_data.get("step", 0))
                        loss_val = float(metric_data.get("loss", 0.0))
                        epoch_val = float(metric_data.get("epoch", 0.0))
                        await state.add_loss_point(step_val, loss_val, epoch_val)
                        parsed_metric = True
                    except Exception as e:
                        print(f"Error parsing [METRIC_JSON]: {e}")
                else:
                    # Regex fallback for standard HuggingFace logging
                    loss_match = metric_pattern.search(line)
                    if loss_match:
                        try:
                            loss_val = float(loss_match.group(1))
                            step_val = int(step_match.group(1)) if (step_match := step_pattern.search(line)) else len(state.loss_history) + 1
                            epoch_val = float(epoch_match.group(1)) if (epoch_match := epoch_pattern.search(line)) else 0.0
                            await state.add_loss_point(step_val, loss_val, epoch_val)
                            parsed_metric = True
                        except Exception:
                            pass

                # Persist MetricPoint to database
                if parsed_metric and state.active_job:
                    run_id = state.active_job.metadata.get("run_id")
                    if run_id:
                        try:
                            from web.db.session import SessionLocal
                            from web.db.models import MetricPoint
                            if SessionLocal:
                                db_m = SessionLocal()
                                db_m.add(MetricPoint(run_id=run_id, step=step_val, epoch=epoch_val, loss=loss_val))
                                db_m.commit()
                                db_m.close()
                        except Exception:
                            pass

                await state.add_log(line)

            returncode = await process.wait()

            # Update Run status and register Checkpoint in PostgreSQL
            run_id = state.active_job.metadata.get("run_id") if state.active_job else None
            if run_id:
                try:
                    from datetime import datetime
                    from web.db.session import SessionLocal
                    from web.db.models import Run, Checkpoint
                    if SessionLocal:
                        db_r = SessionLocal()
                        run_rec = db_r.query(Run).filter(Run.id == run_id).first()
                        if run_rec:
                            run_rec.completed_at = datetime.utcnow()
                            if run_rec.started_at:
                                run_rec.duration_sec = round((run_rec.completed_at - run_rec.started_at).total_seconds(), 1)
                            if state.loss_history:
                                run_rec.final_loss = state.loss_history[-1]["loss"]

                            if returncode == 0:
                                run_rec.status = "COMPLETED"
                                method = state.active_job.metadata.get("method")
                                if method:
                                    out_dir = PROJECT_ROOT / f"output_{method.lower()}"
                                    if out_dir.exists():
                                        sz = round(sum(f.stat().st_size for f in out_dir.glob("*") if f.is_file()) / (1024 * 1024), 2)
                                        db_r.add(Checkpoint(
                                            id=f"ckpt_{run_id}",
                                            run_id=run_id,
                                            name=f"{run_rec.model_id.split('/')[-1]}-{method.upper()}",
                                            adapter_type=method.lower(),
                                            base_model=run_rec.model_id,
                                            disk_path=str(out_dir),
                                            size_mb=sz,
                                        ))
                                elif state.active_job.job_type == JobType.MERGING:
                                    export_dir_str = state.active_job.metadata.get("export_dir", "output_merged")
                                    export_path = Path(export_dir_str)
                                    if not export_path.is_absolute():
                                        export_path = PROJECT_ROOT / export_dir_str
                                    sz = round(sum(f.stat().st_size for f in export_path.glob("*") if f.is_file()) / (1024 * 1024), 2) if export_path.exists() else 0.0
                                    db_r.add(Checkpoint(
                                        id=f"ckpt_{run_id}",
                                        run_id=run_id,
                                        name=f"merged-{export_path.name}",
                                        adapter_type="merged",
                                        base_model=run_rec.model_id if run_rec else "base",
                                        disk_path=str(export_path),
                                        size_mb=sz,
                                    ))

                            elif returncode == -signal.SIGINT or returncode == -signal.SIGTERM:
                                run_rec.status = "ABORTED"
                            else:
                                run_rec.status = "FAILED"
                                run_rec.error_message = f"Process exited with code {returncode}"
                            db_r.commit()
                        db_r.close()
                except Exception:
                    pass

            if returncode == 0:
                await state.add_log("--- [COMPLETED] Process finished successfully ---")
                state.finish_job(JobStatus.COMPLETED)
            elif returncode == -signal.SIGINT or returncode == -signal.SIGTERM:
                await state.add_log("--- [ABORTED] Process terminated by user ---")
                state.finish_job(JobStatus.ABORTED)
            else:
                await state.add_log(f"--- [FAILED] Process exited with code {returncode} ---")
                state.finish_job(JobStatus.FAILED)

        except asyncio.CancelledError:
            await state.add_log("--- [CANCELLED] Reader stream task was cancelled ---")
            state.finish_job(JobStatus.ABORTED)
        finally:
            self.current_process = None

    async def abort_current_job(self) -> bool:
        """Terminate the active running process safely."""
        if not self.current_process or self.current_process.returncode is not None:
            return False

        try:
            self.current_process.send_signal(signal.SIGINT)
            # Give it 3 seconds to shutdown cleanly
            for _ in range(6):
                await asyncio.sleep(0.5)
                if self.current_process.returncode is not None:
                    return True

            # If still alive, terminate firmly
            self.current_process.terminate()
            await asyncio.sleep(1.0)
            if self.current_process.returncode is None:
                self.current_process.kill()
            return True
        except Exception as e:
            await state.add_log(f"Error aborting process: {e}")
            return False


# Singleton manager
process_mgr = ProcessManager()
