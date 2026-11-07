"""
web/services/sweep_service.py
-----------------------------
Hyperparameter Grid Sweep generator.
Expands parameter matrices into sequential training trials managed by QueueManager.
"""

import time
import json
import itertools
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from pipeline.common import PROJECT_ROOT, Config
from web.state import state, JobType
from web.services.queue_manager import queue_manager

logger = logging.getLogger("doctune.sweeps")
RUNS_DIR = PROJECT_ROOT / ".runs"


class SweepService:
    @staticmethod
    def generate_grid_sweep(
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        method: str = "qlora",
        learning_rates: Optional[List[float]] = None,
        lora_ranks: Optional[List[int]] = None,
        epochs_list: Optional[List[int]] = None,
        dataset_path: Optional[str] = None,
        max_seq_length: int = 1024,
        batch_size: int = 2,
        grad_accum: int = 8,
        max_trials: int = 16,
    ) -> Dict[str, Any]:
        """Generate a combinatorial grid of hyperparameter trials and enqueue them."""
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        lrs = learning_rates or [0.0001, 0.0002]
        ranks = lora_ranks or [16, 32]
        epochs = epochs_list or [3]

        sweep_id = f"sweep_{int(time.time())}"
        combinations = list(itertools.product(lrs, ranks, epochs))

        if len(combinations) > max_trials:
            logger.warning("Sweep combinations (%d) exceeded limit of %d, truncating.", len(combinations), max_trials)
            combinations = combinations[:max_trials]

        queued_trials = []
        for idx, (lr, rank, ep) in enumerate(combinations, start=1):
            trial_id = f"{sweep_id}_trial_{idx:02d}_r{rank}_lr{lr}"
            alpha = rank * 2

            cfg = Config(
                model_id=model_id,
                num_train_epochs=ep,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                lora_r=rank,
                lora_alpha=alpha,
                learning_rate_lora=lr,
                max_seq_length=max_seq_length,
            )
            if dataset_path:
                cfg.dataset_path = Path(dataset_path)

            run_cfg_path = RUNS_DIR / f"config_{trial_id}.json"
            with open(run_cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg.to_dict(), f, indent=2)

            # Record in DB if available
            try:
                from web.db.session import SessionLocal
                from web.db.models import Run
                if SessionLocal:
                    db = SessionLocal()
                    new_run = Run(
                        id=trial_id,
                        job_type=f"TRAINING_{method.upper()}",
                        model_id=model_id,
                        method=method.upper(),
                        dataset_path=str(cfg.dataset_path),
                        epochs=ep,
                        batch_size=batch_size,
                        learning_rate=lr,
                        status="QUEUED",
                        config_json={**cfg.to_dict(), "sweep_id": sweep_id, "trial_index": idx},
                        mode="subprocess",
                    )
                    db.add(new_run)
                    db.commit()
                    db.close()
            except Exception:
                pass

            job_spec = {
                "run_id": trial_id,
                "sweep_id": sweep_id,
                "trial_index": idx,
                "method": method,
                "model_id": model_id,
                "config": cfg.to_dict(),
                "config_file": str(run_cfg_path),
                "mode": "subprocess",
                "resume_checkpoint": None,
                "params": {
                    "learning_rate": lr,
                    "lora_r": rank,
                    "lora_alpha": alpha,
                    "epochs": ep,
                },
            }
            queue_manager.enqueue(job_spec)
            queued_trials.append(job_spec)

        # Trigger execution if system is idle
        if state.active_job is None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(queue_manager.dispatch_next())
                else:
                    loop.run_until_complete(queue_manager.dispatch_next())
            except Exception:
                pass

        return {
            "status": "success",
            "sweep_id": sweep_id,
            "total_trials": len(queued_trials),
            "trials": [
                {
                    "run_id": t["run_id"],
                    "params": t["params"],
                }
                for t in queued_trials
            ],
        }


sweep_service = SweepService()
