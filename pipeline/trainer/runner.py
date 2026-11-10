"""
pipeline/trainer/runner.py
--------------------------
Unified runner orchestrating the complete model fine-tuning lifecycle.
Handles pre-flight checks, dataset tokenization preview, training execution,
reproducibility manifest generation, and adapter persistence.
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from pipeline.common import Config, get_tokenizer, get_run_artifacts_dir
from pipeline.hardware import get_hardware_profile
from pipeline.trainer.builder import (
    build_model_and_peft,
    build_dataset_and_eval,
    build_data_collator,
    build_training_arguments,
)
from pipeline.trainer.callbacks import MetricLoggingCallback, HeartbeatAndCancelCallback

logger = logging.getLogger("doctune.trainer.runner")

try:
    import torch
    import transformers
    import peft
    import trl
    from trl import SFTTrainer
    from transformers import EarlyStoppingCallback
    HAS_ML = True
except ImportError:
    torch = None
    transformers = None
    peft = None
    trl = None
    SFTTrainer = None
    EarlyStoppingCallback = None
    HAS_ML = False


def get_git_commit_hash() -> str:
    """Retrieve current repository git commit SHA for auditability."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2.0)
        return res.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_training(
    cfg: Config,
    method: str = "qlora",
    callbacks: Optional[list] = None,
    resume_from_checkpoint: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Path:
    """
    Execute unified training with isolated artifact trees, hardware profiling,
    train/val splitting, and reproducibility manifests.
    """
    if not HAS_ML or SFTTrainer is None:
        raise ImportError("Machine learning libraries (torch, transformers, peft, trl) are required.")

    hw = get_hardware_profile()
    run_id = cfg.run_id or f"run_{method.lower()}_{int(time.time())}"
    run_artifacts_dir = cfg.output_dir or get_run_artifacts_dir(run_id)
    output_dir = run_artifacts_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print(f"DOCTUNE UNIFIED TRAINER — [{method.upper()}]", flush=True)
    print("=" * 72, flush=True)
    print(f"Run ID       : {run_id}", flush=True)
    print(f"Model        : {cfg.model_id}", flush=True)
    print(f"Hardware     : {hw.device_name} ({hw.device.upper()}, {hw.vram_gb} GB VRAM)", flush=True)
    print(f"Precision    : {hw.recommended_dtype} (Optimizer: {hw.recommended_optim})", flush=True)
    print(f"Dataset      : {cfg.dataset_path}", flush=True)
    print(f"Output Path  : {output_dir}", flush=True)
    print(f"LoRA Rank    : r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}", flush=True)
    print(f"Target Mods  : {cfg.target_modules}", flush=True)
    print(f"Loss Masking : {cfg.loss_on} (only on completions)", flush=True)
    if resume_from_checkpoint:
        print(f"Resume State : {resume_from_checkpoint}", flush=True)
    print("=" * 72, flush=True)

    # 1. Tokenizer & Chat Template Verification
    tokenizer = get_tokenizer(cfg.model_id)

    # 2. Dataset and Validation Split (C-03)
    train_ds, eval_ds = build_dataset_and_eval(cfg, tokenizer)
    has_eval = eval_ds is not None

    # Pre-flight: Render sample chat template formatting for preview (C-05)
    try:
        sample_messages = train_ds[0].get("messages")
        if sample_messages:
            rendered = tokenizer.apply_chat_template(sample_messages, tokenize=False)
            print("\n--- [PRE-FLIGHT] Chat Template Formatting Sample ---", flush=True)
            print(rendered[:400] + "..." if len(rendered) > 400 else rendered, flush=True)
            print("-" * 52 + "\n", flush=True)
    except Exception as ex:
        logger.debug("Chat template preview omitted: %s", ex)

    # 3. Model and PEFT (C-01, C-02, C-04)
    model, peft_config = build_model_and_peft(cfg, hw, method=method)

    # 4. Collator & Loss Masking (C-05)
    data_collator = build_data_collator(cfg, tokenizer)

    # 5. Training Arguments (C-03, C-07)
    training_args = build_training_arguments(cfg, hw, has_eval=has_eval, output_dir=output_dir)

    # 6. Callbacks
    cb_list = [
        MetricLoggingCallback(),
        HeartbeatAndCancelCallback(run_id=run_id, job_id=job_id),
    ]

    # Early stopping if eval set is active and patience configured
    patience = getattr(cfg, "early_stopping_patience", 3)
    if has_eval and patience and EarlyStoppingCallback is not None:
        cb_list.append(EarlyStoppingCallback(early_stopping_patience=patience))
        print(f"Early stopping enabled (metric: eval_loss, patience={patience})", flush=True)

    if callbacks:
        cb_list.extend(callbacks)

    # 7. Trainer Instantiation
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=peft_config,
        callbacks=cb_list,
    )

    # 8. Train Execution
    start_time = time.time()
    resume_arg = None
    if resume_from_checkpoint:
        if str(resume_from_checkpoint).lower() in ("true", "auto", "1", "latest"):
            resume_arg = True
        else:
            resume_arg = str(resume_from_checkpoint)

    print("\nExecuting model optimization pass...", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume_arg)
    elapsed_sec = time.time() - start_time
    print(f"\nTraining completed in {elapsed_sec / 60:.1f} minutes ({elapsed_sec:.0f} seconds)", flush=True)

    # 9. Save Weights and Tokenizer to isolated directory
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 10. Reproducibility Manifest (C-09)
    manifest = {
        "run_id": run_id,
        "method": method,
        "model_id": cfg.model_id,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(elapsed_sec, 1),
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds else 0,
        "train_loss": getattr(train_result, "training_loss", None),
        "git_commit": get_git_commit_hash(),
        "libraries": {
            "torch": torch.__version__ if torch else None,
            "transformers": transformers.__version__ if transformers else None,
            "peft": peft.__version__ if peft else None,
            "trl": trl.__version__ if trl else None,
        },
        "hardware": hw.to_dict(),
        "resolved_config": cfg.to_dict(),
    }

    manifest_path = run_artifacts_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Weights saved to: {output_dir}", flush=True)
    print(f"Manifest saved to: {manifest_path}", flush=True)
    return output_dir
