"""
pipeline/train_fft.py
---------------------
Full Fine-Tuning (FFT) of an 8B-class model.
"""

import time
import json
import argparse
from pathlib import Path
from typing import Optional

try:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import SFTTrainer, SFTConfig
except ImportError:
    torch = None
    load_dataset = None
    AutoModelForCausalLM = None
    TrainerCallback = object
    SFTTrainer = None
    SFTConfig = None

try:
    from pipeline.common import Config, get_tokenizer, print_trainable_parameters
except ImportError:
    from common import Config, get_tokenizer, print_trainable_parameters


class MetricLoggingCallback(TrainerCallback):
    """Logs metrics in structured JSON format for web dashboard consumption."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            payload = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), **logs}
            print(f"[METRIC_JSON] {json.dumps(payload)}", flush=True)


def train(cfg: Config, callbacks: Optional[list] = None, resume_from_checkpoint: Optional[str] = None) -> Path:
    """Execute Full Fine-Tuning using unified runner."""
    from pipeline.trainer.runner import run_training
    return run_training(
        cfg=cfg,
        method="fft",
        callbacks=callbacks,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    if not Path(cfg.dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found at {cfg.dataset_path}. Run prepare_data first.")

    tokenizer = get_tokenizer(cfg.model_id)

    print("\nLoading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    for param in model.parameters():
        param.requires_grad = True

    print_trainable_parameters(model)

    print(f"\nLoading dataset from {cfg.dataset_path}...", flush=True)
    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")

    training_args = SFTConfig(
        output_dir=str(cfg.output_fft),
        max_seq_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate_fft,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        seed=cfg.seed,
    )

    cb_list = [MetricLoggingCallback()]
    if callbacks:
        cb_list.extend(callbacks)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=cb_list,
    )

    print("\nStarting FFT training...", flush=True)
    start_time = time.time()

    resume_arg = None
    if resume_from_checkpoint:
        if str(resume_from_checkpoint).lower() in ("true", "auto", "1", "latest"):
            resume_arg = True
        else:
            resume_arg = str(resume_from_checkpoint)
        print(f"Resuming training state from checkpoint: {resume_arg}", flush=True)

    trainer.train(resume_from_checkpoint=resume_arg)

    elapsed = time.time() - start_time
    print(f"\nFFT finished in {elapsed / 3600:.2f} hours ({elapsed:.0f} seconds)", flush=True)

    cfg.output_fft.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_fft))
    tokenizer.save_pretrained(str(cfg.output_fft))

    print(f"Model saved to: {cfg.output_fft}", flush=True)
    print("Done.", flush=True)
    return cfg.output_fft


def main(config_path: Optional[str] = None, resume_checkpoint: Optional[str] = None):
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Config.from_dict(json.load(f))
    else:
        cfg = Config()
    train(cfg, resume_from_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Fine-Tuning")
    parser.add_argument("--config-json", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint directory or 'auto'")
    args = parser.parse_args()
    main(config_path=args.config_json, resume_checkpoint=args.resume_from_checkpoint)
