"""
pipeline/train_lora.py
----------------------
LoRA (Low-Rank Adaptation) fine-tuning in bfloat16.
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
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
except ImportError:
    torch = None
    load_dataset = None
    AutoModelForCausalLM = None
    TrainerCallback = object
    LoraConfig = None
    get_peft_model = None
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
    """Execute LoRA fine-tuning using unified runner."""
    from pipeline.trainer.runner import run_training
    return run_training(
        cfg=cfg,
        method="lora",
        callbacks=callbacks,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    if not Path(cfg.dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found at {cfg.dataset_path}. Run prepare_data first.")

    # 1. Tokenizer
    tokenizer = get_tokenizer(cfg.model_id)

    # 2. Base Model
    print("\nLoading base model in bfloat16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. LoRA Config
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # 4. Dataset
    print(f"\nLoading dataset from {cfg.dataset_path}...", flush=True)
    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")

    # 5. Training arguments
    training_args = SFTConfig(
        output_dir=str(cfg.output_lora),
        max_seq_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate_lora,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        seed=cfg.seed,
    )

    # 6. Trainer
    cb_list = [MetricLoggingCallback()]
    if callbacks:
        cb_list.extend(callbacks)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=cb_list,
    )

    # 7. Train
    print("\nStarting LoRA training...", flush=True)
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
    print(f"\nLoRA finished in {elapsed / 3600:.2f} hours ({elapsed:.0f} seconds)", flush=True)

    # 8. Save
    cfg.output_lora.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_lora))
    tokenizer.save_pretrained(str(cfg.output_lora))

    print(f"Adapters saved to: {cfg.output_lora}", flush=True)
    print("Done.", flush=True)
    return cfg.output_lora


def main(config_path: Optional[str] = None, resume_checkpoint: Optional[str] = None):
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Config.from_dict(json.load(f))
    else:
        cfg = Config()
    train(cfg, resume_from_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning")
    parser.add_argument("--config-json", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint directory or 'auto'")
    args = parser.parse_args()
    main(config_path=args.config_json, resume_checkpoint=args.resume_from_checkpoint)
