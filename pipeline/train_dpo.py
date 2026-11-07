"""
pipeline/train_dpo.py
---------------------
Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO)
fine-tuning from Playground human evaluation ratings (thumbs-up / thumbs-down).
"""

import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, TrainerCallback
    from peft import LoraConfig, get_peft_model
    from trl import DPOTrainer, DPOConfig
except ImportError:
    torch = None
    load_dataset = None
    AutoModelForCausalLM = None
    TrainerCallback = object
    LoraConfig = None
    get_peft_model = None
    DPOTrainer = None
    DPOConfig = None

try:
    from pipeline.common import Config, get_tokenizer, print_trainable_parameters, PROJECT_ROOT
except ImportError:
    from common import Config, get_tokenizer, print_trainable_parameters, PROJECT_ROOT

DEFAULT_PREF_DATASET = PROJECT_ROOT / "data" / "preference_pairs.jsonl"
DEFAULT_DPO_OUTPUT = PROJECT_ROOT / "output_dpo"


class DPOMetricCallback(TrainerCallback):
    """Streams DPO loss, reward margins, and accuracy to the web dashboard."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            payload = {
                "step": state.global_step,
                "epoch": round(state.epoch or 0, 4),
                "loss": round(logs.get("loss", 0.0), 4),
                "rewards_chosen": round(logs.get("rewards/chosen", 0.0), 4) if "rewards/chosen" in logs else None,
                "rewards_rejected": round(logs.get("rewards/rejected", 0.0), 4) if "rewards/rejected" in logs else None,
                "rewards_margin": round(logs.get("rewards/margins", 0.0), 4) if "rewards/margins" in logs else None,
                "rewards_accuracies": round(logs.get("rewards/accuracies", 0.0), 4) if "rewards/accuracies" in logs else None,
            }
            print(f"[METRIC_JSON] {json.dumps(payload)}", flush=True)


def train_dpo(
    cfg: Optional[Config] = None,
    dataset_path: Path = DEFAULT_PREF_DATASET,
    output_dir: Path = DEFAULT_DPO_OUTPUT,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 3,
    callbacks: Optional[list] = None,
) -> Path:
    config = cfg or Config.load()
    output_dir = Path(output_dir)

    print("=" * 70, flush=True)
    print("DIRECT PREFERENCE OPTIMIZATION (DPO) FINE-TUNING", flush=True)
    print("=" * 70, flush=True)
    print(f"Base Model    : {config.model_id}", flush=True)
    print(f"Pref Dataset  : {dataset_path}", flush=True)
    print(f"Output Dir    : {output_dir}", flush=True)
    print(f"DPO Beta (β)  : {beta}", flush=True)
    print(f"Learning Rate : {learning_rate}", flush=True)
    print(f"Epochs        : {num_train_epochs}", flush=True)
    print("=" * 70, flush=True)

    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Preference dataset not found at: {dataset_path}\n"
            "Submit ratings on the Playground page, then click 'Export Preference Dataset'."
        )

    if torch is None or DPOTrainer is None:
        raise ImportError("Core ML stack (torch, transformers, peft, trl) is not installed. Run: pip install -r requirements.txt")

    # 1. Tokenizer
    tokenizer = get_tokenizer(config.model_id)

    # 2. Base Model
    print("\nLoading base model in bfloat16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. LoRA Configuration for DPO Policy Model
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Dataset
    print(f"\nLoading preference dataset from {dataset_path}...", flush=True)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"Preference dataset loaded: {len(dataset)} pairs", flush=True)

    # 5. DPO Training Arguments
    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=max(1, config.per_device_train_batch_size // 2),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        seed=config.seed,
        max_prompt_length=512,
        max_length=config.max_seq_length,
    )

    # 6. Callbacks
    cb_list = [DPOMetricCallback()]
    if callbacks:
        cb_list.extend(callbacks)

    # 7. DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT handles reference model automatically with zero extra VRAM
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=cb_list,
    )

    # 8. Train
    print("\nStarting DPO preference alignment...", flush=True)
    start_time = time.time()
    trainer.train()

    elapsed = time.time() - start_time
    print(f"\nDPO finished in {elapsed / 3600:.2f} hours ({elapsed:.0f} seconds)", flush=True)

    # 9. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Preference-aligned adapters saved to: {output_dir}", flush=True)
    print("Done.", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) Fine-Tuning")
    parser.add_argument("--config-json", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_PREF_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DPO_OUTPUT)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    cfg = Config.load()
    if args.config_json and Path(args.config_json).exists():
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = Config.from_dict(json.load(f))

    train_dpo(
        cfg=cfg,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
