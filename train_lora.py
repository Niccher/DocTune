"""
train_lora.py
-------------
LoRA (Low-Rank Adaptation) fine-tuning.

- Freezes the original model weights.
- Trains only small low-rank adapter matrices (~0.5 % of parameters).
- Fits on a single high-end GPU (A100 40/80GB or similar).
- Produces a tiny adapter file (~150-200 MB) that can later be merged.

Run:
    python train_lora.py
"""

import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from common import Config, get_tokenizer, print_trainable_parameters


def main():
    cfg = Config()

    print("=" * 70)
    print("LoRA FINE-TUNING")
    print("=" * 70)
    print(f"Model        : {cfg.model_id}")
    print(f"Dataset      : {cfg.dataset_path}")
    print(f"Output dir   : {cfg.output_lora}")
    print(f"LoRA rank    : {cfg.lora_r}")
    print(f"LoRA alpha   : {cfg.lora_alpha}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = get_tokenizer(cfg.model_id)

    # ------------------------------------------------------------------
    # 2. Load base model in bfloat16 (will be frozen by PEFT)
    # ------------------------------------------------------------------
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",   # comment out if not installed
    )

    # ------------------------------------------------------------------
    # 3. Configure and inject LoRA adapters
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Dataset
    # ------------------------------------------------------------------
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.dataset_path}\n"
            "Run prepare_data.py first."
        )

    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")
    print(f"Loaded {len(dataset)} training examples")

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
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
        optim="adamw_torch",
        report_to="none",
        seed=cfg.seed,
    )

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("\nStarting LoRA training...")
    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nLoRA finished in {hours:.2f} hours ({elapsed:.0f} seconds)")

    # ------------------------------------------------------------------
    # 8. Save only the adapter weights (tiny file)
    # ------------------------------------------------------------------
    cfg.output_lora.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_lora))
    tokenizer.save_pretrained(str(cfg.output_lora))

    print(f"Adapters saved to: {cfg.output_lora}")
    print("Done.")


if __name__ == "__main__":
    main()
