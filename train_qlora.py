"""
train_qlora.py
--------------
QLoRA (Quantized LoRA) fine-tuning – the most practical method for
most users.

- Loads the base model in 4-bit (NF4) → dramatically lower VRAM.
- Still trains the same LoRA adapters in higher precision.
- Can fine-tune an 8B model on a single RTX 3090 / 4090 24 GB
  (or even 16 GB cards with careful settings).

Run:
    python train_qlora.py
"""

import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

from common import Config, get_tokenizer, print_trainable_parameters


def main():
    cfg = Config()

    print("=" * 70)
    print("QLoRA FINE-TUNING (4-bit)")
    print("=" * 70)
    print(f"Model        : {cfg.model_id}")
    print(f"Dataset      : {cfg.dataset_path}")
    print(f"Output dir   : {cfg.output_qlora}")
    print(f"LoRA rank    : {cfg.lora_r}")
    print(f"LoRA alpha   : {cfg.lora_alpha}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = get_tokenizer(cfg.model_id)

    # ------------------------------------------------------------------
    # 2. 4-bit quantization configuration
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # Best quality 4-bit format
        bnb_4bit_use_double_quant=True,         # Extra memory saving
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    )

    # ------------------------------------------------------------------
    # 3. Load quantized base model
    # ------------------------------------------------------------------
    print("\nLoading 4-bit quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",   # comment out if not installed
    )

    # Cast layer norms etc. so that training is numerically stable
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # 4. Configure and inject LoRA adapters
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
    # 5. Dataset
    # ------------------------------------------------------------------
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.dataset_path}\n"
            "Run prepare_data.py first."
        )

    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")
    print(f"Loaded {len(dataset)} training examples")

    # ------------------------------------------------------------------
    # 6. Training arguments
    #    paged_adamw_8bit is important – it pages optimizer states
    #    to CPU when VRAM is tight, preventing OOM spikes.
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=str(cfg.output_qlora),
        max_seq_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate_lora,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",               # key for QLoRA stability
        report_to="none",
        seed=cfg.seed,
    )

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    print("\nStarting QLoRA training...")
    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nQLoRA finished in {hours:.2f} hours ({elapsed:.0f} seconds)")

    # ------------------------------------------------------------------
    # 9. Save only the adapter weights
    # ------------------------------------------------------------------
    cfg.output_qlora.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_qlora))
    tokenizer.save_pretrained(str(cfg.output_qlora))

    print(f"Adapters saved to: {cfg.output_qlora}")
    print("Done.")


if __name__ == "__main__":
    main()
