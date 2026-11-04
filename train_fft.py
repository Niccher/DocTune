"""
train_fft.py
------------
Full Fine-Tuning (FFT) of an 8B-class model.

- Updates every parameter of the network.
- Requires multi-GPU hardware (typically 4–8 × A100/H100 80GB)
  because of the large optimizer states.
- Highest possible quality, but expensive and can cause
  catastrophic forgetting of the original capabilities.

Run:
    python train_fft.py
"""

import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from common import Config, get_tokenizer, print_trainable_parameters


def main():
    cfg = Config()

    print("=" * 70)
    print("FULL FINE-TUNING (FFT)")
    print("=" * 70)
    print(f"Model        : {cfg.model_id}")
    print(f"Dataset      : {cfg.dataset_path}")
    print(f"Output dir   : {cfg.output_fft}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = get_tokenizer(cfg.model_id)

    # ------------------------------------------------------------------
    # 2. Load the full model in bfloat16
    #    device_map="auto" will spread layers across all visible GPUs
    # ------------------------------------------------------------------
    print("\nLoading base model (this can take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",   # comment out if flash-attn not installed
    )

    # Make every parameter trainable
    for param in model.parameters():
        param.requires_grad = True

    print_trainable_parameters(model)

    # ------------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------------
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.dataset_path}\n"
            "Run prepare_data.py first."
        )

    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")
    print(f"Loaded {len(dataset)} training examples")

    # ------------------------------------------------------------------
    # 4. Training arguments
    #    Memory is the limiting factor → small batch + gradient accumulation
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=str(cfg.output_fft),
        max_seq_length=cfg.max_seq_length,
        per_device_train_batch_size=1,               # keep very small
        gradient_accumulation_steps=16,             # effective batch = 16
        learning_rate=cfg.learning_rate_fft,        # much lower than LoRA
        num_train_epochs=2,                         # usually 1–2 is enough
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,                # essential for memory
        report_to="none",
        seed=cfg.seed,
    )

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("\nStarting Full Fine-Tuning...")
    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nFFT finished in {hours:.2f} hours ({elapsed:.0f} seconds)")

    # ------------------------------------------------------------------
    # 7. Save the complete model (~16 GB for an 8B model)
    # ------------------------------------------------------------------
    cfg.output_fft.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.output_fft))
    tokenizer.save_pretrained(str(cfg.output_fft))

    print(f"Full model saved to: {cfg.output_fft}")
    print("Done.")


if __name__ == "__main__":
    main()
