"""
pipeline/train_qlora.py
-----------------------
QLoRA (Quantized LoRA) fine-tuning in 4-bit.
"""

import time
import json
import argparse
from pathlib import Path
from typing import Optional

try:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer, SFTConfig
except ImportError:
    torch = None
    load_dataset = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    TrainerCallback = object
    LoraConfig = None
    prepare_model_for_kbit_training = None
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


def main(config_path: Optional[str] = None):
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = Config.from_dict(json.load(f))
    else:
        cfg = Config()

    if torch is None or SFTTrainer is None:
        raise ImportError("Core ML stack (torch, transformers, peft, trl) is not installed. Run: pip install -r requirements.txt")

    print("=" * 70, flush=True)
    print("QLoRA FINE-TUNING (4-bit)", flush=True)
    print("=" * 70, flush=True)
    print(f"Model        : {cfg.model_id}", flush=True)
    print(f"Dataset      : {cfg.dataset_path}", flush=True)
    print(f"Output dir   : {cfg.output_qlora}", flush=True)
    print(f"LoRA rank    : {cfg.lora_r}", flush=True)
    print(f"LoRA alpha   : {cfg.lora_alpha}", flush=True)
    print(f"Epochs       : {cfg.num_train_epochs}", flush=True)
    print(f"Batch size   : {cfg.per_device_train_batch_size} (grad_accum={cfg.gradient_accumulation_steps})", flush=True)
    print("=" * 70, flush=True)

    if not Path(cfg.dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found at {cfg.dataset_path}. Run prepare_data first.")

    # 1. Tokenizer
    tokenizer = get_tokenizer(cfg.model_id)

    # 2. 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. Load quantized base model
    print("\nLoading 4-bit quantized model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # 5. LoRA configuration
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

    # 6. Load dataset
    print(f"\nLoading dataset from {cfg.dataset_path}...", flush=True)
    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")
    print(f"Dataset loaded: {len(dataset)} examples", flush=True)

    # 7. Training arguments
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
        optim="paged_adamw_8bit",
        report_to="none",
        seed=cfg.seed,
    )

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricLoggingCallback()],
    )

    # 9. Train
    print("\nStarting QLoRA training...", flush=True)
    start_time = time.time()
    trainer.train()

    elapsed = time.time() - start_time
    print(f"\nQLoRA finished in {elapsed / 3600:.2f} hours ({elapsed:.0f} seconds)", flush=True)

    # 10. Save
    cfg.output_qlora.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_qlora))
    tokenizer.save_pretrained(str(cfg.output_qlora))

    print(f"Adapters saved to: {cfg.output_qlora}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning")
    parser.add_argument("--config-json", type=str, default=None, help="Path to JSON configuration file")
    args = parser.parse_args()
    main(config_path=args.config_json)
