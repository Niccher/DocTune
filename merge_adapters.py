"""
merge_adapters.py
-----------------
Merge trained LoRA / QLoRA adapters back into the base model
so you obtain a single, normal Hugging Face model with zero
extra inference latency.

After merging you can serve the model with vLLM, TGI, Ollama,
llama.cpp, etc. exactly like any other model.

Run:
    python merge_adapters.py
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common import Config, get_tokenizer


def main():
    cfg = Config()

    # Which adapters to merge?
    ADAPTER_DIR = cfg.output_qlora          # change to cfg.output_lora if needed
    EXPORT_DIR = cfg.output_merged

    print("=" * 70)
    print("MERGING LoRA / QLoRA ADAPTERS INTO BASE MODEL")
    print("=" * 70)
    print(f"Base model   : {cfg.model_id}")
    print(f"Adapters     : {ADAPTER_DIR}")
    print(f"Export dir   : {EXPORT_DIR}")
    print("=" * 70)

    if not (Path(ADAPTER_DIR) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {ADAPTER_DIR}. "
            "Did you run train_lora.py or train_qlora.py first?"
        )

    # ------------------------------------------------------------------
    # 1. Load base model in full precision (important for clean merge)
    #    Using CPU is safer and avoids running out of VRAM.
    # ------------------------------------------------------------------
    print("\nLoading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # 2. Attach the trained adapters
    # ------------------------------------------------------------------
    print("Loading adapters...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    # ------------------------------------------------------------------
    # 3. Fold the low-rank updates into the original weight matrices
    #    W ← W₀ + (α / r) · B · A
    # ------------------------------------------------------------------
    print("Merging weights (this may take a minute)...")
    merged_model = peft_model.merge_and_unload()

    # ------------------------------------------------------------------
    # 4. Save a normal standalone model
    # ------------------------------------------------------------------
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {EXPORT_DIR} ...")
    merged_model.save_pretrained(str(EXPORT_DIR))

    tokenizer = get_tokenizer(cfg.model_id)
    tokenizer.save_pretrained(str(EXPORT_DIR))

    print("\nMerge complete!")
    print(f"You can now load the model from: {EXPORT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
