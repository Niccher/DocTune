"""
pipeline/merge_adapters.py
--------------------------
Merge trained LoRA / QLoRA adapters back into the base model.
"""

import argparse
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PeftModel = None

try:
    from pipeline.common import Config, get_tokenizer
except ImportError:
    from common import Config, get_tokenizer


def run_merge(adapter_dir: Path, export_dir: Path, model_id: str):
    print("=" * 70, flush=True)
    print("MERGING LoRA / QLoRA ADAPTERS INTO BASE MODEL", flush=True)
    print("=" * 70, flush=True)
    print(f"Base model   : {model_id}", flush=True)
    print(f"Adapters     : {adapter_dir}", flush=True)
    print(f"Export dir   : {export_dir}", flush=True)
    print("=" * 70, flush=True)

    if not (Path(adapter_dir) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_dir}. "
            "Did you run train_lora.py or train_qlora.py first?"
        )

    # 1. Load base model on CPU
    print("\nLoading base model on CPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # 2. Attach adapters
    print("Loading adapters...", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    # 3. Merge weights
    print("Merging weights (this may take a minute)...", flush=True)
    merged_model = peft_model.merge_and_unload()

    # 4. Save
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {export_dir} ...", flush=True)
    merged_model.save_pretrained(str(export_dir))

    tokenizer = get_tokenizer(model_id)
    tokenizer.save_pretrained(str(export_dir))

    print("\nMerge complete!", flush=True)
    print(f"You can now load the model from: {export_dir}", flush=True)
    print("=" * 70, flush=True)


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Merge LoRA Adapters into Base Model")
    parser.add_argument("--adapter-dir", type=Path, default=cfg.output_qlora)
    parser.add_argument("--export-dir", type=Path, default=cfg.output_merged)
    parser.add_argument("--model-id", type=str, default=cfg.model_id)
    args = parser.parse_args()

    run_merge(args.adapter_dir, args.export_dir, args.model_id)


if __name__ == "__main__":
    main()
