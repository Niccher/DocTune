"""
common.py
---------
Shared configuration, constants and small helper functions used by
all training and evaluation scripts.

Change the values in the Config class to match your environment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer


# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_JSONL = DATA_DIR / "private_docs.jsonl"


# ============================================================
# CENTRAL CONFIGURATION
# ============================================================
@dataclass
class Config:
    """
    Single place to control model, data and training hyper-parameters.
    Import this class in every training script.
    """

    # --- Model ---
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Alternative examples:
    # model_id: str = "google/gemma-2-9b-it"
    # model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # --- Data ---
    dataset_path: Path = DEFAULT_JSONL
    max_seq_length: int = 1024

    # --- LoRA / QLoRA defaults (can be overridden) ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",          # Attention
        "gate_proj", "up_proj", "down_proj",             # MLP / Feed-forward
    )

    # --- Training defaults ---
    num_train_epochs: int = 3
    learning_rate_lora: float = 2e-4
    learning_rate_fft: float = 1e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    seed: int = 42

    # --- Output directories ---
    output_fft: Path = PROJECT_ROOT / "output_fft"
    output_lora: Path = PROJECT_ROOT / "output_lora"
    output_qlora: Path = PROJECT_ROOT / "output_qlora"
    output_merged: Path = PROJECT_ROOT / "output_merged"


def get_tokenizer(model_id: str):
    """
    Load tokenizer and make sure pad_token is set.
    Llama-3 and many modern models do not define pad_token by default.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def print_trainable_parameters(model):
    """
    Convenience wrapper that works for both PEFT and regular models.
    """
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} "
              f"|| trainable%: {100 * trainable / total:.4f}%")


def get_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Create data directory if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
