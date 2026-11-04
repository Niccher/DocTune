"""
pipeline/common.py
------------------
Shared configuration, constants and small helper functions used by
all training and evaluation scripts.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


# ============================================================
# PROJECT PATHS
# ============================================================
PIPELINE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PIPELINE_DIR.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = PROJECT_ROOT / "raw_documents"
DEFAULT_JSONL = DATA_DIR / "private_docs.jsonl"


# ============================================================
# CENTRAL CONFIGURATION
# ============================================================
@dataclass
class Config:
    """
    Single place to control model, data and training hyper-parameters.
    Import this class in every training script or access via web UI.
    """

    # --- Model ---
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Instantiate Config from dictionary with type conversions."""
        kwargs = {}
        for k, v in data.items():
            if k in ["dataset_path", "output_fft", "output_lora", "output_qlora", "output_merged"]:
                kwargs[k] = Path(v) if v else getattr(cls, k)
            elif k == "target_modules" and isinstance(v, list):
                kwargs[k] = tuple(v)
            elif hasattr(cls, k):
                field_type = cls.__annotations__.get(k)
                if field_type is int:
                    kwargs[k] = int(v)
                elif field_type is float:
                    kwargs[k] = float(v)
                else:
                    kwargs[k] = v
        return cls(**kwargs)


def get_tokenizer(model_id: str):
    """
    Load tokenizer and make sure pad_token is set.
    """
    if AutoTokenizer is None:
        raise ImportError("transformers is not installed. Please install requirements.txt")

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
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
