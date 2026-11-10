"""
pipeline/common.py
------------------
Shared configuration, constants and small helper functions used by
all training, evaluation, and inference pipelines.
Supports isolated per-run artifact trees under ARTIFACTS_ROOT.
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


# ============================================================
# PROJECT PATHS & STORAGE ROOTS
# ============================================================
PIPELINE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PIPELINE_DIR.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = PROJECT_ROOT / "raw_documents"
DEFAULT_JSONL = DATA_DIR / "private_docs.jsonl"

# Unified isolated artifacts root
ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", PROJECT_ROOT / "artifacts")).resolve()


def get_run_artifacts_dir(run_id: str, artifacts_root: Optional[Path] = None) -> Path:
    """
    Produce an isolated per-run workspace tree.
    Structure:
      ARTIFACTS_ROOT/runs/<run_id>/
        checkpoints/
        adapter/
        merged/
        gguf/
        logs/
    """
    root = artifacts_root or ARTIFACTS_ROOT
    run_dir = root / "runs" / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (run_dir / "merged").mkdir(parents=True, exist_ok=True)
    (run_dir / "gguf").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


# ============================================================
# CENTRAL CONFIGURATION
# ============================================================
@dataclass
class Config:
    """
    Single place to control model, data and training hyper-parameters.
    Import this class in every training script or access via web UI.
    """

    # --- Run Identification & Isolation ---
    run_id: Optional[str] = None
    output_dir: Optional[Path] = None

    # --- Model ---
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # --- Data ---
    dataset_path: Path = DEFAULT_JSONL
    dataset_version_id: Optional[str] = None
    dataset_sha256: Optional[str] = None
    max_seq_length: int = 1024
    val_split: float = 0.1
    loss_on: str = "assistant"                          # "assistant" (completion-only) or "all"
    packing: bool = True

    # --- LoRA / QLoRA defaults ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("all-linear",)             # Default to all linear modules (PEFT >= 0.10)

    # --- Training defaults ---
    num_train_epochs: int = 3
    learning_rate_lora: float = 2e-4
    learning_rate_fft: float = 1e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    seed: int = 42
    early_stopping_patience: Optional[int] = 3

    # --- Legacy Output directories (redirect to output_dir if set) ---
    output_fft: Path = PROJECT_ROOT / "output_fft"
    output_lora: Path = PROJECT_ROOT / "output_lora"
    output_qlora: Path = PROJECT_ROOT / "output_qlora"
    output_merged: Path = PROJECT_ROOT / "output_merged"

    def __post_init__(self):
        """Re-route output directories to isolated run path if output_dir is configured."""
        if self.output_dir:
            out = Path(self.output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            self.output_qlora = out / "adapter"
            self.output_lora = out / "adapter"
            self.output_fft = out / "adapter"
            self.output_merged = out / "merged"
        elif self.run_id:
            run_dir = get_run_artifacts_dir(self.run_id)
            self.output_dir = run_dir
            self.output_qlora = run_dir / "adapter"
            self.output_lora = run_dir / "adapter"
            self.output_fft = run_dir / "adapter"
            self.output_merged = run_dir / "merged"

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
            if k in ["dataset_path", "output_dir", "output_fft", "output_lora", "output_qlora", "output_merged"]:
                kwargs[k] = Path(v) if v else None
            elif k == "target_modules" and isinstance(v, (list, tuple)):
                kwargs[k] = tuple(v)
            elif hasattr(cls, k):
                field_type = cls.__annotations__.get(k)
                if field_type is int and v is not None:
                    kwargs[k] = int(v)
                elif field_type is float and v is not None:
                    kwargs[k] = float(v)
                elif field_type is bool and v is not None:
                    kwargs[k] = bool(v)
                else:
                    kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from JSON settings file if present,
        falling back to default dataclass parameters.
        """
        path = config_path or CONFIG_FILE
        if path.exists():
            try:
                import json
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                flattened: Dict[str, Any] = {}
                if "huggingface" in data and "default_model_id" in data["huggingface"]:
                    flattened["model_id"] = data["huggingface"]["default_model_id"]

                if "training_defaults" in data:
                    td = data["training_defaults"]
                    for k in [
                        "num_train_epochs", "per_device_train_batch_size",
                        "gradient_accumulation_steps", "learning_rate_lora",
                        "learning_rate_fft", "max_seq_length", "logging_steps", "seed",
                        "val_split", "loss_on", "packing"
                    ]:
                        if k in td:
                            flattened[k] = td[k]
                    if "lora" in td:
                        lora = td["lora"]
                        if "r" in lora:
                            flattened["lora_r"] = lora["r"]
                        if "alpha" in lora:
                            flattened["lora_alpha"] = lora["alpha"]
                        if "dropout" in lora:
                            flattened["lora_dropout"] = lora["dropout"]
                        if "target_modules" in lora:
                            flattened["target_modules"] = tuple(lora["target_modules"])

                return cls.from_dict(flattened)
            except Exception:
                return cls()
        return cls()


def get_tokenizer(model_id: str):
    """Load tokenizer and make sure pad_token is set."""
    if AutoTokenizer is None:
        raise ImportError("transformers is not installed. Please install requirements.txt")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def print_trainable_parameters(model):
    """Convenience wrapper that works for both PEFT and regular models."""
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
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
