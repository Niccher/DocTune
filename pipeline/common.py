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
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"
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
                        "learning_rate_fft", "max_seq_length", "logging_steps", "seed"
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

                if "storage_paths" in data:
                    sp = data["storage_paths"]
                    if "data_dir" in sp:
                        flattened["dataset_path"] = PROJECT_ROOT / sp["data_dir"] / "private_docs.jsonl"
                    if "output_fft_dir" in sp:
                        flattened["output_fft"] = PROJECT_ROOT / sp["output_fft_dir"]
                    if "output_lora_dir" in sp:
                        flattened["output_lora"] = PROJECT_ROOT / sp["output_lora_dir"]
                    if "output_qlora_dir" in sp:
                        flattened["output_qlora"] = PROJECT_ROOT / sp["output_qlora_dir"]
                    if "output_merged_dir" in sp:
                        flattened["output_merged"] = PROJECT_ROOT / sp["output_merged_dir"]

                return cls.from_dict(flattened)
            except Exception:
                return cls()
        return cls()



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
