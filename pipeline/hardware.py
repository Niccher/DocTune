"""
pipeline/hardware.py
--------------------
Dynamic hardware profiling and optimal precision/optimizer selection.
Gracefully adapts across NVIDIA CUDA (Ampere+, Turing, Pascal), Apple Silicon (MPS),
and CPU without crashing on missing bitsandbytes or non-bfloat16 architectures.
"""

import os
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

logger = logging.getLogger("doctune.hardware")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def get_system_ram_gb() -> float:
    """Retrieve host system RAM in gigabytes."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        kb = float(parts[1])
                        return round(kb / (1024 ** 2), 2)
    except Exception:
        pass
    return 0.0


@dataclass
class HardwareProfile:
    """Hardware capabilities profile detected on the host or worker container."""
    device: str                                         # "cuda", "mps", "cpu"
    device_name: str                                    # e.g., "NVIDIA GeForce RTX 4090"
    vram_gb: float                                      # Available VRAM in gigabytes
    n_gpus: int                                         # Number of accessible GPUs
    compute_capability: Optional[str]                   # e.g., "8.9"
    bf16_ok: bool                                       # True if hardware natively supports bfloat16 (Ampere+)
    fp16_ok: bool                                       # True if hardware natively supports float16
    bnb_ok: bool                                        # True if bitsandbytes 4-bit / 8-bit is supported and working
    flash_attn_ok: bool                                 # True if FlashAttention-2 is available
    recommended_dtype: str                              # "bfloat16", "float16", "float32"
    recommended_optim: str                              # "paged_adamw_8bit", "adamw_torch", "adamw_bnb_8bit"
    system_ram_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def device_type(self) -> str:
        return self.device

    @property
    def gpu_count(self) -> int:
        return self.n_gpus

    @property
    def cuda_available(self) -> bool:
        return self.device == "cuda"

    @property
    def total_ram_gb(self) -> float:
        if self.device == "cuda" and self.vram_gb > 0:
            return self.vram_gb
        return self.system_ram_gb if self.system_ram_gb > 0 else self.vram_gb


def detect_hardware() -> HardwareProfile:
    """Detect local compute capabilities, precision support, and memory."""
    sys_ram = get_system_ram_gb()
    if not HAS_TORCH or torch is None:
        return HardwareProfile(
            device="cpu",
            device_name="Generic CPU (No PyTorch)",
            vram_gb=0.0,
            n_gpus=0,
            compute_capability=None,
            bf16_ok=False,
            fp16_ok=False,
            bnb_ok=False,
            flash_attn_ok=False,
            recommended_dtype="float32",
            recommended_optim="adamw_torch",
            system_ram_gb=sys_ram,
        )

    # 1. CUDA Architecture
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        cc_str = f"{major}.{minor}"

        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(total_vram_bytes / (1024 ** 3), 2)

        # Ampere (SM 8.0+) or newer natively supports bfloat16 fast path
        bf16_ok = major >= 8
        fp16_ok = major >= 7

        # Test bitsandbytes 4-bit / 8-bit compatibility
        bnb_ok = False
        try:
            import bitsandbytes as bnb
            bnb_ok = True
        except Exception:
            bnb_ok = False

        # FlashAttention-2 check
        flash_attn_ok = False
        if major >= 8:
            try:
                import flash_attn
                flash_attn_ok = True
            except ImportError:
                flash_attn_ok = False

        recommended_dtype = "bfloat16" if bf16_ok else ("float16" if fp16_ok else "float32")
        recommended_optim = "paged_adamw_8bit" if bnb_ok else "adamw_torch"

        return HardwareProfile(
            device="cuda",
            device_name=dev_name,
            vram_gb=vram_gb,
            n_gpus=n_gpus,
            compute_capability=cc_str,
            bf16_ok=bf16_ok,
            fp16_ok=fp16_ok,
            bnb_ok=bnb_ok,
            flash_attn_ok=flash_attn_ok,
            recommended_dtype=recommended_dtype,
            recommended_optim=recommended_optim,
            system_ram_gb=sys_ram,
        )

    # 2. Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareProfile(
            device="mps",
            device_name="Apple Silicon (Metal Performance Shaders)",
            vram_gb=16.0,  # Shared Unified Memory approximation
            n_gpus=1,
            compute_capability=None,
            bf16_ok=True,  # MPS supports bfloat16 in PyTorch 2.1+
            fp16_ok=True,
            bnb_ok=False,  # bitsandbytes is CUDA-specific
            flash_attn_ok=False,
            recommended_dtype="bfloat16",
            recommended_optim="adamw_torch",
            system_ram_gb=sys_ram,
        )

    # 3. CPU Fallback
    return HardwareProfile(
        device="cpu",
        device_name="Host CPU",
        vram_gb=0.0,
        n_gpus=0,
        compute_capability=None,
        bf16_ok=False,
        fp16_ok=False,
        bnb_ok=False,
        flash_attn_ok=False,
        recommended_dtype="float32",
        recommended_optim="adamw_torch",
        system_ram_gb=sys_ram,
    )


# Cached singleton
_HARDWARE_CACHE: Optional[HardwareProfile] = None

def get_hardware_profile(force_refresh: bool = False) -> HardwareProfile:
    """Retrieve cached hardware profile."""
    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is None or force_refresh:
        _HARDWARE_CACHE = detect_hardware()
    return _HARDWARE_CACHE


def configure_training_hardware(method: str, hw: Optional[HardwareProfile] = None) -> Dict[str, Any]:
    """
    Select appropriate precision flags, optimizers, and bitsandbytes configuration
    based on the validated hardware profile.
    """
    profile = hw or get_hardware_profile()
    is_qlora = method.lower() == "sft-qlora" or method.lower() == "qlora"

    if is_qlora and not profile.bnb_ok and profile.device != "cuda":
        raise RuntimeError(
            f"QLoRA 4-bit quantization requires an NVIDIA GPU with bitsandbytes. "
            f"Detected environment: {profile.device_name} ({profile.device}). "
            f"Please select LoRA (16-bit) or GGUF fine-tuning instead."
        )

    # Dtype mapping
    if profile.bf16_ok:
        torch_dtype = "bfloat16"
        bf16_flag = True
        fp16_flag = False
    elif profile.fp16_ok:
        torch_dtype = "float16"
        bf16_flag = False
        fp16_flag = True
    else:
        torch_dtype = "float32"
        bf16_flag = False
        fp16_flag = False

    # Optimizer selection
    if profile.bnb_ok and profile.device == "cuda":
        optim = "paged_adamw_8bit"
    else:
        optim = "adamw_torch"

    # Attention implementation
    if profile.flash_attn_ok:
        attn_impl = "flash_attention_2"
    elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        attn_impl = "sdpa"
    else:
        attn_impl = "eager"

    return {
        "device": profile.device,
        "torch_dtype": torch_dtype,
        "bf16": bf16_flag,
        "fp16": fp16_flag,
        "optim": optim,
        "attn_implementation": attn_impl,
        "bnb_4bit_compute_dtype": torch_dtype,
    }
