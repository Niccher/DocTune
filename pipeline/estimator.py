"""
pipeline/estimator.py
---------------------
Real-time VRAM requirement and training feasibility estimator.
Inspects model metadata directly from Hugging Face Hub (without downloading full weights)
or architecture heuristics to calculate a breakdown of weights, activations, and optimizer memory.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pipeline.hardware import get_hardware_profile, HardwareProfile

logger = logging.getLogger("doctune.estimator")

# Known model parameter counts fallback catalog
PARAMETER_CATALOG = {
    "huggingfacetb/smollm2-135m-instruct": 135_000_000,
    "huggingfacetb/smollm2-360m-instruct": 360_000_000,
    "qwen/qwen2.5-0.5b-instruct": 500_000_000,
    "qwen/qwen2.5-1.5b-instruct": 1_540_000_000,
    "qwen/qwen2.5-3b-instruct": 3_090_000_000,
    "qwen/qwen2.5-7b-instruct": 7_610_000_000,
    "google/gemma-2-2b-it": 2_600_000_000,
    "google/gemma-2-9b-it": 9_240_000_000,
    "meta-llama/meta-llama-3-8b-instruct": 8_030_000_000,
    "meta-llama/llama-3.1-8b-instruct": 8_030_000_000,
    "meta-llama/llama-3.2-1b-instruct": 1_230_000_000,
    "meta-llama/llama-3.2-3b-instruct": 3_210_000_000,
    "mistralai/mistral-7b-instruct-v0.3": 7_250_000_000,
    "microsoft/phi-3-mini-4k-instruct": 3_820_000_000,
}


def estimate_parameter_count(model_id: str) -> int:
    """Resolve model parameter count via Hugging Face Hub or catalog heuristics."""
    clean_id = model_id.lower().strip()
    if clean_id in PARAMETER_CATALOG:
        return PARAMETER_CATALOG[clean_id]

    # Try inspecting config.json via huggingface_hub without downloading weights
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id=model_id, filename="config.json", timeout=3.0)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Approximate from layers and hidden dims
        hidden_size = cfg.get("hidden_size", 4096)
        num_hidden_layers = cfg.get("num_hidden_layers", 32)
        vocab_size = cfg.get("vocab_size", 32000)
        # Transformer parameter estimate ~ 12 * layers * hidden^2 + vocab * hidden
        approx = (12 * num_hidden_layers * (hidden_size ** 2)) + (vocab_size * hidden_size)
        return int(approx)
    except Exception:
        pass

    # Heuristic guess from string name (e.g., "7b", "8b", "13b", "70b")
    for tag, count in [
        ("70b", 70_000_000_000),
        ("14b", 14_000_000_000),
        ("13b", 13_000_000_000),
        ("9b", 9_000_000_000),
        ("8b", 8_000_000_000),
        ("7b", 7_000_000_000),
        ("3b", 3_000_000_000),
        ("2b", 2_000_000_000),
        ("1.5b", 1_500_000_000),
        ("1b", 1_000_000_000),
        ("500m", 500_000_000),
        ("360m", 360_000_000),
        ("135m", 135_000_000),
    ]:
        if tag in clean_id:
            return count

    return 7_000_000_000  # Default 7B fallback


def estimate_vram_and_feasibility(
    model_id: str,
    method: str = "qlora",
    seq_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    lora_r: int = 16,
    gradient_checkpointing: bool = True,
    hardware: Optional[HardwareProfile] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute a memory breakdown in GB and return feasibility assessment.
    Accepts aliases: max_seq_length, per_device_batch_size, total_vram_gb.
    """
    seq_length = seq_length if seq_length is not None else kwargs.get("max_seq_length", 1024)
    batch_size = batch_size if batch_size is not None else kwargs.get("per_device_batch_size", 4)
    total_vram_override = kwargs.get("total_vram_gb")

    hw = hardware or get_hardware_profile()
    params = estimate_parameter_count(model_id)
    norm_method = method.lower().replace("sft-", "")

    # 1. Base Model Weights
    if norm_method == "qlora":
        # 4-bit quantized base weights ~ 0.55 bytes per parameter
        weights_bytes = params * 0.55
    elif norm_method == "lora":
        # 16-bit half precision base weights ~ 2.1 bytes per parameter
        weights_bytes = params * 2.1
    else:
        # Full Fine-Tuning (FFT) 16-bit model weights
        weights_bytes = params * 2.1

    weights_gb = round(weights_bytes / (1024 ** 3), 2)

    # 2. Activations Memory
    # Base activations ~ batch_size * seq_len * (hidden_dim ~ sqrt(params/12/layers))
    # Empirical scaling: ~ 0.00000000035 * params * seq_len * batch_size
    raw_act = (params / 1_000_000_000) * (seq_length / 1024) * batch_size * 0.45 * (1024 ** 3)
    if gradient_checkpointing:
        # Gradient checkpointing trades recomputation for memory: ~70% activation reduction
        act_bytes = raw_act * 0.32
    else:
        act_bytes = raw_act

    activations_gb = round(act_bytes / (1024 ** 3), 2)

    # 3. Optimizer & Gradients Memory
    if norm_method in ("qlora", "lora"):
        # LoRA only trains adapter matrices (typically ~0.5% - 2% of parameters)
        trainable_ratio = 0.01 * (lora_r / 16)
        trainable_params = params * trainable_ratio
        # 8-bit AdamW ~ 6 bytes per trainable parameter
        optimizer_bytes = (trainable_params * 6) + 250_000_000  # + buffer
    else:
        # FFT trains 100% of parameters with 8-bit / 16-bit optimizer states ~ 8-16 bytes/param
        optimizer_bytes = params * 8

    optimizer_gb = round(optimizer_bytes / (1024 ** 3), 2)

    # 4. Total and Feasibility Check
    # Include 1.0 GB PyTorch CUDA context / PyTorch caching overhead
    overhead_gb = 1.0 if hw.device == "cuda" else 0.5
    total_gb = round(weights_gb + activations_gb + optimizer_gb + overhead_gb, 2)

    vram_avail = total_vram_override if total_vram_override is not None else hw.vram_gb
    if hw.device == "cpu" and total_vram_override is None:
        # For CPU, evaluate against RAM (estimate 16GB available)
        vram_avail = 16.0

    fits = total_gb <= vram_avail

    # Suggestions if not fitting
    suggestion = "Configuration looks optimal for your hardware."
    if not fits:
        if norm_method == "fft":
            suggestion = f"Full Fine-Tuning requires ~{total_gb}GB VRAM. Switch to QLoRA (4-bit) or LoRA to fit in {vram_avail}GB."
        elif norm_method == "lora" and hw.bnb_ok:
            suggestion = f"LoRA 16-bit requires ~{total_gb}GB VRAM. Enable QLoRA (4-bit quantization) to reduce memory to ~{round(total_gb * 0.45, 1)}GB."
        elif batch_size > 1:
            suggestion = f"Reduce per-device batch size from {batch_size} to 1 or 2, and increase gradient accumulation steps."
        elif seq_length > 512:
            suggestion = f"Reduce max sequence length from {seq_length} to 512 or 1024."
        else:
            suggestion = f"Model {model_id} is too large for {vram_avail}GB VRAM. Choose a smaller base model (e.g., Llama-3.2-3B, Gemma-2-2B, or SmolLM2-135M)."

    breakdown = {
        "weights_gb": weights_gb,
        "activations_gb": activations_gb,
        "optimizer_gb": optimizer_gb,
        "overhead_gb": overhead_gb,
        "total_estimated_gb": total_gb,
    }

    return {
        "model_id": model_id,
        "method": method,
        "param_count": params,
        "param_count_human": f"{params / 1_000_000_000:.1f}B" if params >= 1_000_000_000 else f"{params / 1_000_000:.0f}M",
        "weights_gb": weights_gb,
        "activations_gb": activations_gb,
        "optimizer_gb": optimizer_gb,
        "overhead_gb": overhead_gb,
        "total_gb": total_gb,
        "vram_available_gb": vram_avail,
        "device": hw.device,
        "fits": fits,
        "feasible": fits,
        "breakdown": breakdown,
        "suggestion": suggestion,
    }

