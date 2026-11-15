"""
pipeline/errors.py
------------------
Structured exception classification and actionable diagnostic hints.
Translates cryptic stack traces (CUDA OOM, 401/403 gated repo, NaN loss, missing chat template)
into clear, actionable UI guidance.
"""

import re
from typing import Dict, Any, Optional

# Structured Error Codes
CUDA_OOM = "CUDA_OOM"
AUTH_GATED_REPO = "AUTH_GATED_REPO"
MISSING_CHAT_TEMPLATE = "MISSING_CHAT_TEMPLATE"
LOSS_DIVERGENCE_NAN = "LOSS_DIVERGENCE_NAN"
STORAGE_EXHAUSTED = "STORAGE_EXHAUSTED"
GPU_REQUIRED_FOR_QLORA = "GPU_REQUIRED_FOR_QLORA"
BITSANDBYTES_MISSING = "BITSANDBYTES_MISSING"
BATCH_SIZE_EXCEEDS_DATASET = "BATCH_SIZE_EXCEEDS_DATASET"
PIPELINE_ERROR = "PIPELINE_ERROR"


def classify_training_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze exception string and context to produce a structured diagnostic dictionary.
    Returns:
      {
        "code": str,
        "title": str,
        "message": str,
        "hint": str
      }
    """
    err_str = str(error)
    err_type = type(error).__name__
    ctx = context or {}

    # 1. CUDA Out-of-Memory (OOM)
    if "CUDA out of memory" in err_str or "OutOfMemoryError" in err_type:
        match = re.search(r"tried to allocate (\d+(?:\.\d+)?)\s*(?:GiB|MiB)", err_str, re.IGNORECASE)
        alloc_detail = f" (allocation attempt: {match.group(0)})" if match else ""
        return {
            "code": "CUDA_OOM",
            "title": "GPU Out of Memory (CUDA OOM)",
            "message": f"PyTorch exceeded available GPU VRAM{alloc_detail}.",
            "hint": "Try: (1) Switch to QLoRA (4-bit), (2) Decrease per-device batch size to 1 or 2, (3) Enable sample packing and gradient checkpointing, or (4) Reduce max sequence length.",
        }

    # 2. Gated Hugging Face Model / 401 Unauthorized
    if "401" in err_str or "403" in err_str or "gated repo" in err_str.lower() or "restricted" in err_str.lower():
        model_id = ctx.get("model_id", "the selected base model")
        return {
            "code": "AUTH_GATED_REPO",
            "title": "Gated Model Access Denied",
            "message": f"Hugging Face repository '{model_id}' requires accepted terms and an access token.",
            "hint": "Go to Hugging Face, accept the model license terms, generate an Access Token (Read permissions), and configure it in Studio Settings.",
        }

    # 3. Missing Chat Template
    if "chat template" in err_str.lower() or "apply_chat_template" in err_str:
        return {
            "code": "CHAT_TEMPLATE_MISSING",
            "title": "Missing Tokenizer Chat Template",
            "message": "The base model tokenizer has no built-in chat template.",
            "hint": "Select an '-Instruct' or '-Chat' variant of the model, or specify a custom chat template in the advanced settings drawer.",
        }

    # 4. NaN / Loss Divergence
    if "loss is nan" in err_str.lower() or "diverged" in err_str.lower():
        return {
            "code": "LOSS_DIVERGENCE_NAN",
            "title": "Training Loss Diverged (NaN)",
            "message": "Numerical overflow occurred during backpropagation, resulting in NaN loss.",
            "hint": "Try: (1) Reduce the learning rate by 2x to 5x (e.g. from 2e-4 to 5e-5), (2) Ensure bfloat16 or float32 precision is used instead of float16, or (3) Enable max gradient norm clipping (e.g. 0.3).",
        }

    # 5. Disk / Storage Quota Exhausted
    if "No space left on device" in err_str or "Errno 28" in err_str:
        return {
            "code": "STORAGE_EXHAUSTED",
            "title": "Storage Volume Full",
            "message": "The local artifact or container filesystem has run out of disk space.",
            "hint": "Prune intermediate checkpoints in Settings -> Storage, or increase your storage volume allocation.",
        }

    # 6. GPU Required for QLoRA 4-Bit
    if "requires an nvidia gpu" in err_str.lower() or ("qlora" in err_str.lower() and "requires" in err_str.lower()):
        return {
            "code": "GPU_REQUIRED_FOR_QLORA",
            "title": "NVIDIA GPU Required for QLoRA (4-Bit)",
            "message": "QLoRA 4-bit NormalFloat (NF4) quantization strictly requires an NVIDIA GPU with CUDA. Your environment is running on Host CPU.",
            "hint": "Switch the fine-tuning method to LoRA (16-bit), or launch the worker container with an NVIDIA GPU attached (via docker-compose.gpu.yml).",
        }

    # 7. bitsandbytes Unavailable
    if "bitsandbytes" in err_str.lower() and ("not found" in err_str.lower() or "not compiled" in err_str.lower()):
        return {
            "code": "BITSANDBYTES_MISSING",
            "title": "bitsandbytes 4-Bit Acceleration Unavailable",
            "message": "bitsandbytes is not compiled or supported for your device architecture.",
            "hint": "QLoRA requires an NVIDIA GPU with bitsandbytes. On Apple Silicon (MPS) or CPU, select standard LoRA (16-bit) or GGUF quantization.",
        }

    # 8. Batch Size Exceeds Dataset
    if "batch size" in err_str.lower() and ("dataset" in err_str.lower() or "sample" in err_str.lower() or "exceed" in err_str.lower()):
        return {
            "code": "BATCH_SIZE_EXCEEDS_DATASET",
            "title": "Batch Size Exceeds Dataset Count",
            "message": "Your training batch size is larger than the total number of samples in the dataset.",
            "hint": "Decrease the per-device batch size to 1 or 2 in Training Hyperparameters, or generate more dataset pairs in Data Prep Studio.",
        }

    # Generic Fallback
    return {
        "code": "PIPELINE_ERROR",
        "title": "Pipeline Execution Error",
        "message": err_str[:300],
        "hint": f"Failure reason: {err_str[:200]}. Inspect complete worker logs for detailed traceback.",
    }


classify_training_exception = classify_training_error
TrainingFailureDiagnostic = Dict[str, Any]
