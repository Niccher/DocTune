"""
pipeline/trainer/__init__.py
----------------------------
Unified fine-tuning engine supporting SFT (QLoRA, LoRA, Full) and preference alignment.
"""

from pipeline.trainer.runner import run_training

__all__ = ["run_training"]
