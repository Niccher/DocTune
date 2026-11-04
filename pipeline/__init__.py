"""
pipeline package
----------------
Private Document Fine-Tuning Pipeline components:
- common: configuration and helpers
- prepare_data: documents chunking and teacher LLM dataset generation
- train_qlora: 4-bit QLoRA fine-tuning
- train_lora: 16-bit LoRA fine-tuning
- train_fft: Full parameter fine-tuning
- evaluate: Fact retrieval and hallucination evaluation
- merge_adapters: Standalone model merge utility
"""

from pipeline.common import Config, get_tokenizer, get_device

__all__ = ["Config", "get_tokenizer", "get_device"]
