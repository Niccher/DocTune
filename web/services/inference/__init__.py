"""
web/services/inference/__init__.py
----------------------------------
Modular LLM inference subsystem supporting llama.cpp (default), Ollama, and PyTorch backends.
"""

from web.services.inference.base import InferenceBackend
from web.services.inference.llamacpp_backend import LlamaCppBackend
from web.services.inference.ollama_backend import OllamaBackend
from web.services.inference.hf_backend import HuggingFaceBackend
from web.services.inference.manager import inference_manager, InferenceManager

__all__ = [
    "InferenceBackend",
    "LlamaCppBackend",
    "OllamaBackend",
    "HuggingFaceBackend",
    "InferenceManager",
    "inference_manager",
]
