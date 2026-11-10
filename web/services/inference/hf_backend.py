"""
web/services/inference/hf_backend.py
-----------------------------------
Hugging Face Transformers inference backend using PyTorch and PEFT.
Used when torch/transformers is installed and direct in-memory execution is requested.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator

from web.services.inference.base import InferenceBackend

logger = logging.getLogger("doctune.inference.hf")


class HuggingFaceBackend(InferenceBackend):
    """In-memory PyTorch / Transformers backend with PEFT LoRA adapter support."""

    name: str = "transformers"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded_model_id = None
        self.loaded_adapter_path = None

    async def is_available(self) -> bool:
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    async def health(self) -> Dict[str, Any]:
        has_torch = await self.is_available()
        cuda_avail = False
        device = "cpu"
        if has_torch:
            import torch
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                device = f"cuda ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

        return {
            "backend": self.name,
            "available": has_torch,
            "device": device,
            "loaded_model": self.loaded_model_id,
            "loaded_adapter": self.loaded_adapter_path,
        }

    def _ensure_loaded(self, model_id: str, adapter_path: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if self.model is not None and self.loaded_model_id == model_id and self.loaded_adapter_path == adapter_path:
            return

        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device in ("cuda", "mps") else torch.float32)

        logger.info("Loading HuggingFace model %s on %s (%s)", model_id, device, dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            model.to(device)

        if adapter_path and os.path.exists(adapter_path):
            logger.info("Applying PEFT adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model
        self.tokenizer = tokenizer
        self.loaded_model_id = model_id
        self.loaded_adapter_path = adapter_path

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not await self.is_available():
            raise RuntimeError("HuggingFace PyTorch backend unavailable: torch or transformers not installed.")

        import torch
        self._ensure_loaded(model, adapter_path)

        prompt_text = ""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        if not prompt_text:
            prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.01, temperature),
                do_sample=temperature > 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "id": f"hf-{int(time.time()*1000)}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs["input_ids"][0]),
                "completion_tokens": len(new_tokens),
                "total_tokens": len(outputs[0]),
            },
        }

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        # For simplicity, generate and chunk
        result = await self.chat(messages, model, temperature, max_tokens, adapter_path)
        content = result["choices"][0]["message"]["content"]

        # Stream words as mock delta chunks
        for word in content.split(" "):
            chunk = {
                "choices": [{"delta": {"content": word + " "}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"
