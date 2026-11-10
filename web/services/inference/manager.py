"""
web/services/inference/manager.py
---------------------------------
Inference manager coordinating llama.cpp (default), Ollama, and HuggingFace backends.
Enforces API key authentication and manages active checkpoint deployments.
Eliminates simulated/canned responses.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from pathlib import Path

from pipeline.common import PROJECT_ROOT, Config
from web.services.inference.base import InferenceBackend
from web.services.inference.llamacpp_backend import LlamaCppBackend
from web.services.inference.ollama_backend import OllamaBackend
from web.services.inference.hf_backend import HuggingFaceBackend
from web.db.session import SessionLocal
from web.db.models import Checkpoint, ApiKey

logger = logging.getLogger("doctune.inference.manager")


class InferenceManager:
    """Coordinates multi-engine LLM inference with llama.cpp as default priority."""

    def __init__(self):
        self.llamacpp = LlamaCppBackend()
        self.ollama = OllamaBackend()
        self.hf = HuggingFaceBackend()
        self.preferred_backend = os.getenv("DEFAULT_INFERENCE_ENGINE", "llamacpp").lower()

    async def get_active_backend(self) -> InferenceBackend:
        """
        Determine and return the active, responsive backend.
        Priority:
        1. llama.cpp (default requested engine)
        2. Ollama (if running)
        3. HuggingFace PyTorch (if torch installed)
        Fallback: llama.cpp (will raise descriptive guidance on connection failure)
        """
        if self.preferred_backend in ("llamacpp", "llama.cpp"):
            if await self.llamacpp.is_available():
                return self.llamacpp

        if await self.ollama.is_available():
            return self.ollama

        if await self.hf.is_available():
            return self.hf

        # Return llama.cpp as default backend
        return self.llamacpp

    def authenticate_api_key(self, token: str) -> Optional[ApiKey]:
        """
        Verify API key from Authorization header against api_keys table.
        Format: dt_live_...
        """
        if not token:
            return None
        clean_token = token.replace("Bearer ", "").strip()
        if not clean_token:
            return None

        import hashlib
        key_hash = hashlib.sha256(clean_token.encode("utf-8")).hexdigest()

        if SessionLocal:
            db = SessionLocal()
            try:
                ak = db.query(ApiKey).filter(ApiKey.key_hash == key_hash, ApiKey.revoked_at.is_(None)).first()
                if ak:
                    ak.last_used_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    db.commit()
                    return ak
            finally:
                db.close()
        return None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List registered checkpoints, base models, and GGUF packages."""
        models = []
        now_ts = int(time.time())
        cfg = Config.load()

        # 1. Base model
        models.append({
            "id": cfg.model_id,
            "object": "model",
            "created": now_ts,
            "owned_by": "base",
            "type": "base_model",
        })

        # 2. Checkpoints from database
        if SessionLocal:
            try:
                db = SessionLocal()
                ckpts = db.query(Checkpoint).all()
                for c in ckpts:
                    models.append({
                        "id": c.id,
                        "name": c.name,
                        "object": "model",
                        "created": int(c.created_at.timestamp()) if c.created_at else now_ts,
                        "owned_by": "doctune",
                        "type": c.adapter_type or "adapter",
                        "base_model": c.base_model or cfg.model_id,
                        "disk_path": c.disk_path,
                    })
                db.close()
            except Exception:
                pass

        # 3. Local checkpoint output dirs if not in DB
        for path, name, typ in [
            (cfg.output_merged, "doctune-merged", "merged"),
            (cfg.output_qlora, "doctune-qlora", "qlora"),
            (cfg.output_lora, "doctune-lora", "lora"),
        ]:
            if path.exists() and not any(m["id"] == name for m in models):
                models.append({
                    "id": name,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "doctune",
                    "type": typ,
                    "disk_path": str(path),
                })

        return models

    def resolve_model_and_adapter(self, requested_model: str) -> tuple[str, Optional[str]]:
        """
        Resolve a requested model ID to base model and optional adapter path.
        Returns (base_model_id, adapter_path).
        """
        cfg = Config.load()
        if not requested_model or requested_model in ("default", "doctune-default"):
            return cfg.model_id, None

        # Check DB checkpoints
        if SessionLocal:
            try:
                db = SessionLocal()
                ckpt = db.query(Checkpoint).filter(Checkpoint.id == requested_model).first()
                if ckpt:
                    base = ckpt.base_model or cfg.model_id
                    adapter = ckpt.disk_path if ckpt.adapter_type in ("qlora", "lora") else None
                    return base, adapter
                db.close()
            except Exception:
                pass

        if requested_model == "doctune-merged":
            return str(cfg.output_merged), None
        elif requested_model == "doctune-qlora":
            return cfg.model_id, str(cfg.output_qlora)
        elif requested_model == "doctune-lora":
            return cfg.model_id, str(cfg.output_lora)

        return requested_model, None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Route chat request to active backend with zero hard-coded responses."""
        backend = await self.get_active_backend()
        base_model, adapter = self.resolve_model_and_adapter(model)
        return await backend.chat(
            messages=messages,
            model=base_model,
            temperature=temperature,
            max_tokens=max_tokens,
            adapter_path=adapter,
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        """Stream chat request from active backend."""
        backend = await self.get_active_backend()
        base_model, adapter = self.resolve_model_and_adapter(model)
        async for chunk in backend.stream_chat(
            messages=messages,
            model=base_model,
            temperature=temperature,
            max_tokens=max_tokens,
            adapter_path=adapter,
        ):
            yield chunk

    async def get_system_health(self) -> Dict[str, Any]:
        """Check all backend statuses."""
        return {
            "default_engine": "llama.cpp",
            "llamacpp": await self.llamacpp.health(),
            "ollama": await self.ollama.health(),
            "transformers": await self.hf.health(),
        }


inference_manager = InferenceManager()
