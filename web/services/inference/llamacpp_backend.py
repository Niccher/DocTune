"""
web/services/inference/llamacpp_backend.py
-----------------------------------------
Default inference backend utilizing llama.cpp (llama-server or llama-cpp-python).
Provides high-performance, cross-platform inference across CPU, Apple Metal, and CUDA.
Supports GGUF quantized models and dynamic LoRA adapter loading.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import httpx

from web.services.inference.base import InferenceBackend

logger = logging.getLogger("doctune.inference.llamacpp")


class LlamaCppBackend(InferenceBackend):
    """Primary default inference engine backed by llama.cpp / llama-server."""

    name: str = "llama.cpp"

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = (endpoint or os.getenv("LLAMA_CPP_ENDPOINT") or "http://localhost:8080").rstrip("/")
        self._timeout = float(os.getenv("LLAMA_CPP_TIMEOUT", "120.0"))
        self._in_process_model = None
        self._in_process_model_path = None
        self._in_process_lora_path = None

    async def is_available(self) -> bool:
        """Check if llama-server is healthy or in-process llama_cpp is installed."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.endpoint}/health")
                if res.status_code == 200:
                    return True
                # Also try /v1/models
                res = await client.get(f"{self.endpoint}/v1/models")
                if res.status_code == 200:
                    return True
        except Exception:
            pass

        # Check in-process binding
        try:
            import llama_cpp
            return True
        except ImportError:
            pass

        return False

    async def health(self) -> Dict[str, Any]:
        """Check server endpoint status and hardware execution info."""
        server_running = False
        server_info = {}
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.endpoint}/health")
                if res.status_code == 200:
                    server_running = True
                    try:
                        server_info = res.json()
                    except Exception:
                        server_info = {"status": "ok"}
        except Exception as e:
            server_info = {"error": str(e)}

        in_proc = False
        try:
            import llama_cpp
            in_proc = True
        except ImportError:
            pass

        return {
            "backend": self.name,
            "endpoint": self.endpoint,
            "server_running": server_running,
            "in_process_available": in_proc,
            "details": server_info,
        }

    async def _resolve_lora_payload(self, client: httpx.AsyncClient, adapter_path: str) -> Any:
        """Resolve adapter path or ID using llama-server registered /lora-adapters."""
        try:
            res = await client.get(f"{self.endpoint}/lora-adapters")
            if res.status_code == 200:
                adapters = res.json()
                if isinstance(adapters, list):
                    for a in adapters:
                        if isinstance(a, dict) and (a.get("path") == adapter_path or os.path.basename(adapter_path) in a.get("path", "")):
                            return [{"id": a.get("id", 0), "scale": 1.0}]
        except Exception:
            pass
        return [{"id": 0, "scale": 1.0}] if adapter_path.isdigit() else adapter_path

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute chat completion via llama-server /v1/chat/completions or in-process."""
        # 1. Try remote llama-server
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if adapter_path:
                    payload["lora"] = await self._resolve_lora_payload(client, adapter_path)

                res = await client.post(f"{self.endpoint}/v1/chat/completions", json=payload)
                if res.status_code == 200:
                    return res.json()
                elif res.status_code != 404:
                    logger.warning("llama-server returned HTTP %d: %s", res.status_code, res.text)
        except httpx.ConnectError:
            pass
        except Exception as e:
            logger.warning("Error querying llama-server: %s", e)

        # 2. Try in-process llama_cpp if installed
        try:
            import llama_cpp
            if (
                self._in_process_model is None
                or self._in_process_model_path != model
                or self._in_process_lora_path != adapter_path
            ):
                # Load GGUF model if path exists
                if os.path.exists(model):
                    self._in_process_model = llama_cpp.Llama(
                        model_path=model,
                        lora_path=adapter_path,
                        n_ctx=2048,
                        verbose=False,
                    )
                    self._in_process_model_path = model
                    self._in_process_lora_path = adapter_path

            if self._in_process_model:
                out = self._in_process_model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return out
        except ImportError:
            pass
        except Exception as e:
            logger.warning("In-process llama_cpp error: %s", e)

        # If neither is running, return structured service unavailable error
        raise RuntimeError(
            f"llama.cpp inference engine is currently not reachable at {self.endpoint}. "
            "Please start the llama-server process (e.g., `llama-server -m <model.gguf> --port 8080`) "
            "or install `llama-cpp-python`."
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks via SSE."""
        client = httpx.AsyncClient(timeout=self._timeout)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if adapter_path:
            payload["lora"] = await self._resolve_lora_payload(client, adapter_path)

        try:
            async with client.stream("POST", f"{self.endpoint}/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield f"data: {json.dumps({'error': f'llama-server returned HTTP {response.status_code}: {err_body.decode()}'})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
                        if "[DONE]" in line:
                            break
        except httpx.ConnectError:
            err_msg = (
                f"Cannot connect to llama-server at {self.endpoint}. "
                "Ensure llama-server is started."
            )
            yield f"data: {json.dumps({'error': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await client.aclose()
