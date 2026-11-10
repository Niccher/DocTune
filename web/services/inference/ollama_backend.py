"""
web/services/inference/ollama_backend.py
---------------------------------------
Inference backend connecting to a local or remote Ollama daemon.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import httpx

from web.services.inference.base import InferenceBackend

logger = logging.getLogger("doctune.inference.ollama")


class OllamaBackend(InferenceBackend):
    """Inference engine connecting to Ollama daemon (port 11434)."""

    name: str = "ollama"

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = (endpoint or os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434").rstrip("/")
        self._timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.endpoint}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def health(self) -> Dict[str, Any]:
        is_up = await self.is_available()
        models = []
        if is_up:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    res = await client.get(f"{self.endpoint}/api/tags")
                    if res.status_code == 200:
                        models = [m.get("name") for m in res.json().get("models", [])]
            except Exception:
                pass
        return {
            "backend": self.name,
            "endpoint": self.endpoint,
            "running": is_up,
            "available_models": models,
        }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3:latest",
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            res = await client.post(f"{self.endpoint}/api/chat", json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"Ollama server returned {res.status_code}: {res.text}")
            data = res.json()
            # Convert Ollama format to OpenAI-compatible format
            msg = data.get("message", {})
            return {
                "id": f"ollama-{int(data.get('created_at', 0)) if isinstance(data.get('created_at'), (int, float)) else 0}",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": msg.get("role", "assistant"),
                            "content": msg.get("content", ""),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
            }

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3:latest",
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        client = httpx.AsyncClient(timeout=self._timeout)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with client.stream("POST", f"{self.endpoint}/api/chat", json=payload) as response:
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': f'Ollama error: HTTP {response.status_code}'})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        done = chunk.get("done", False)
                        sse_payload = {
                            "choices": [
                                {
                                    "delta": {"content": content},
                                    "finish_reason": "stop" if done else None,
                                }
                            ]
                        }
                        yield f"data: {json.dumps(sse_payload)}\n\n"
                        if done:
                            yield "data: [DONE]\n\n"
                            break
                    except Exception:
                        pass
        finally:
            await client.aclose()
