"""
web/services/inference_service.py
---------------------------------
Inference service delegating to the unified InferenceManager (llama.cpp default).
Provides OpenAI-compatible chat completions and streaming Server-Sent Events (SSE).
Zero hard-coded or simulated responses.
"""

import time
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator

from web.services.inference.manager import inference_manager

logger = logging.getLogger("doctune.inference_service")


class InferenceService:
    """Manages LLM completions via the active InferenceBackend."""

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Discover available checkpoints and merged models ready for inference."""
        return inference_manager.list_available_models()

    async def generate_completion_async(
        self,
        messages: List[Dict[str, str]],
        model_name: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Generate a chat completion via the active backend (llama.cpp default)."""
        try:
            return await inference_manager.chat(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("Inference generation error: %s", e)
            return {
                "id": f"chatcmpl-err-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                f"[Inference Error]: {str(e)}\n\n"
                                "To resolve: ensure llama-server is running on port 8080 "
                                "(e.g., `llama-server -m <path-to-gguf> --port 8080`) or configure OLLAMA_ENDPOINT."
                            ),
                        },
                        "finish_reason": "error",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model_name: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for generate_completion_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running in an event loop, run in a new loop or executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run,
                        self.generate_completion_async(messages, model_name, temperature, max_tokens),
                    ).result(timeout=130)
            else:
                return loop.run_until_complete(
                    self.generate_completion_async(messages, model_name, temperature, max_tokens)
                )
        except Exception as e:
            logger.error("Sync inference wrapper error: %s", e)
            return {
                "id": f"chatcmpl-err-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"[Inference Engine Unavailable]: {str(e)}",
                        },
                        "finish_reason": "error",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model_name: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        """Yield Server-Sent Events (SSE) token chunks conforming to OpenAI format."""
        try:
            async for chunk in inference_manager.stream_chat(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
        except Exception as e:
            logger.error("Streaming inference error: %s", e)
            err_payload = {
                "id": f"err-{int(time.time())}",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n[Inference Error: {str(e)}]"},
                        "finish_reason": "error",
                    }
                ],
            }
            yield f"data: {json.dumps(err_payload)}\n\n"
            yield "data: [DONE]\n\n"


inference_service = InferenceService()
