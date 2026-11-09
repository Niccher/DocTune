"""
web/services/teacher_service.py
--------------------------------
Universal Teacher LLM Gateway Service.

Harmonizes interactions across local self-hosted engines (Ollama, llama.cpp, vLLM)
and commercial cloud foundation models (Google Gemini, xAI Grok, OpenAI) via
the standard OpenAI-compatible HTTP REST protocol.
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urljoin
import httpx

logger = logging.getLogger(__name__)

# In-memory registry of active model pull/download tasks: keyed by f"{provider}:{model.strip()}"
_pull_tasks: Dict[str, Dict[str, Any]] = {}

# Standard Provider Presets
PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "id": "ollama",
        "name": "Ollama (Local / On-Premise)",
        "base_url": "http://localhost:11434/v1",
        "docker_base_url": "http://host.docker.internal:11434/v1",
        "default_model": "llama3.1:latest",
        "suggested_models": [
            "llama3.1:latest",
            "llama3:latest",
            "mistral:latest",
            "qwen2.5:7b",
            "qwen2.5:14b",
            "gemma2:9b",
            "phi3:latest",
        ],
        "auth_required": False,
        "default_api_key": "not-needed",
        "hint": "Local private inference via Ollama on port 11434. 100% confidential and free.",
    },
    "llamacpp": {
        "id": "llamacpp",
        "name": "llama.cpp Server (llama-server)",
        "base_url": "http://localhost:8080/v1",
        "docker_base_url": "http://host.docker.internal:8080/v1",
        "default_model": "default",
        "suggested_models": ["default"],
        "auth_required": False,
        "default_api_key": "not-needed",
        "hint": "High-efficiency GGUF execution via llama-server on port 8080. Zero cost.",
    },
    "vllm": {
        "id": "vllm",
        "name": "vLLM Engine",
        "base_url": "http://localhost:8000/v1",
        "docker_base_url": "http://host.docker.internal:8000/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "suggested_models": [
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "auth_required": False,
        "default_api_key": "not-needed",
        "hint": "High-throughput vLLM serving engine. Native OpenAI compatibility.",
    },
    "gemini": {
        "id": "gemini",
        "name": "Google Gemini (Google AI Studio)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "docker_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "suggested_models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "auth_required": True,
        "default_api_key": "",
        "hint": "Official Google AI Studio API via OpenAI compatibility. Requires Gemini API Key.",
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek (DeepSeek API)",
        "base_url": "https://api.deepseek.com/v1",
        "docker_base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "suggested_models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "auth_required": True,
        "default_api_key": "",
        "hint": "High-intelligence reasoning & coding models. Requires DeepSeek API Key (sk-...).",
    },
    "grok": {
        "id": "grok",
        "name": "xAI Grok",
        "base_url": "https://api.x.ai/v1",
        "docker_base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-latest",
        "suggested_models": [
            "grok-2-latest",
            "grok-2-1212",
            "grok-beta",
            "grok-vision-beta",
        ],
        "auth_required": True,
        "default_api_key": "",
        "hint": "xAI foundation models. Requires xAI API Key (xai-...).",
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI API",
        "base_url": "https://api.openai.com/v1",
        "docker_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "suggested_models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "o3-mini",
        ],
        "auth_required": True,
        "default_api_key": "",
        "hint": "OpenAI foundation models. Requires OpenAI API Key (sk-...).",
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic Claude (Anthropic API)",
        "base_url": "https://api.anthropic.com/v1",
        "docker_base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-5-sonnet-latest",
        "suggested_models": [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
        ],
        "auth_required": True,
        "default_api_key": "",
        "hint": "Anthropic Claude models. Requires Anthropic API Key (sk-ant-...).",
    },
    "custom": {
        "id": "custom",
        "name": "Custom OpenAI-Compatible Endpoint",
        "base_url": "http://localhost:8000/v1",
        "docker_base_url": "http://host.docker.internal:8000/v1",
        "default_model": "",
        "suggested_models": [],
        "auth_required": False,
        "default_api_key": "",
        "hint": "Any OpenAI-compatible server (LocalAI, TGI, LM Studio, etc.).",
    },
}


class TeacherService:
    """Manages Teacher LLM presets, health probing, model listing, and connection verification."""

    @staticmethod
    def is_running_in_docker() -> bool:
        """Detect if the current Python process is executing within a Docker container."""
        return (
            os.path.exists("/.dockerenv")
            or os.environ.get("DOCKER_CONTAINER", "").lower() in ("1", "true", "yes")
            or os.environ.get("IS_DOCKER", "").lower() in ("1", "true", "yes")
        )

    @classmethod
    def get_presets(cls) -> Dict[str, Dict[str, Any]]:
        """Return all provider presets, dynamically tailoring default URLs for host vs container."""
        in_docker = cls.is_running_in_docker()
        presets = {}
        for key, preset in PROVIDER_PRESETS.items():
            item = dict(preset)
            if in_docker and "docker_base_url" in item:
                item["effective_base_url"] = item["docker_base_url"]
            else:
                item["effective_base_url"] = item["base_url"]
            presets[key] = item
        return presets

    @classmethod
    def resolve_url(cls, url: str) -> str:
        """
        If running inside Docker and target is localhost/127.0.0.1,
        remap to host.docker.internal so the container can reach host services.
        """
        trimmed = url.strip()
        if cls.is_running_in_docker():
            if "localhost" in trimmed:
                return trimmed.replace("localhost", "host.docker.internal")
            if "127.0.0.1" in trimmed:
                return trimmed.replace("127.0.0.1", "host.docker.internal")
        return trimmed

    @classmethod
    def normalize_base_url(cls, provider: str, url: str) -> str:
        """
        Normalize and resolve Base URL:
        1. Resolve Docker host bridge if in container.
        2. Ensure standard /v1 or /openai path suffix based on provider.
        """
        trimmed = cls.resolve_url(url.strip().rstrip("/"))
        if not trimmed:
            return trimmed

        # Standard OpenAI-compatible runners and Anthropic require /v1
        if provider in ("ollama", "llamacpp", "vllm", "openai", "grok", "deepseek", "anthropic"):
            if not trimmed.endswith("/v1"):
                trimmed = f"{trimmed}/v1"
        elif provider == "gemini":
            if not trimmed.endswith("/openai"):
                if trimmed.endswith("/v1beta"):
                    trimmed = f"{trimmed}/openai"
                elif not trimmed.endswith("/openai/"):
                    trimmed = f"{trimmed}/v1beta/openai"
        return trimmed

    @staticmethod
    def should_verify_ssl(url: str) -> bool:
        """
        Validate whether SSL certificates should be verified for the given URL.
        Allows unverified TLS only for local loopback or explicit local/internal endpoints.
        Enforces TLS verification for public cloud providers to prevent MitM attacks.
        """
        if not url:
            return True
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme == "http":
                return False
            host = (parsed.hostname or "").lower()
            if host in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"):
                return False
            if host.endswith(".local") or host.endswith(".internal"):
                return False
            return True
        except Exception:
            return True

    @classmethod
    async def fetch_available_models(
        cls,
        provider: str,
        base_url: str,
        api_key: str = "",
        timeout_sec: float = 6.0,
    ) -> List[str]:
        """
        Probe endpoint to discover installed/available models dynamically.
        Supports Ollama (/api/tags and /v1/models), Anthropic (/v1/models), and standard OpenAI (/v1/models).
        Sanitizes Google Gemini model prefixes and filters non-conversational models.
        """
        clean_url = cls.normalize_base_url(provider, base_url)
        headers = {"Accept": "application/json"}
        if api_key and api_key.strip() and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        discovered: List[str] = []

        async with httpx.AsyncClient(timeout=timeout_sec, verify=cls.should_verify_ssl(clean_url)) as client:
            # 1. If provider is Ollama, probe native /api/tags endpoint first
            if provider == "ollama":
                # Strip trailing /v1 to reach root port
                root_url = clean_url[:-3] if clean_url.endswith("/v1") else clean_url
                tags_url = f"{root_url}/api/tags"
                try:
                    res = await client.get(tags_url, headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        tags = data.get("models", [])
                        for item in tags:
                            name = item.get("name") or item.get("model")
                            if name:
                                discovered.append(name)
                        if discovered:
                            return sorted(discovered)
                except Exception as e:
                    logger.debug("Ollama /api/tags probe failed: %s", e)

            # 2. If provider is Anthropic, probe native /v1/models endpoint with x-api-key
            elif provider == "anthropic":
                anthropic_headers = {
                    "x-api-key": api_key.strip() if api_key and api_key != "not-needed" else "",
                    "anthropic-version": "2023-06-01",
                    "Accept": "application/json",
                }
                models_url = f"{clean_url}/models" if not clean_url.endswith("/models") else clean_url
                try:
                    res = await client.get(models_url, headers=anthropic_headers)
                    if res.status_code == 200:
                        data = res.json()
                        raw_list = data.get("data", [])
                        for item in raw_list:
                            m_id = item.get("id") if isinstance(item, dict) else str(item)
                            if m_id:
                                discovered.append(m_id)
                except Exception as e:
                    logger.debug("Anthropic /models probe failed: %s", e)

            # 3. Standard OpenAI /models endpoint (DeepSeek, Grok, Gemini, vLLM, llama.cpp, OpenAI, Custom)
            else:
                models_url = f"{clean_url}/models" if not clean_url.endswith("/models") else clean_url
                try:
                    res = await client.get(models_url, headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        raw_list = data.get("data", [])
                        for item in raw_list:
                            m_id = item.get("id") if isinstance(item, dict) else str(item)
                            if m_id:
                                # Strip Google Gemini models/ prefix if present
                                if provider == "gemini" and m_id.startswith("models/"):
                                    m_id = m_id[7:]
                                discovered.append(m_id)
                except Exception as e:
                    logger.debug("OpenAI /models probe failed: %s", e)

        # Filter non-generative / embedding models for Gemini
        if provider == "gemini" and discovered:
            filtered = []
            non_chat_keywords = ("embedding", "imagen", "aqa", "learnlm", "bison", "whisper")
            for m in discovered:
                if not any(k in m.lower() for k in non_chat_keywords):
                    filtered.append(m)
            if filtered:
                discovered = filtered

        # Fallback to suggested preset models if discovery did not return results
        if not discovered:
            preset = PROVIDER_PRESETS.get(provider, {})
            discovered = preset.get("suggested_models", [])

        # Deduplicate while preserving sensible order (newer versions prioritized)
        def model_sort_key(name: str):
            if provider == "gemini":
                if "2.5" in name:
                    return (0, name)
                if "2.0" in name:
                    return (1, name)
                if "1.5" in name:
                    return (2, name)
                return (3, name)
            elif provider == "anthropic":
                n_lower = name.lower()
                if "sonnet" in n_lower:
                    return (0, name)
                if "haiku" in n_lower:
                    return (1, name)
                if "opus" in n_lower:
                    return (2, name)
                return (3, name)
            return (0, name)

        unique_models = list(dict.fromkeys(discovered))
        if provider in ("gemini", "anthropic"):
            unique_models.sort(key=model_sort_key)
        else:
            unique_models.sort()

        return unique_models

    @classmethod
    async def test_connection(
        cls,
        provider: str,
        base_url: str,
        api_key: str = "",
        model: str = "",
        timeout_sec: float = 8.0,
    ) -> Dict[str, Any]:
        """
        Verify live connectivity, calculate latency, check authorization,
        and validate model readiness for the target Teacher LLM.
        """
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
        clean_url = cls.normalize_base_url(provider, base_url)

        if not clean_url.startswith(("http://", "https://")):
            return {
                "valid": False,
                "latency_ms": 0,
                "models": [],
                "error": "Invalid Base URL: Must start with http:// or https://",
            }

        # Validate authentication requirement
        if preset.get("auth_required") and not (api_key and api_key.strip()):
            return {
                "valid": False,
                "latency_ms": 0,
                "models": [],
                "error": f"API Key is required for provider '{preset['name']}'.",
            }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key and api_key.strip() and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=timeout_sec, verify=cls.should_verify_ssl(clean_url)) as client:
                # 1. Fetch available models
                models = await cls.fetch_available_models(
                    provider=provider,
                    base_url=clean_url,
                    api_key=api_key,
                    timeout_sec=timeout_sec,
                )

                # 2. Test chat completion / message ping
                target_model = model.strip() if model and model.strip() else preset.get("default_model")
                if not target_model and models:
                    target_model = models[0]
                if not target_model:
                    target_model = "default"

                if provider == "anthropic":
                    chat_url = f"{clean_url}/messages"
                    anthropic_headers = {
                        "x-api-key": api_key.strip() if api_key and api_key != "not-needed" else "",
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                    ping_payload = {
                        "model": target_model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 2,
                    }
                    chat_res = await client.post(chat_url, json=ping_payload, headers=anthropic_headers)
                else:
                    chat_url = f"{clean_url}/chat/completions"
                    ping_payload = {
                        "model": target_model,
                        "messages": [
                            {"role": "user", "content": "ping"}
                        ],
                        "max_tokens": 2,
                        "temperature": 0.0,
                    }
                    chat_res = await client.post(chat_url, json=ping_payload, headers=headers)

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                if chat_res.status_code == 200:
                    return {
                        "valid": True,
                        "latency_ms": latency_ms,
                        "models": models,
                        "active_model": target_model,
                        "message": (
                            f"Connection verified successfully! ({preset['name']}, "
                            f"Model: {target_model}, Latency: {latency_ms}ms)"
                        ),
                    }
                elif chat_res.status_code in (401, 403):
                    return {
                        "valid": False,
                        "latency_ms": latency_ms,
                        "models": models,
                        "error": f"Authentication failed (HTTP {chat_res.status_code}). Please verify your API Key.",
                    }
                elif chat_res.status_code == 404:
                    # Some endpoints allow model query but not chat completion with the specific model ID
                    if models:
                        return {
                            "valid": True,
                            "latency_ms": latency_ms,
                            "models": models,
                            "active_model": target_model,
                            "message": (
                                f"Connected to server ({latency_ms}ms), but model '{target_model}' "
                                f"returned 404. Choose from discovered models."
                            ),
                        }
                    return {
                        "valid": False,
                        "latency_ms": latency_ms,
                        "models": [],
                        "error": f"Endpoint not found (HTTP 404) at {chat_url}. Check Base URL.",
                    }
                else:
                    err_msg = chat_res.text[:200]
                    try:
                        err_json = chat_res.json()
                        err_msg = err_json.get("error", {}).get("message") or err_msg
                    except Exception:
                        pass
                    return {
                        "valid": False,
                        "latency_ms": latency_ms,
                        "models": models,
                        "error": f"Server returned HTTP {chat_res.status_code}: {err_msg}",
                    }

        except httpx.TimeoutException:
            return {
                "valid": False,
                "latency_ms": 0,
                "models": [],
                "error": f"Connection timed out ({timeout_sec}s) reaching {clean_url}. Ensure the service is active.",
            }

        except httpx.ConnectError as e:
            # If in Docker and connecting to host.docker.internal failed for Ollama, probe internal container http://ollama:11434/v1
            if provider == "ollama" and "host.docker.internal" in clean_url and cls.is_running_in_docker():
                try:
                    alt_url = "http://ollama:11434/v1"
                    async with httpx.AsyncClient(timeout=2.0) as alt_client:
                        alt_res = await alt_client.get(f"{alt_url}/models")
                        if alt_res.status_code == 200:
                            alt_models = await cls.fetch_available_models("ollama", alt_url, timeout_sec=3.0)
                            return {
                                "valid": True,
                                "latency_ms": int((time.perf_counter() - start_time) * 1000),
                                "models": alt_models,
                                "message": "Connected to Docker Ollama container (http://ollama:11434/v1)! Update Base URL to http://ollama:11434/v1.",
                            }
                except Exception:
                    pass

            hint = ""
            if provider == "ollama":
                hint = (
                    " Neither host Ollama nor Docker Ollama container was reachable. "
                    "Options: 1) Run in Docker: 'docker compose --profile ollama up -d' (URL: http://ollama:11434/v1), "
                    "or 2) Run on host: 'OLLAMA_HOST=0.0.0.0:11434 ollama serve &', "
                    "or 3) Switch provider to 'Google Gemini' (free & requires zero host RAM)."
                )
            elif provider == "llamacpp":
                hint = (
                    " llama-server is not running. "
                    "Run on host: './llama-server -m models/<model>.gguf --host 0.0.0.0 --port 8080', "
                    "or switch to Google Gemini (free API, zero host RAM)."
                )
            elif provider == "vllm":
                hint = " Run on host: 'vllm serve <model> --host 0.0.0.0 --port 8000'."

            return {
                "valid": False,
                "latency_ms": 0,
                "models": [],
                "error": f"Connection refused at {clean_url}.{hint} (Detail: {e})",
            }
        except Exception as e:
            logger.error("Teacher connection test error: %s", e)
            return {
                "valid": False,
                "latency_ms": 0,
                "models": [],
                "error": f"Connection test failed: {str(e)}",
            }

    @classmethod
    def get_task_key(cls, provider: str, model: str) -> str:
        return f"{provider.lower()}:{model.strip().lower()}"

    @classmethod
    def get_pull_status(cls, provider: str, model: str) -> Dict[str, Any]:
        """Return current status of an ongoing or completed model pull."""
        task_key = cls.get_task_key(provider, model)
        return _pull_tasks.get(
            task_key,
            {"status": "IDLE", "progress": 0, "detail": "No active pull task.", "model": model},
        )

    @classmethod
    def start_pull_task(
        cls,
        provider: str,
        base_url: str,
        model: str,
        api_key: str = "",
    ) -> Dict[str, Any]:
        """Dispatch asynchronous background download/pull of a model."""
        clean_model = model.strip()
        if not clean_model:
            return {"status": "error", "message": "Model name cannot be empty."}

        task_key = cls.get_task_key(provider, clean_model)
        current = _pull_tasks.get(task_key, {})
        if current.get("status") == "DOWNLOADING":
            return {"status": "started", "message": f"Model {clean_model} is already downloading.", "progress": current.get("progress", 0)}

        _pull_tasks[task_key] = {
            "status": "STARTING",
            "progress": 5,
            "detail": f"Initializing download for {clean_model}...",
            "model": clean_model,
        }

        # Launch background task
        asyncio.create_task(
            cls._background_pull(task_key, provider, base_url, clean_model, api_key)
        )
        return {"status": "started", "model": clean_model, "provider": provider}

    @classmethod
    async def _background_pull(
        cls,
        task_key: str,
        provider: str,
        base_url: str,
        model: str,
        api_key: str = "",
    ):
        """Execute model download in background with real-time status updates."""
        try:
            clean_url = cls.normalize_base_url(provider, base_url)

            # 1. Ollama Native Streaming Pull API (/api/pull)
            if provider == "ollama":
                root_url = clean_url[:-3] if clean_url.endswith("/v1") else clean_url
                pull_url = f"{root_url}/api/pull"
                payload = {"name": model, "stream": True}
                headers = {"Accept": "application/json"}
                if api_key and api_key.strip() and api_key != "not-needed":
                    headers["Authorization"] = f"Bearer {api_key.strip()}"

                _pull_tasks[task_key] = {
                    "status": "DOWNLOADING",
                    "progress": 10,
                    "detail": f"Connecting to Ollama at {pull_url}...",
                    "model": model,
                }

                async with httpx.AsyncClient(timeout=3600.0) as client:
                    async with client.stream("POST", pull_url, json=payload, headers=headers) as response:
                        if response.status_code != 200:
                            err_bytes = await response.aread()
                            err_msg = err_bytes.decode("utf-8", errors="ignore")
                            _pull_tasks[task_key] = {
                                "status": "FAILED",
                                "progress": 0,
                                "error": f"Ollama returned HTTP {response.status_code}: {err_msg}",
                                "model": model,
                            }
                            return

                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if "error" in data:
                                    _pull_tasks[task_key] = {
                                        "status": "FAILED",
                                        "progress": 0,
                                        "error": f"Ollama error: {data['error']}",
                                        "model": model,
                                    }
                                    return

                                status_str = data.get("status", "")
                                completed = data.get("completed", 0)
                                total = data.get("total", 0)
                                prog = int((completed / total) * 100) if total > 0 else 0

                                if status_str == "success":
                                    _pull_tasks[task_key] = {
                                        "status": "COMPLETED",
                                        "progress": 100,
                                        "detail": f"Model '{model}' downloaded and ready in Ollama!",
                                        "model": model,
                                    }
                                    return
                                else:
                                    _pull_tasks[task_key] = {
                                        "status": "DOWNLOADING",
                                        "progress": prog if prog > 0 else 20,
                                        "detail": f"{status_str} ({prog}%)" if prog > 0 else status_str,
                                        "model": model,
                                    }
                            except Exception:
                                pass

                # If stream terminated normally
                curr = _pull_tasks.get(task_key, {})
                if curr.get("status") != "FAILED":
                    _pull_tasks[task_key] = {
                        "status": "COMPLETED",
                        "progress": 100,
                        "detail": f"Model '{model}' successfully pulled!",
                        "model": model,
                    }
                return

            # 2. llama.cpp / GGUF model download via Hugging Face Hub
            elif provider == "llamacpp":
                from pipeline.common import PROJECT_ROOT
                models_dir = PROJECT_ROOT / "models"
                models_dir.mkdir(parents=True, exist_ok=True)

                _pull_tasks[task_key] = {
                    "status": "DOWNLOADING",
                    "progress": 30,
                    "detail": f"Downloading GGUF model '{model}' from Hugging Face Hub...",
                    "model": model,
                }

                try:
                    from huggingface_hub import hf_hub_download, snapshot_download

                    def _download_hf():
                        parts = model.split("/")
                        if len(parts) >= 2 and parts[-1].endswith(".gguf"):
                            repo = "/".join(parts[:-1])
                            fname = parts[-1]
                            hf_hub_download(repo_id=repo, filename=fname, local_dir=str(models_dir))
                        else:
                            snapshot_download(repo_id=model, local_dir=str(models_dir / parts[-1]), allow_patterns="*.gguf")

                    await asyncio.to_thread(_download_hf)

                    _pull_tasks[task_key] = {
                        "status": "COMPLETED",
                        "progress": 100,
                        "detail": f"GGUF model saved to models/ directory for llama-server!",
                        "model": model,
                    }
                except Exception as hf_err:
                    _pull_tasks[task_key] = {
                        "status": "FAILED",
                        "progress": 0,
                        "error": f"Hugging Face GGUF download error: {str(hf_err)}",
                        "model": model,
                    }
                return

            # 3. Cloud / other providers
            else:
                _pull_tasks[task_key] = {
                    "status": "COMPLETED",
                    "progress": 100,
                    "detail": f"Cloud provider '{provider}' models are hosted remotely and ready to use.",
                    "model": model,
                }

        except httpx.ConnectError as ce:
            _pull_tasks[task_key] = {
                "status": "FAILED",
                "progress": 0,
                "error": f"Connection refused connecting to {provider} at {clean_url}. Ensure runner is active with OLLAMA_HOST=0.0.0.0:11434. ({ce})",
                "model": model,
            }
        except Exception as ex:
            logger.error("Model pull error for %s: %s", model, ex)
            _pull_tasks[task_key] = {
                "status": "FAILED",
                "progress": 0,
                "error": f"Pull failed: {str(ex)}",
                "model": model,
            }


# Singleton instance
teacher_service = TeacherService()
