"""
web/services/inference/base.py
------------------------------
Abstract base class for LLM inference engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator


class InferenceBackend(ABC):
    """Abstract interface for local and containerized LLM serving engines."""

    name: str = "base"

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the backend engine is running, healthy, and reachable."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete chat completion."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        adapter_path: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks (text delta)."""
        pass

    @abstractmethod
    async def health(self) -> Dict[str, Any]:
        """Return backend status, device placement, and engine telemetry."""
        pass
