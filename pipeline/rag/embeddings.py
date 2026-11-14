"""
pipeline/rag/embeddings.py
--------------------------
Modular embedding generator supporting local sentence-transformers models
(BAAI/bge-small-en-v1.5, all-MiniLM-L6-v2), external gateway endpoints
(Ollama/OpenAI /v1/embeddings), and resilient dense fallback.
"""

import os
import math
import hashlib
import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger("doctune.rag.embeddings")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


class EmbeddingService:
    """Provides normalized dense vector embeddings for RAG indexing and search."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
        gateway_url: Optional[str] = None,
        gateway_api_key: Optional[str] = None,
    ):
        self.model_name = os.getenv("RAG_EMBEDDING_MODEL", model_name)
        self.dimensions = dimensions
        self.gateway_url = gateway_url or os.getenv("RAG_EMBEDDING_GATEWAY_URL")
        self.gateway_api_key = gateway_api_key or os.getenv("RAG_EMBEDDING_API_KEY", "not-needed")
        self._local_model = None

    def _get_local_model(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            return None
        if self._local_model is None:
            try:
                logger.info("Loading sentence-transformers embedding model: %s", self.model_name)
                self._local_model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning("Could not load sentence-transformers model '%s': %s", self.model_name, e)
                self._local_model = None
        return self._local_model

    def _fallback_embedding(self, text: str) -> List[float]:
        """
        Deterministic, zero-external-dependency normalized dense vector fallback (384-dim).
        Used when sentence-transformers weights are downloading or offline in constrained environments.
        """
        vec = [0.0] * self.dimensions
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Hash word into bucket indices with pseudo-random projection
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimensions
            val = ((h >> 4) % 1000) / 1000.0 - 0.5
            vec[idx] += val

        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for a batch of strings."""
        if not texts:
            return []

        # 1. Try local sentence-transformers
        model = self._get_local_model()
        if model is not None:
            try:
                embeddings = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                if isinstance(embeddings, np.ndarray):
                    return embeddings.tolist()
                return [list(e) for e in embeddings]
            except Exception as e:
                logger.warning("SentenceTransformer encode error: %s. Using fallback.", e)

        # 2. Try external gateway if configured
        if self.gateway_url:
            try:
                import httpx
                headers = {"Content-Type": "application/json"}
                if self.gateway_api_key and self.gateway_api_key != "not-needed":
                    headers["Authorization"] = f"Bearer {self.gateway_api_key}"

                endpoint = self.gateway_url.rstrip("/")
                if not endpoint.endswith("/embeddings"):
                    endpoint = f"{endpoint}/embeddings"

                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        endpoint,
                        json={"model": self.model_name, "input": texts},
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_data = data.get("data", [])
                        raw_data.sort(key=lambda x: x.get("index", 0))
                        return [item["embedding"] for item in raw_data]
            except Exception as e:
                logger.warning("Gateway embeddings error: %s. Using fallback.", e)

        # 3. Deterministic dense fallback
        return [self._fallback_embedding(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * self.dimensions


embedding_service = EmbeddingService()
