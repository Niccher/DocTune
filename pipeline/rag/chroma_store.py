"""
pipeline/rag/chroma_store.py
----------------------------
Embedded ChromaDB vector store manager.
Runs in-process inside Web and Celery containers with zero external container dependencies.
Persists vectors and HNSW indexes locally under data/vectorstore/.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from pipeline.common import PROJECT_ROOT, DATA_DIR
from pipeline.rag.chunker import TextChunk
from pipeline.rag.embeddings import embedding_service

logger = logging.getLogger("doctune.rag.chroma")

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None
    HAS_CHROMADB = False


class ChromaVectorStore:
    """Manages persistent embedded ChromaDB collections partitioned by project."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or (DATA_DIR / "vectorstore")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._memory_fallback: Dict[str, List[Dict[str, Any]]] = {}

    def _get_client(self):
        if not HAS_CHROMADB:
            return None
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.info("Initialized embedded ChromaDB at: %s", self.persist_dir)
            except Exception as e:
                logger.warning("Failed to initialize ChromaDB PersistentClient: %s", e)
                self._client = None
        return self._client

    def _collection_name(self, project_id: Optional[str]) -> str:
        clean_id = (project_id or "default").replace("-", "_").lower()
        return f"doctune_{clean_id}"

    def get_or_create_collection(self, project_id: Optional[str] = "proj_default"):
        client = self._get_client()
        if client is None:
            return None
        name = self._collection_name(project_id)
        # Use cosine distance space
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add_chunks(
        self,
        chunks: List[TextChunk],
        project_id: Optional[str] = "proj_default",
    ) -> int:
        """Embed and upsert text chunks into the project's vector collection."""
        if not chunks:
            return 0

        client = self._get_client()
        texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_texts(texts)
        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Sanitize metadata values to types supported by Chroma (str, int, float, bool)
        sanitized_metas = []
        for m in metadatas:
            clean_m = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                elif v is None:
                    clean_m[k] = ""
                else:
                    clean_m[k] = str(v)
            sanitized_metas.append(clean_m)

        if client is not None:
            collection = self.get_or_create_collection(project_id)
            # Upsert chunks in batches of 100 to prevent buffer limits
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                collection.upsert(
                    ids=ids[i : i + batch_size],
                    documents=texts[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size],
                    metadatas=sanitized_metas[i : i + batch_size],
                )
        else:
            # Fallback memory store if chromadb binary is not loaded
            coll_name = self._collection_name(project_id)
            if coll_name not in self._memory_fallback:
                self._memory_fallback[coll_name] = []
            for cid, txt, emb, meta in zip(ids, texts, embeddings, sanitized_metas):
                self._memory_fallback[coll_name].append({
                    "id": cid,
                    "document": txt,
                    "embedding": emb,
                    "metadata": meta,
                })

        return len(chunks)

    def query_similar(
        self,
        query_text: str,
        project_id: Optional[str] = "proj_default",
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-K semantically closest chunks to query_text."""
        if not query_text or not query_text.strip():
            return []

        query_emb = embedding_service.embed_query(query_text)
        client = self._get_client()

        if client is not None:
            collection = self.get_or_create_collection(project_id)
            count = collection.count()
            if count == 0:
                return []

            actual_k = min(top_k, count)
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=actual_k,
                where=where if where else None,
            )

            hits = []
            if results and results.get("ids") and results["ids"][0]:
                res_ids = results["ids"][0]
                res_docs = results.get("documents", [[]])[0]
                res_metas = results.get("metadatas", [[]])[0]
                res_dists = results.get("distances", [[]])[0]

                for i, cid in enumerate(res_ids):
                    dist = res_dists[i] if i < len(res_dists) else 0.5
                    # In cosine space, similarity = 1 - distance
                    sim = max(0.0, min(1.0, 1.0 - dist))
                    hits.append({
                        "id": cid,
                        "text": res_docs[i] if i < len(res_docs) else "",
                        "metadata": res_metas[i] if i < len(res_metas) else {},
                        "distance": round(dist, 4),
                        "similarity": round(sim, 4),
                    })
            return hits

        # Fallback query calculation (exact cosine dot-product)
        coll_name = self._collection_name(project_id)
        records = self._memory_fallback.get(coll_name, [])
        if not records:
            return []

        scored = []
        for r in records:
            emb = r["embedding"]
            # Cosine similarity for normalized vectors is simply dot product
            sim = sum(a * b for a, b in zip(query_emb, emb))
            scored.append((sim, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for sim, r in scored[:top_k]:
            hits.append({
                "id": r["id"],
                "text": r["document"],
                "metadata": r["metadata"],
                "distance": round(1.0 - sim, 4),
                "similarity": round(max(0.0, min(1.0, sim)), 4),
            })
        return hits

    def delete_document(self, document_name: str, project_id: Optional[str] = "proj_default") -> int:
        """Remove all indexed chunks belonging to a given document."""
        client = self._get_client()
        if client is not None:
            collection = self.get_or_create_collection(project_id)
            try:
                collection.delete(where={"document_name": document_name})
                return 1
            except Exception as e:
                logger.warning("Failed to delete document chunks from Chroma: %s", e)
                return 0
        else:
            coll_name = self._collection_name(project_id)
            if coll_name in self._memory_fallback:
                before = len(self._memory_fallback[coll_name])
                self._memory_fallback[coll_name] = [
                    r for r in self._memory_fallback[coll_name]
                    if r["metadata"].get("document_name") != document_name
                ]
                return before - len(self._memory_fallback[coll_name])
        return 0

    def clear_collection(self, project_id: Optional[str] = "proj_default") -> bool:
        """Drop and reset the collection for a project."""
        client = self._get_client()
        coll_name = self._collection_name(project_id)
        if client is not None:
            try:
                client.delete_collection(name=coll_name)
                return True
            except Exception:
                return False
        else:
            self._memory_fallback.pop(coll_name, None)
            return True

    def get_stats(self, project_id: Optional[str] = "proj_default") -> Dict[str, Any]:
        """Return total chunks, collection name, and persistence info."""
        client = self._get_client()
        coll_name = self._collection_name(project_id)
        count = 0
        is_ready = False

        if client is not None:
            try:
                coll = self.get_or_create_collection(project_id)
                count = coll.count()
                is_ready = True
            except Exception as e:
                logger.debug("Chroma get_stats count notice: %s", e)
        else:
            count = len(self._memory_fallback.get(coll_name, []))
            is_ready = count > 0

        return {
            "collection_name": coll_name,
            "project_id": project_id or "proj_default",
            "total_chunks": count,
            "persist_dir": str(self.persist_dir),
            "backend": "chromadb_embedded" if client is not None else "memory_fallback",
            "ready": is_ready and count > 0,
        }


chroma_store = ChromaVectorStore()
