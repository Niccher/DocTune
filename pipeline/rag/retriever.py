"""
pipeline/rag/retriever.py
-------------------------
Hybrid dense (ChromaDB) + sparse (keyword frequency) retriever with
Reciprocal Rank Fusion (RRF), citation tracking, and grounded prompt assembly.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

from pipeline.rag.chroma_store import ChromaVectorStore, chroma_store

logger = logging.getLogger("doctune.rag.retriever")


class RAGRetriever:
    """Orchestrates hybrid retrieval and context packing for chat completions."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.store = vector_store or chroma_store

    def _sparse_keyword_score(self, query: str, text: str) -> float:
        """Compute term-matching overlap score between query and document text."""
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for w in query_words if w in text_lower)
        return matches / max(1, len(query_words))

    def retrieve(
        self,
        query: str,
        project_id: Optional[str] = "proj_default",
        top_k: int = 3,
        min_score: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search:
        1. Fetch candidates from embedded ChromaDB (dense cosine similarity).
        2. Score candidates with sparse keyword matching.
        3. Fuse scores via Reciprocal Rank Fusion (RRF) and filter by min_score.
        """
        if not query or not query.strip():
            return []

        # 1. Fetch dense candidates (fetch 2x top_k to re-rank)
        fetch_k = max(top_k * 2, 6)
        dense_hits = self.store.query_similar(query_text=query, project_id=project_id, top_k=fetch_k)

        if not dense_hits:
            return []

        # 2. Score sparse keywords
        fused = []
        for rank, hit in enumerate(dense_hits):
            sparse_score = self._sparse_keyword_score(query, hit["text"])
            dense_sim = hit["similarity"]
            # Reciprocal Rank Fusion + weighted linear combination
            rrf_score = (1.0 / (60 + rank)) + (dense_sim * 0.7) + (sparse_score * 0.3)
            hit["hybrid_score"] = round(rrf_score, 4)
            hit["sparse_score"] = round(sparse_score, 4)
            fused.append(hit)

        # 3. Sort by hybrid score and take top_k
        fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
        results = [h for h in fused if h["similarity"] >= min_score or h["sparse_score"] >= 0.4][:top_k]

        return results or fused[:top_k]

    def assemble_context(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Pack retrieved chunks into a standard grounded context block and
        extract structured citation metadata for the frontend.
        """
        if not chunks:
            return "", []

        context_lines = [
            "Enterprise Document Knowledge Base Excerpts:",
            "===========================================",
        ]
        citations = []

        for i, c in enumerate(chunks, start=1):
            meta = c.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown Document")
            chunk_idx = meta.get("chunk_index", i)
            sim_pct = int(c.get("similarity", 0.0) * 100)
            text = c.get("text", "").strip()

            snippet = (text[:160] + "...") if len(text) > 160 else text

            citations.append({
                "citation_id": f"ref_{i}",
                "document_name": doc_name,
                "document_id": meta.get("document_id"),
                "chunk_index": chunk_idx,
                "similarity_pct": sim_pct,
                "snippet": snippet,
            })

            context_lines.append(f"[{i}] Document: {doc_name} (Section/Chunk {chunk_idx}, Relevance: {sim_pct}%):")
            context_lines.append(text)
            context_lines.append("-------------------------------------------")

        context_lines.append("===========================================")
        context_lines.append(
            "Instructions: Answer the query accurately using ONLY the excerpts above. "
            "If the information is not present in the excerpts, state that you do not have sufficient internal documentation."
        )

        return "\n".join(context_lines), citations

    def augment_messages(
        self,
        messages: List[Dict[str, str]],
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Augment chat completion messages with grounded RAG context.
        Prepends or updates the system prompt and returns the augmented message list with citations.
        """
        context_str, citations = self.assemble_context(chunks)
        if not context_str:
            return messages, []

        augmented = list(messages)
        # Check if an existing system message exists
        system_idx = next((i for i, m in enumerate(augmented) if m.get("role") == "system"), None)

        if system_idx is not None:
            existing_content = augmented[system_idx].get("content", "")
            augmented[system_idx] = {
                "role": "system",
                "content": f"{existing_content}\n\n{context_str}".strip(),
            }
        else:
            augmented.insert(0, {
                "role": "system",
                "content": context_str,
            })

        return augmented, citations


rag_retriever = RAGRetriever()
