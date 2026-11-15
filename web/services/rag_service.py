"""
web/services/rag_service.py
---------------------------
Application-level service orchestrating vector index construction,
document parsing, real-time telemetry, and RAG context augmentation.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from pipeline.common import PROJECT_ROOT, RAW_DOCS_DIR
from pipeline.rag.chunker import DocumentChunker, TextChunk
from pipeline.rag.chroma_store import chroma_store
from pipeline.rag.retriever import rag_retriever
from web.db.session import SessionLocal
from web.db.models import UploadedDocument

logger = logging.getLogger("doctune.web.rag_service")


class RAGService:
    """Coordinates document vectorization, storage, and retrieval augmentation."""

    def __init__(self):
        self.chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
        self.store = chroma_store
        self.retriever = rag_retriever

    def get_index_status(self, project_id: Optional[str] = "proj_default") -> Dict[str, Any]:
        """Return status, chunk counts, and readiness for a project's vector store."""
        stats = self.store.get_stats(project_id)

        # Count raw documents available on disk or DB
        raw_files = []
        if RAW_DOCS_DIR.exists():
            for pat in ["*.pdf", "*.docx", "*.csv", "*.txt", "*.md"]:
                raw_files.extend(RAW_DOCS_DIR.glob(pat))
        doc_count = len(set(raw_files))

        stats["total_documents_on_disk"] = doc_count
        stats["is_indexed"] = stats["total_chunks"] > 0
        return stats

    def build_index(
        self,
        project_id: Optional[str] = "proj_default",
        docs_dir: Optional[Path] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Parse all documents in docs_dir, split into chunks, compute embeddings,
        and upsert into the project's embedded ChromaDB collection.
        """
        target_dir = docs_dir or RAW_DOCS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        # Sync from MinIO S3 if available
        try:
            from web.services.storage_service import storage_service
            storage_service.sync_bucket_to_dir(storage_service.bucket_docs, target_dir)
        except Exception as e:
            logger.debug("S3 sync skipped or unavailable: %s", e)

        # Discover all supported documents
        patterns = ["*.pdf", "*.docx", "*.csv", "*.txt", "*.md", "*.PDF", "*.DOCX", "*.CSV", "*.TXT", "*.MD"]
        found_files = []
        for pat in patterns:
            found_files.extend(target_dir.glob(pat))
        all_files = sorted(set(found_files))

        if not all_files:
            return {
                "status": "warning",
                "message": f"No documents found in {target_dir}. Please upload documents first.",
                "total_documents": 0,
                "total_chunks": 0,
            }

        start_time = time.time()
        total_chunks_added = 0
        indexed_files = []

        # Clear existing collection before full re-index to prevent duplicate stale chunks
        self.store.clear_collection(project_id)

        # Process each document
        for idx, file_path in enumerate(all_files, start=1):
            try:
                chunks = self.chunker.process_file(
                    file_path=file_path,
                    document_id=f"doc_{file_path.stem}",
                    extra_metadata={"project_id": project_id or "proj_default"},
                )
                if chunks:
                    added = self.store.add_chunks(chunks, project_id=project_id)
                    total_chunks_added += added
                    indexed_files.append({
                        "file_name": file_path.name,
                        "chunks_count": len(chunks),
                    })

                if progress_callback:
                    progress_callback({
                        "current": idx,
                        "total": len(all_files),
                        "file": file_path.name,
                        "chunks": len(chunks),
                    })

            except Exception as e:
                logger.error("Error indexing document %s: %s", file_path.name, e)

        duration = round(time.time() - start_time, 2)
        logger.info(
            "RAG indexing completed for project '%s': %d documents, %d chunks in %ss",
            project_id, len(all_files), total_chunks_added, duration,
        )

        return {
            "status": "success",
            "project_id": project_id,
            "total_documents": len(all_files),
            "total_chunks": total_chunks_added,
            "indexed_files": indexed_files,
            "duration_sec": duration,
            "message": f"Successfully indexed {len(all_files)} documents into {total_chunks_added} vector chunks.",
        }

    def search(
        self,
        query: str,
        project_id: Optional[str] = "proj_default",
        top_k: int = 3,
        min_score: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K matching chunks for a user query."""
        return self.retriever.retrieve(
            query=query,
            project_id=project_id,
            top_k=top_k,
            min_score=min_score,
        )

    def augment_chat_messages(
        self,
        messages: List[Dict[str, str]],
        project_id: Optional[str] = "proj_default",
        top_k: int = 3,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Augment a list of chat messages by retrieving context for the latest user prompt.
        Returns:
            - augmented_messages: messages list with context injected into system prompt
            - citations: list of structured citation objects for UI rendering
            - raw_chunks: list of retrieved chunk candidates with relevance scores
        """
        # Find the latest user message
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        if not user_query:
            return messages, [], []

        chunks = self.search(query=user_query, project_id=project_id, top_k=top_k)
        if not chunks:
            return messages, [], []

        augmented_messages, citations = self.retriever.augment_messages(messages, chunks)
        return augmented_messages, citations, chunks

    def clear_index(self, project_id: Optional[str] = "proj_default") -> bool:
        """Clear the project's vector store."""
        return self.store.clear_collection(project_id)

    def get_all_chunks(self, project_id: Optional[str] = "proj_default", limit: int = 150, offset: int = 0) -> List[Dict[str, Any]]:
        """Return all text chunks currently indexed in ChromaDB for a project."""
        return self.store.get_all_chunks(project_id=project_id, limit=limit, offset=offset)


rag_service = RAGService()
