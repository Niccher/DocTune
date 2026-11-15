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
        self._indexing_state: Dict[str, Dict[str, Any]] = {}

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

    def get_indexing_progress(self, project_id: Optional[str] = "proj_default") -> Dict[str, Any]:
        """Retrieve real-time vector indexing progress for UI telemetry."""
        pid = project_id or "proj_default"
        state = self._indexing_state.get(pid)
        if not state:
            stats = self.get_index_status(pid)
            total = stats.get("total_chunks", 0)
            return {
                "status": "COMPLETED" if total > 0 else "IDLE",
                "progress": 100 if total > 0 else 0,
                "percentage": 100 if total > 0 else 0,
                "message": f"{total} chunks indexed in ChromaDB" if total > 0 else "Ready to build vector index",
                "current_file": None,
                "current_doc": 0,
                "total_docs": stats.get("total_documents_on_disk", 0),
                "total_chunks": total,
            }
        res = dict(state)
        res["percentage"] = res.get("progress", 0)
        return res

    def update_indexing_progress(self, project_id: Optional[str] = "proj_default", **kwargs):
        """Update the in-memory indexing telemetry state."""
        pid = project_id or "proj_default"
        cur = self._indexing_state.get(pid, {
            "status": "RUNNING",
            "progress": 0,
            "message": "",
            "current_file": None,
            "current_doc": 0,
            "total_docs": 0,
            "total_chunks": 0,
            "start_time": time.time(),
        })
        cur.update(kwargs)
        self._indexing_state[pid] = cur

    def snapshot_chunks_to_dataset(self, project_id: Optional[str] = "proj_default", slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Convert indexed ChromaDB vector chunks into an immutable fine-tuning dataset (.jsonl)
        and register it in the Dataset Version catalog and MinIO.
        """
        from pipeline.common import DATA_DIR
        from web.services.dataset_version_service import dataset_version_service
        import tempfile
        import re

        chunks = self.get_all_chunks(project_id=project_id, limit=5000)
        if not chunks:
            return None

        clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", (slug or "rag_corpus").strip()).strip("_") or "rag_corpus"
        temp_file = Path(tempfile.gettempdir()) / f"temp_{clean_slug}_{int(time.time())}.jsonl"

        written = 0
        with open(temp_file, "w", encoding="utf-8") as f:
            for c in chunks:
                txt = (c.get("text") or "").strip()
                if not txt:
                    continue
                src = c.get("source") or c.get("document") or "Document"
                record = {
                    "messages": [
                        {"role": "system", "content": "You are an internal enterprise AI assistant trained on proprietary documentation."},
                        {"role": "user", "content": f"Based on the following reference material from document '{src}', summarize the key facts and requirements:\n\n{txt}"},
                        {"role": "assistant", "content": f"Key information extracted from {src}:\n{txt}"}
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        if written == 0:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            return None

        try:
            snapshot = dataset_version_service.create_auto_version(
                source_file=temp_file,
                slug_hint=f"{clean_slug}_rag_chunks",
                description=f"Auto-generated corpus snapshot from {written} ChromaDB vector chunks",
                metadata={
                    "teacher_model": "ChromaDB-Embeddings",
                    "teacher_provider": "RAG Vector Store",
                    "files_count": len(set(c.get('source') for c in chunks if c.get('source'))),
                    "duration_sec": 0,
                },
                set_active=False,
            )
            return snapshot
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def build_index(
        self,
        project_id: Optional[str] = "proj_default",
        docs_dir: Optional[Path] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Parse all documents in docs_dir, split into chunks, compute embeddings,
        and upsert into the project's embedded ChromaDB collection.
        Automatically snapshots chunks to MinIO and registers in Dataset Version catalog.
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
            self.update_indexing_progress(
                project_id=project_id,
                status="COMPLETED",
                progress=0,
                message=f"No documents found in {target_dir}. Please upload documents first.",
            )
            return {
                "status": "warning",
                "message": f"No documents found in {target_dir}. Please upload documents first.",
                "total_documents": 0,
                "total_chunks": 0,
            }

        start_time = time.time()
        total_chunks_added = 0
        indexed_files = []

        self.update_indexing_progress(
            project_id=project_id,
            status="RUNNING",
            progress=5,
            message=f"Starting indexing for {len(all_files)} documents...",
            total_docs=len(all_files),
            current_doc=0,
            total_chunks=0,
            start_time=start_time,
        )

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

                pct = 5 + int((idx / len(all_files)) * 85)
                progress_data = {
                    "current": idx,
                    "total": len(all_files),
                    "file": file_path.name,
                    "chunks": len(chunks) if chunks else 0,
                    "total_chunks": total_chunks_added,
                    "progress": pct,
                }

                self.update_indexing_progress(
                    project_id=project_id,
                    status="RUNNING",
                    progress=pct,
                    current_file=file_path.name,
                    current_doc=idx,
                    total_docs=len(all_files),
                    total_chunks=total_chunks_added,
                    message=f"Indexed {file_path.name} ({len(chunks) if chunks else 0} chunks) [{idx}/{len(all_files)}]",
                )

                if progress_callback:
                    progress_callback(progress_data)

            except Exception as e:
                logger.error("Error indexing document %s: %s", file_path.name, e)

        duration = round(time.time() - start_time, 2)
        logger.info(
            "RAG indexing completed for project '%s': %d documents, %d chunks in %ss",
            project_id, len(all_files), total_chunks_added, duration,
        )

        # Auto-create immutable dataset snapshot from indexed vector chunks
        snapshot_res = None
        if total_chunks_added > 0:
            try:
                doc_slug = all_files[0].stem if all_files else "corpus"
                snapshot_res = self.snapshot_chunks_to_dataset(project_id=project_id, slug=doc_slug)
            except Exception as snap_ex:
                logger.warning("Failed to auto-snapshot vector chunks: %s", snap_ex)

        done_msg = f"Successfully indexed {len(all_files)} documents into {total_chunks_added} vector chunks."
        if snapshot_res:
            done_msg += f" Auto-saved snapshot: {snapshot_res.get('filename')}"

        self.update_indexing_progress(
            project_id=project_id,
            status="COMPLETED",
            progress=100,
            message=done_msg,
            current_file=None,
            current_doc=len(all_files),
            total_docs=len(all_files),
            total_chunks=total_chunks_added,
            snapshot=snapshot_res,
        )

        return {
            "status": "success",
            "project_id": project_id,
            "total_documents": len(all_files),
            "total_chunks": total_chunks_added,
            "indexed_files": indexed_files,
            "duration_sec": duration,
            "snapshot": snapshot_res,
            "message": done_msg,
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
