"""
pipeline/rag/chunker.py
-----------------------
Semantic, sliding-window document chunker preserving structural context,
document boundaries, and source metadata for vector indexing.
"""

import os
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import polars as pl
except ImportError:
    pl = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


@dataclass
class TextChunk:
    id: str
    text: str
    document_name: str
    document_id: Optional[str]
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentChunker:
    """Extracts and chunks enterprise documents with sliding-window overlap."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_length: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def extract_text(self, file_path: Path) -> str:
        """Extract clean text from PDF, DOCX, CSV, TXT, and Markdown files."""
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                if PdfReader is None:
                    return ""
                reader = PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)

            elif ext == ".docx":
                if Document is None:
                    return ""
                doc = Document(str(file_path))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n\n".join(paragraphs)

            elif ext == ".csv":
                if pl is not None:
                    df = pl.read_csv(str(file_path))
                    rows = [str(row) for row in df.iter_rows(named=True)]
                    return "\n".join(rows)
                else:
                    return file_path.read_text(encoding="utf-8", errors="ignore")

            elif ext in (".txt", ".md", ".json", ".log"):
                return file_path.read_text(encoding="utf-8", errors="ignore")

        except Exception as e:
            print(f"[WARN] Failed to read {file_path.name}: {e}")

        return ""

    def chunk_text(
        self,
        text: str,
        document_name: str,
        document_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split text into overlapping character/word chunks with metadata."""
        if not text or not text.strip():
            return []

        # Normalize linebreaks and whitespace
        cleaned_text = re.sub(r"\r\n|\r", "\n", text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

        chunks: List[TextChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        text_len = len(cleaned_text)

        chunk_index = 0
        for start in range(0, text_len, step):
            end = min(text_len, start + self.chunk_size)
            chunk_slice = cleaned_text[start:end].strip()

            if len(chunk_slice) < self.min_chunk_length and chunk_index > 0:
                continue

            # Deterministic chunk ID
            hash_src = f"{document_name}_{chunk_index}_{start}_{end}_{chunk_slice[:30]}"
            chunk_id = f"chunk_{hashlib.sha256(hash_src.encode('utf-8')).hexdigest()[:16]}"
            token_est = max(1, len(chunk_slice.split()))

            meta = {
                "document_name": document_name,
                "document_id": document_id or "doc_unknown",
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
                "token_count": token_est,
                **(extra_metadata or {}),
            }

            chunks.append(
                TextChunk(
                    id=chunk_id,
                    text=chunk_slice,
                    document_name=document_name,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    char_start=start,
                    char_end=end,
                    token_count=token_est,
                    metadata=meta,
                )
            )
            chunk_index += 1

            if end >= text_len:
                break

        return chunks

    def process_file(
        self,
        file_path: Path,
        document_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Extract text and chunk a document in one call."""
        text = self.extract_text(file_path)
        return self.chunk_text(
            text=text,
            document_name=file_path.name,
            document_id=document_id,
            extra_metadata=extra_metadata,
        )
