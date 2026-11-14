"""
pipeline/rag/__init__.py
------------------------
Embedded ChromaDB Retrieval-Augmented Generation (RAG) Subsystem for DocTune Studio.
"""

from pipeline.rag.chunker import TextChunk, DocumentChunker
from pipeline.rag.embeddings import EmbeddingService, embedding_service
from pipeline.rag.chroma_store import ChromaVectorStore, chroma_store
from pipeline.rag.retriever import RAGRetriever, rag_retriever

__all__ = [
    "TextChunk",
    "DocumentChunker",
    "EmbeddingService",
    "embedding_service",
    "ChromaVectorStore",
    "chroma_store",
    "RAGRetriever",
    "rag_retriever",
]
