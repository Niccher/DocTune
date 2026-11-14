# ADR 0004: In-Process ChromaDB Architecture for Hybrid RAG

* **Status**: Accepted
* **Date**: 2026-11-14

## Context and Problem Statement
DocTune Studio requires a Retrieval-Augmented Generation (RAG) subsystem to benchmark fine-tuned models against ground-truth corporate documents and provide factual grounding during live inference. Deploying an external vector database (such as Milvus, Qdrant, or Pinecone) introduces additional container overhead, port management complexity, network latency, and memory footprints that can degrade performance on resource-constrained single-node workstations.

## Decision Drivers
* Zero new container overhead: must run efficiently without adding a 6th or 7th Docker container.
* Air-gapped privacy: embeddings, chunks, and vector similarity operations must remain 100% on-premises without external SaaS dependencies.
* Hybrid search accuracy: corporate technical terms and acronyms require lexical keyword matching (BM25) combined with dense semantic embeddings.
* Multi-tenant project isolation: collections must be strictly separated by project identifier.

## Considered Options
1. Standalone Vector DB Container (e.g. Qdrant / Milvus / Weaviate container).
2. Embedded In-Process ChromaDB (`chromadb.PersistentClient`) with SQLite and HNSW backend.
3. PostgreSQL `pgvector` extension inside the existing PostgreSQL container.

## Decision Outcome
**Option 2: Embedded In-Process ChromaDB**.
ChromaDB's persistent client runs directly in-process within the FastAPI web application and Celery background workers. Vector indices and SQLite metadata persist under `./data/vectorstore/`, sharing existing volume mounts without requiring extra ports, Docker network configurations, or background daemon processes. Hybrid search is achieved via Reciprocal Rank Fusion ($k=60$) combining dense cosine similarity (`BAAI/bge-small-en-v1.5`) and BM25 token scoring.
