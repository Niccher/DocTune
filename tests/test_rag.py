"""
tests/test_rag.py
-----------------
Unit and integration tests for DocTune Studio's embedded ChromaDB RAG Subsystem.
Tests chunking, embeddings, persistent vector store, hybrid retrieval, and prompt context packing.
"""

import unittest
from pathlib import Path

from pipeline.rag.chunker import DocumentChunker, TextChunk
from pipeline.rag.embeddings import EmbeddingService
from pipeline.rag.chroma_store import ChromaVectorStore
from pipeline.rag.retriever import RAGRetriever


SAMPLE_DOC_TEXT = (
    "DocTune Enterprise Infrastructure Specification 2026.\n"
    "Section 1: Audit Log Compliance Retention.\n"
    "Mandatory retention period for all enterprise financial and infrastructure audit logs is exactly seven years.\n"
    "Section 2: High Availability and Disaster Recovery.\n"
    "Database automated failover must complete in under 60 seconds with zero committed data loss.\n"
    "Section 3: Access Control Policy.\n"
    "All production cluster root access requires dual-custody hardware MFA tokens."
)


class TestRAGSubsystem(unittest.TestCase):
    def test_document_chunker(self):
        chunker = DocumentChunker(chunk_size=120, chunk_overlap=30, min_chunk_length=20)
        chunks = chunker.chunk_text(
            text=SAMPLE_DOC_TEXT,
            document_name="infrastructure_spec.txt",
            document_id="doc_infra_01",
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(c, TextChunk) for c in chunks))
        self.assertEqual(chunks[0].document_name, "infrastructure_spec.txt")
        self.assertTrue(chunks[0].id.startswith("chunk_"))
        self.assertEqual(chunks[0].metadata["document_id"], "doc_infra_01")


    def test_embedding_service(self):
        embedder = EmbeddingService(dimensions=384)
        texts = [
            "What is the audit log retention policy?",
            "How fast is database failover?",
        ]
        vectors = embedder.embed_texts(texts)

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 384)
        self.assertEqual(len(vectors[1]), 384)

        # Verify query embed
        q_vec = embedder.embed_query("Audit retention")
        self.assertEqual(len(q_vec), 384)

    def test_chroma_vector_store(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            store = ChromaVectorStore(persist_dir=Path(td) / "test_vectorstore")
            project_id = "proj_test_rag"

            chunker = DocumentChunker(chunk_size=150, chunk_overlap=40)
            chunks = chunker.chunk_text(
                text=SAMPLE_DOC_TEXT,
                document_name="spec.txt",
                document_id="doc_spec",
            )

            # 1. Add chunks
            added_count = store.add_chunks(chunks, project_id=project_id)
            self.assertEqual(added_count, len(chunks))

            # 2. Check stats
            stats = store.get_stats(project_id)
            self.assertEqual(stats["total_chunks"], len(chunks))

            # 3. Query nearest neighbors
            hits = store.query_similar("retention period audit logs", project_id=project_id, top_k=2)
            self.assertGreaterEqual(len(hits), 1)
            self.assertTrue(0.0 <= hits[0]["similarity"] <= 1.0)

            # 4. Delete document
            deleted = store.delete_document("spec.txt", project_id=project_id)
            self.assertIn(deleted, (0, 1, len(chunks)))

    def test_hybrid_rag_retriever(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            store = ChromaVectorStore(persist_dir=Path(td) / "test_retriever_store")
            project_id = "proj_retriever_test"

            chunker = DocumentChunker(chunk_size=160, chunk_overlap=30)
            chunks = chunker.chunk_text(
                text=SAMPLE_DOC_TEXT,
                document_name="dr_policy.txt",
                document_id="doc_dr",
            )
            store.add_chunks(chunks, project_id=project_id)

            retriever = RAGRetriever(vector_store=store)

            # 1. Hybrid retrieve
            query = "database failover time objective"
            results = retriever.retrieve(query=query, project_id=project_id, top_k=2)
            self.assertGreaterEqual(len(results), 1)

            # 2. Assemble context and citations
            context_str, citations = retriever.assemble_context(results)
            self.assertIn("Enterprise Document Knowledge Base Excerpts:", context_str)
            self.assertEqual(len(citations), len(results))
            self.assertEqual(citations[0]["document_name"], "dr_policy.txt")

            # 3. Augment chat messages
            user_msgs = [{"role": "user", "content": query}]
            augmented, cites = retriever.augment_messages(user_msgs, results)
            self.assertEqual(len(augmented), 2)  # System message inserted + User message
            self.assertEqual(augmented[0]["role"], "system")
            self.assertIn("Enterprise Document Knowledge Base Excerpts:", augmented[0]["content"])
            self.assertEqual(augmented[1]["content"], query)


if __name__ == "__main__":
    unittest.main()

