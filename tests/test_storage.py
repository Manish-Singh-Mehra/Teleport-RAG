"""
Unit tests for the FAISSVectorStore (Storage module).

Coverage:
  * Empty store returns no results.
  * Adding documents increases count.
  * Search returns correct number of results.
  * Results are sorted by score (descending).
  * Known-similar document appears in top-k.
  * Shape mismatch raises ValueError.
  * Persist → load round-trip preserves all documents and scores.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

# Ensure src/ is on sys.path when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.embeddings.local_embedder import LocalEmbedder
from rag_engine.models import Document, DocumentMetadata
from rag_engine.storage.faiss_store import FAISSVectorStore


class TestFAISSVectorStore:
    """Unit tests for the FAISSVectorStore."""

    # Empty store 

    def test_empty_store_count(self, empty_store: FAISSVectorStore) -> None:
        assert empty_store.document_count == 0

    def test_empty_store_search_returns_empty(
        self,
        empty_store: FAISSVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        q_vec = embedder.embed("anything")
        results = empty_store.search(q_vec, top_k=3)
        assert results == []

    # Document ingestion 

    def test_add_documents_increases_count(
        self,
        populated_store: FAISSVectorStore,
        sample_documents: list[Document],
    ) -> None:
        assert populated_store.document_count == len(sample_documents)

    def test_shape_mismatch_raises(
        self,
        empty_store: FAISSVectorStore,
        sample_documents: list[Document],
    ) -> None:
        bad_embeddings = np.zeros((2, empty_store._dim), dtype=np.float32)
        with pytest.raises(ValueError, match="document count"):
            empty_store.add_documents(
                sample_documents[:3],  # 3 docs
                bad_embeddings,         # 2 embeddings
            )

    # Search

    def test_search_returns_top_k(
        self,
        populated_store: FAISSVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        q_vec = embedder.embed("auto-scaling CPU load")
        results = populated_store.search(q_vec, top_k=3)
        assert len(results) == 3

    def test_search_results_sorted_descending(
        self,
        populated_store: FAISSVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        q_vec = embedder.embed("caching database Redis")
        results = populated_store.search(q_vec, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by score."

    def test_search_rank_starts_at_one(
        self,
        populated_store: FAISSVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        q_vec = embedder.embed("circuit breaker resilience")
        results = populated_store.search(q_vec, top_k=3)
        assert results[0].rank == 1

    def test_relevant_doc_in_top_results(
        self,
        populated_store: FAISSVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        """
        The 'Redis Caching' document should rank first for a query about
        Redis and LRU caching — a basic sanity check on retrieval quality.
        """
        q_vec = embedder.embed("Redis LRU eviction cache")
        results = populated_store.search(q_vec, top_k=3)
        top_ids = [r.document.id for r in results]
        assert "test_002" in top_ids, (
            f"Expected 'test_002' (Redis Caching) in top results, got {top_ids}"
        )

    def test_search_top_k_capped_by_store_size(
        self,
        populated_store: FAISSVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        q_vec = embedder.embed("anything")
        results = populated_store.search(q_vec, top_k=100)
        assert len(results) <= populated_store.document_count


    def test_save_load_roundtrip(
        self,
        populated_store: FAISSVectorStore,
        embedder: LocalEmbedder,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.index"
            populated_store.save(index_path)

            # Verify both files exist.
            assert index_path.exists()
            assert index_path.with_suffix(".docs.json").exists()

            # Reload and verify.
            loaded = FAISSVectorStore.load(index_path)
            assert loaded.document_count == populated_store.document_count

            q_vec = embedder.embed("Kafka consumer groups")
            original_results = populated_store.search(q_vec, top_k=3)
            loaded_results = loaded.search(q_vec, top_k=3)

            orig_ids = [r.document.id for r in original_results]
            load_ids = [r.document.id for r in loaded_results]
            assert orig_ids == load_ids, "Save/load must preserve result ordering."

    def test_save_requires_path(
        self, empty_store: FAISSVectorStore
    ) -> None:
        with pytest.raises(ValueError, match="No path"):
            empty_store.save()
