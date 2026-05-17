from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on sys.path when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.embeddings.local_embedder import LocalEmbedder


class TestLocalEmbedder:
    """Unit tests for the LocalEmbedder class."""

    def test_embed_returns_1d_array(self, embedder: LocalEmbedder) -> None:
        vec = embedder.embed("Horizontal scaling distributes load.")
        assert vec.ndim == 1, "embed() must return a 1-D array."

    def test_embed_dim_matches_property(self, embedder: LocalEmbedder) -> None:
        vec = embedder.embed("Test sentence.")
        assert vec.shape[0] == embedder.embedding_dim

    def test_embed_batch_shape(self, embedder: LocalEmbedder) -> None:
        texts = ["sentence one", "sentence two", "sentence three"]
        matrix = embedder.embed_batch(texts)
        assert matrix.shape == (3, embedder.embedding_dim)

    def test_embed_batch_dtype(self, embedder: LocalEmbedder) -> None:
        matrix = embedder.embed_batch(["test"])
        assert matrix.dtype == np.float32

    # Normalisation  

    def test_embed_is_unit_norm(self, embedder: LocalEmbedder) -> None:
        vec = embedder.embed("Caching reduces database load.")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}."

    def test_embed_batch_all_unit_norm(self, embedder: LocalEmbedder) -> None:
        texts = [
            "Redis Cluster with LRU eviction.",
            "Kafka consumer groups with exactly-once semantics.",
            "Circuit breaker opens at 50% error rate.",
        ]
        matrix = embedder.embed_batch(texts)
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Not all unit norms: {norms}"

    #  Determinism 

    def test_embed_deterministic(self, embedder: LocalEmbedder) -> None:
        text = "Geo-redundant failover with CDC replication."
        vec1 = embedder.embed(text)
        vec2 = embedder.embed(text)
        assert np.allclose(vec1, vec2), "Embedding must be deterministic."

    def test_different_texts_differ(self, embedder: LocalEmbedder) -> None:
        v1 = embedder.embed("Horizontal auto-scaling.")
        v2 = embedder.embed("Kafka event streaming.")
        assert not np.allclose(v1, v2), "Different texts must yield different vectors."

    #  Batch vs single consistency 

    def test_batch_matches_individual(self, embedder: LocalEmbedder) -> None:
        texts = ["Scaling.", "Caching.", "Recovery."]
        matrix = embedder.embed_batch(texts)
        for i, text in enumerate(texts):
            single = embedder.embed(text)
            assert np.allclose(
                matrix[i], single, atol=1e-5
            ), f"Batch[{i}] ≠ individual embedding for '{text}'."

    # Empty batch 

    def test_empty_batch_returns_empty(self, embedder: LocalEmbedder) -> None:
        result = embedder.embed_batch([])
        assert result.shape == (0, embedder.embedding_dim) or result.shape[0] == 0
