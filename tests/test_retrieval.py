"""
tests/test_retrieval.py
=======================
Integration tests for the full RAG pipeline — both Strategy A and B.

Coverage:
  * Strategy A returns ``QueryResult`` with ``strategy='A'`` and no expansions.
  * Strategy A scores are in [−1, 1] (cosine similarity range).
  * Strategy A returns at most ``top_k`` results.
  * Strategy A surfaces relevant documents for known queries.
  * Strategy B produces expanded queries.
  * Strategy B results include at least as many relevant docs as Strategy A.
  * Strategy B deduplicates: no repeated document IDs in results.
  * Strategy B re-ranks: results are ordered by score descending.
  * Mock expander is used (no GCP calls).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.models import Document, QueryResult
from rag_engine.pipeline.rag_pipeline import RAGPipeline
from rag_engine.embeddings.local_embedder import LocalEmbedder
from rag_engine.query_expansion.generative_expander import GenerativeQueryExpander
from rag_engine.storage.faiss_store import FAISSVectorStore
from rag_engine.config import Settings


# ─── Strategy A tests ─────────────────────────────────────────────────────────


class TestStrategyA:
    """Tests for raw vector similarity retrieval."""

    def test_returns_query_result(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        result = pipeline_with_corpus.query_strategy_a("horizontal scaling")
        assert isinstance(result, QueryResult)

    def test_strategy_label(self, pipeline_with_corpus: RAGPipeline) -> None:
        result = pipeline_with_corpus.query_strategy_a("caching")
        assert result.strategy == "A"

    def test_no_expanded_queries(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        result = pipeline_with_corpus.query_strategy_a("disaster recovery")
        assert result.expanded_queries == []

    def test_top_k_respected(self, pipeline_with_corpus: RAGPipeline) -> None:
        result = pipeline_with_corpus.query_strategy_a("Kafka messaging events")
        assert len(result.results) <= pipeline_with_corpus.config.top_k

    def test_scores_in_range(self, pipeline_with_corpus: RAGPipeline) -> None:
        result = pipeline_with_corpus.query_strategy_a("circuit breaker")
        for res in result.results:
            assert -1.0 <= res.score <= 1.0 + 1e-5, (
                f"Score {res.score} out of cosine range."
            )

    def test_results_sorted_descending(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        result = pipeline_with_corpus.query_strategy_a("Redis LRU eviction")
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_retrieved(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        """
        A query about 'Kafka consumer groups exactly-once' should surface
        the Kafka document in the top results.
        """
        result = pipeline_with_corpus.query_strategy_a(
            "Kafka consumer groups exactly-once"
        )
        ids = [r.document.id for r in result.results]
        assert any("kafka" in doc_id.lower() or "004" in doc_id for doc_id in ids), (
            f"Expected a Kafka-related document, got {ids}"
        )

    def test_original_query_preserved(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        q = "What is the RPO for disaster recovery?"
        result = pipeline_with_corpus.query_strategy_a(q)
        assert result.original_query == q


# ─── Strategy B tests ─────────────────────────────────────────────────────────


class TestStrategyB:
    """Tests for AI-enhanced retrieval with query expansion."""

    def test_returns_query_result(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        result = pipeline_with_corpus.query_strategy_b("caching")
        assert isinstance(result, QueryResult)

    def test_strategy_label(self, pipeline_with_corpus: RAGPipeline) -> None:
        result = pipeline_with_corpus.query_strategy_b("circuit breaker")
        assert result.strategy == "B"

    def test_produces_expanded_queries(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        result = pipeline_with_corpus.query_strategy_b("scaling")
        assert len(result.expanded_queries) > 0, (
            "Strategy B must produce at least one expanded query."
        )

    def test_no_duplicate_doc_ids(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        result = pipeline_with_corpus.query_strategy_b("disaster recovery")
        ids = [r.document.id for r in result.results]
        assert len(ids) == len(set(ids)), "Duplicate document IDs in results."

    def test_results_sorted_descending(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        result = pipeline_with_corpus.query_strategy_b("kafka messaging")
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, pipeline_with_corpus: RAGPipeline) -> None:
        result = pipeline_with_corpus.query_strategy_b("caching Redis")
        assert len(result.results) <= pipeline_with_corpus.config.top_k

    def test_strategy_b_at_least_as_good_as_a(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        """
        Strategy B should never return a lower top-1 score than Strategy A
        for the same query (expansion can only add more search coverage).
        """
        query = "circuit breaker error rate resilience"
        result_a = pipeline_with_corpus.query_strategy_a(query)
        result_b = pipeline_with_corpus.query_strategy_b(query)

        if result_a.results and result_b.results:
            # Strategy B might find a different top doc (via expansion),
            # but its top score should be >= Strategy A's top score.
            assert result_b.results[0].score >= result_a.results[0].score - 0.05, (
                "Strategy B top score should be no worse than Strategy A."
            )

    def test_mock_expander_used(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        """Verify that the expander is in mock mode (no GCP calls)."""
        assert pipeline_with_corpus.expander.mock is True


# ─── Cross-strategy comparison ────────────────────────────────────────────────


class TestStrategyComparison:
    """Comparative tests between Strategy A and Strategy B."""

    @pytest.mark.parametrize("query", [
        "horizontal scaling CPU threshold",
        "Redis LRU eviction database load",
        "Kafka exactly-once semantics consumer",
    ])
    def test_both_strategies_return_results(
        self, pipeline_with_corpus: RAGPipeline, query: str
    ) -> None:
        result_a = pipeline_with_corpus.query_strategy_a(query)
        result_b = pipeline_with_corpus.query_strategy_b(query)
        assert len(result_a.results) > 0, f"Strategy A returned 0 results for: {query}"
        assert len(result_b.results) > 0, f"Strategy B returned 0 results for: {query}"

    def test_rank_one_is_consistent(
        self, pipeline_with_corpus: RAGPipeline
    ) -> None:
        """
        For a highly specific query, both strategies should agree on rank-1.
        """
        query = "Resilience4j sliding window error rate 50%"
        r_a = pipeline_with_corpus.query_strategy_a(query)
        r_b = pipeline_with_corpus.query_strategy_b(query)
        if r_a.results and r_b.results:
            # Both should return the circuit-breaker doc as the best match.
            assert r_a.results[0].document.id == r_b.results[0].document.id, (
                "Both strategies should agree on rank-1 for a highly specific query."
            )
