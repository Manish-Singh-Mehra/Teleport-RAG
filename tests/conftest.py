from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ── Ensure src/ is on sys.path when running tests directly 
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.config import Settings
from rag_engine.embeddings.local_embedder import LocalEmbedder
from rag_engine.models import Document, DocumentMetadata
from rag_engine.pipeline.rag_pipeline import RAGPipeline
from rag_engine.query_expansion.generative_expander import GenerativeQueryExpander
from rag_engine.storage.faiss_store import FAISSVectorStore



@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings to keep tests self-contained."""
    return Settings(
        mock_gcp=True,
        top_k=3,
        embedding_model="all-MiniLM-L6-v2",
        corpus_path=Path(__file__).parent.parent / "data" / "corpus.json",
    )


@pytest.fixture(scope="session")
def sample_documents() -> list[Document]:
    """Five minimal documents sufficient for unit-level retrieval tests."""
    return [
        Document(
            id="test_001",
            title="Horizontal Scaling",
            content="Auto-scaler provisions instances when CPU exceeds threshold under peak load.",
            metadata=DocumentMetadata(tags=["scaling"]),
        ),
        Document(
            id="test_002",
            title="Redis Caching",
            content="Redis Cluster with LRU eviction reduces database load during traffic spikes.",
            metadata=DocumentMetadata(tags=["caching"]),
        ),
        Document(
            id="test_003",
            title="Circuit Breaker",
            content="Resilience4j circuit breaker opens when error rate exceeds 50% in sliding window.",
            metadata=DocumentMetadata(tags=["resilience"]),
        ),
        Document(
            id="test_004",
            title="Kafka Messaging",
            content="Kafka consumer groups process events asynchronously with exactly-once semantics.",
            metadata=DocumentMetadata(tags=["messaging"]),
        ),
        Document(
            id="test_005",
            title="Disaster Recovery",
            content="Geo-redundant active-passive failover with RPO of 30 seconds via CDC replication.",
            metadata=DocumentMetadata(tags=["disaster-recovery"]),
        ),
    ]


@pytest.fixture(scope="session")
def full_corpus_path() -> Path:
    return Path(__file__).parent.parent / "data" / "corpus.json"


@pytest.fixture(scope="session")
def embedder() -> LocalEmbedder:
    """Real sentence-transformer embedder — loaded once per test session."""
    return LocalEmbedder(model_name="all-MiniLM-L6-v2", normalise=True)


@pytest.fixture
def empty_store(embedder: LocalEmbedder) -> FAISSVectorStore:
    """A fresh, empty FAISS store."""
    return FAISSVectorStore(embedding_dim=embedder.embedding_dim)


@pytest.fixture
def populated_store(
    embedder: LocalEmbedder,
    sample_documents: list[Document],
) -> FAISSVectorStore:
    """A FAISS store pre-populated with five test documents."""
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    texts = [doc.to_embedding_text() for doc in sample_documents]
    embeddings = embedder.embed_batch(texts)
    store.add_documents(sample_documents, embeddings)
    return store


@pytest.fixture
def mock_expander() -> GenerativeQueryExpander:
    """Query expander forced into mock mode."""
    return GenerativeQueryExpander(mock=True)


@pytest.fixture
def pipeline_with_corpus(
    embedder: LocalEmbedder,
    test_settings: Settings,
) -> RAGPipeline:
    """A fully initialised RAGPipeline with the full 10-document corpus loaded."""
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    expander = GenerativeQueryExpander(mock=True)
    pipe = RAGPipeline(
        embedder=embedder,
        store=store,
        expander=expander,
        config=test_settings,
    )
    pipe.ingest_corpus(save_index=False)
    return pipe
