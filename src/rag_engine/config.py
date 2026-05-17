
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Runtime configuration for the RAG pipeline."""

    # Embedding 
    embedding_model: str = "all-MiniLM-L6-v2"
    """Local sentence-transformer model that simulates textembedding-gecko."""

    normalise: bool = True
    """L2-normalise embeddings so cosine similarity == dot product."""

    # Retrieval
    top_k: int = 3
    """Number of chunks to return per query."""

    similarity_metric: str = "cosine"
    """'cosine' or 'euclidean' — used in documentation / benchmark only."""

    # Storage
    index_path: Path = field(default_factory=lambda: Path("data/faiss.index"))
    """Path to persist the FAISS index."""

    corpus_path: Path = field(default_factory=lambda: Path("data/corpus.json"))
    """JSON file containing the 10 technical paragraphs for ingestion."""

    # GCP / mocking
    mock_gcp: bool = True
    """When True, all GCP SDK calls are intercepted by mocks."""

    gcp_project: str = "teleport-rag-demo"
    gcp_location: str = "us-central1"

    # Query expansion
    expander_temperature: float = 0.2
    """Temperature used by the (mocked) GenerativeModel for query rewriting."""
