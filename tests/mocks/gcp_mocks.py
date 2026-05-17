"""

Lightweight mocks for the Vertex AI SDK.

These mocks intercept every GCP API call so that the full test suite
runs without network access, GCP credentials, or project billing.

Classes

MockTextEmbeddingModel
    Simulates ``vertexai.language_models.TextEmbeddingModel``.
    Returns deterministic, seeded random vectors so similarity
    assertions are repeatable across machines.

MockGenerativeModel
    Simulates ``vertexai.generative_models.GenerativeModel`` (Gemini).
    Returns hand-crafted query expansions keyed on common test queries.

make_vertexai_mock()
    Returns a ``unittest.mock.MagicMock`` that patches the ``vertexai``
    module at the import level.
"""

from __future__ import annotations

import hashlib
from typing import List
from unittest.mock import MagicMock

import numpy as np


# ``TextEmbeddingModel mock  


class _FakeEmbeddingResponse:
    """Mirrors the shape of a real TextEmbedding response object."""

    def __init__(self, values: List[float]) -> None:
        self.values = values


class MockTextEmbeddingModel:
    """
    Mock of ``vertexai.language_models.TextEmbeddingModel``.

    Embeddings are deterministic: the seed is derived from the SHA-256
    of the input string so that identical texts always yield identical
    vectors, while different texts yield different vectors.
    """

    EMBEDDING_DIM = 768  # matches textembedding-gecko@003

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockTextEmbeddingModel":  # noqa
        return cls()

    def get_embeddings(
        self, texts: List[str]
    ) -> List[_FakeEmbeddingResponse]:
        responses = []
        for text in texts:
            # Seed from text hash → deterministic, text-specific vector.
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.EMBEDDING_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-12  # L2-normalise
            responses.append(_FakeEmbeddingResponse(vec.tolist()))
        return responses


# GenerativeModel mock 


class _FakeGenerativeResponse:
    """Mirrors the shape of a real GenerateContentResponse."""

    def __init__(self, text: str) -> None:
        self.text = text


_EXPANSION_TABLE: dict[str, str] = {
    "scaling": (
        "How does the system handle horizontal scaling under load?\n"
        "What mechanisms exist for auto-provisioning compute resources?\n"
        "How are traffic spikes managed in the infrastructure?"
    ),
    "caching": (
        "What caching strategy is used to reduce database load?\n"
        "How is the Redis cache invalidated during updates?\n"
        "What eviction policy governs cache memory management?"
    ),
    "circuit breaker": (
        "How does the system prevent cascade failures between services?\n"
        "What thresholds trigger the resilience circuit breaker?\n"
        "How does the half-open state work in the fault-tolerance layer?"
    ),
    "kafka": (
        "How are events processed asynchronously in the messaging layer?\n"
        "What consumer group strategy is used for event parallelism?\n"
        "How does the system guarantee exactly-once message delivery?"
    ),
    "disaster recovery": (
        "What is the recovery point objective for the primary database?\n"
        "How does the geo-redundant failover mechanism work?\n"
        "What replication strategy backs the disaster recovery plan?"
    ),
}

_DEFAULT_EXPANSIONS = (
    "Can you rephrase this question from a different angle?\n"
    "What is another way to ask about this topic?\n"
    "How would an expert reformulate this information need?"
)


class MockGenerativeModel:
    """
    Mock of ``vertexai.generative_models.GenerativeModel``.

    Returns pre-canned expansions for known test-query keywords,
    or a generic three-line expansion for unknown queries.
    """

    def generate_content(self, prompt: str) -> _FakeGenerativeResponse:
        prompt_lower = prompt.lower()
        for keyword, expansions in _EXPANSION_TABLE.items():
            if keyword in prompt_lower:
                return _FakeGenerativeResponse(expansions)
        return _FakeGenerativeResponse(_DEFAULT_EXPANSIONS)


# Module-level patcher factory

def make_vertexai_mock() -> MagicMock:
    """
    Build a MagicMock that replaces the ``vertexai`` module.
    """
    mock_vertexai = MagicMock()
    mock_vertexai.language_models.TextEmbeddingModel = MockTextEmbeddingModel
    mock_vertexai.generative_models.GenerativeModel = MockGenerativeModel
    return mock_vertexai
