
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine.config import Settings
from rag_engine.embeddings.local_embedder import LocalEmbedder
from rag_engine.pipeline.rag_pipeline import RAGPipeline
from rag_engine.query_expansion.generative_expander import GenerativeQueryExpander
from rag_engine.storage.faiss_store import FAISSVectorStore


def build_pipeline() -> RAGPipeline:
    settings = Settings(
        mock_gcp=True,
        top_k=3,
        corpus_path=Path(__file__).parent / "data" / "corpus.json",
    )
    embedder = LocalEmbedder(model_name=settings.embedding_model, normalise=True)
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    expander = GenerativeQueryExpander(mock=True, n_expansions=3)
    pipeline = RAGPipeline(
        embedder=embedder,
        store=store,
        expander=expander,
        config=settings,
    )
    n = pipeline.ingest_corpus(save_index=False)
    print(f"✓ Ingested {n} documents into FAISS index.\n")
    return pipeline


def print_result(result, strategy_name: str) -> None:
    print(f"{'─'*60}")
    print(f"  {strategy_name}")
    print(f"{'─'*60}")
    if hasattr(result, 'expanded_queries') and result.expanded_queries:
        print("  Expanded queries:")
        for eq in result.expanded_queries:
            print(f"    • {eq}")
        print()
    for r in result.results:
        print(f"  [{r.rank}] score={r.score:.4f}  id={r.document.id}  {r.document.title}")
        snippet = r.document.content[:120].replace("\n", " ")
        print(f"       {snippet}…")
    print()


DEMO_QUERIES = [
    "How does Redis caching reduce database load during traffic spikes?",
    "What prevents cascade failures when a downstream service fails?",
    "How are Kafka events traced across microservices for audit compliance?",
]


def main() -> None:
    pipeline = build_pipeline()

    for query in DEMO_QUERIES:
        print(f"\n{'='*60}")
        print(f"  Query: {query}")
        print(f"{'='*60}\n")

        result_a = pipeline.query_strategy_a(query)
        print_result(result_a, "Strategy A — Raw Vector Search")

        result_b = pipeline.query_strategy_b(query)
        print_result(result_b, "Strategy B — AI-Enhanced (Query Expansion)")


if __name__ == "__main__":
    main()