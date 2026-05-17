"""
pipeline/rag_pipeline.py
========================
End-to-end RAG pipeline supporting two retrieval strategies.

Strategy A — Raw Vector Search
    1. Embed the query with ``LocalEmbedder``.
    2. Search FAISS for top-k nearest neighbours.
    3. Return results (no LLM synthesis in Strategy A — pure retrieval).

Strategy B — AI-Enhanced Retrieval
    1. Expand the query into ``n`` alternatives via ``GenerativeQueryExpander``.
    2. Embed ALL queries (original + expansions).
    3. Search FAISS for each query vector.
    4. Merge results; deduplicate by document ID; re-rank by max score.
    5. Return top-k results.

The pipeline also manages corpus ingestion: it reads a JSON file,
embeds all documents in a single batch call, and adds them to the store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from rag_engine.config import Settings
from rag_engine.embeddings.local_embedder import LocalEmbedder
from rag_engine.models import Document, DocumentMetadata, QueryResult, SearchResult
from rag_engine.query_expansion.generative_expander import GenerativeQueryExpander
from rag_engine.storage.faiss_store import FAISSVectorStore


class RAGPipeline:

    def __init__(
        self,
        embedder: LocalEmbedder,
        store: FAISSVectorStore,
        expander: GenerativeQueryExpander,
        config: Optional[Settings] = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.expander = expander
        self.config = config or Settings()


    def ingest_corpus(self, save_index: bool = True) -> int:

        corpus_path = Path(self.config.corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        with corpus_path.open(encoding="utf-8") as fh:
            raw_docs = json.load(fh)

        documents: List[Document] = []
        for raw in raw_docs:
            meta = raw.get("metadata", {})
            documents.append(
                Document(
                    id=raw["id"],
                    title=raw["title"],
                    content=raw["content"],
                    metadata=DocumentMetadata(
                        source=meta.get("source", ""),
                        tags=meta.get("tags", []),
                        page=meta.get("page"),
                    ),
                )
            )

        texts = [doc.to_embedding_text() for doc in documents]
        embeddings = self.embedder.embed_batch(texts)
        self.store.add_documents(documents, embeddings)

        if save_index:
            self.store.save(self.config.index_path)

        return len(documents)

    def ingest_documents(
        self, documents: List[Document], save_index: bool = False
    ) -> None:
        """
        Ingest an arbitrary list of documents programmatically.
        Useful for tests that want to bypass the corpus file.
        """
        texts = [doc.to_embedding_text() for doc in documents]
        embeddings = self.embedder.embed_batch(texts)
        self.store.add_documents(documents, embeddings)
        if save_index:
            self.store.save(self.config.index_path)

    # ── Query — Strategy A ────────────────────────────────────────────────────

    def query_strategy_a(self, query: str) -> QueryResult:
        """
        Strategy A: raw vector similarity search.

        Embeds the query and retrieves top-k documents from FAISS.
        No query expansion; no LLM synthesis.

        Parameters
        ----------
        query:
            Natural-language question from the user.

        Returns
        -------
        QueryResult
            ``strategy='A'``, ``expanded_queries=[]``, ``answer=''``.
        """
        q_vec = self.embedder.embed(query)
        results = self.store.search(q_vec, top_k=self.config.top_k)
        return QueryResult(
            original_query=query,
            expanded_queries=[],
            results=results,
            answer="",  # Strategy A is retrieval-only
            strategy="A",
        )

    # ── Query — Strategy B ────────────────────────────────────────────────────

    def query_strategy_b(self, query: str) -> QueryResult:
        """
        Strategy B: AI-enhanced retrieval via query expansion.

        1. Expand the query into ``n`` alternatives.
        2. Embed every variant (including the original).
        3. Search FAISS per variant; merge and re-rank by max score.

        Parameters
        ----------
        query:
            Natural-language question from the user.

        Returns
        -------
        QueryResult
            ``strategy='B'``, ``expanded_queries=[...]``.
        """
        expanded = self.expander.expand(query)
        all_queries = [query] + expanded

        # Collect results from every query variant.
        seen: dict[str, SearchResult] = {}
        for variant in all_queries:
            q_vec = self.embedder.embed(variant)
            for res in self.store.search(q_vec, top_k=self.config.top_k):
                doc_id = res.document.id
                # Keep the highest score across all variants.
                if doc_id not in seen or res.score > seen[doc_id].score:
                    seen[doc_id] = res

        # Re-rank globally by score descending, cap at top_k.
        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        top_results = merged[: self.config.top_k]

        # Assign new contiguous ranks.
        ranked = [
            SearchResult(document=r.document, score=r.score, rank=i)
            for i, r in enumerate(top_results, start=1)
        ]

        return QueryResult(
            original_query=query,
            expanded_queries=expanded,
            results=ranked,
            answer="",
            strategy="B",
        )


