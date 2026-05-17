"""
FAISS-backed vector store for the RAG Engine.

Similarity metric choice — cosine vs euclidean
----------------------------------------------
We use **cosine similarity** (via normalised inner product):

  * Scale-invariant: embedding magnitude does not inflate scores.
  * Matches what ``textembedding-gecko`` recommends in its documentation.
  * Scores are bounded in [-1, 1] → easy threshold reasoning.

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from rag_engine.models import Document, DocumentMetadata, SearchResult

class FAISSVectorStore:
    def __init__(
        self,
        embedding_dim: int,
        index_path: Optional[Path] = None,
        ) -> None:
            self._dim = embedding_dim
            self._index = faiss.IndexFlatIP(embedding_dim)  # cosine (normalised)
            self._documents: List[Document] = []
            self._index_path = index_path

    # Ingestion

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"document count ({len(documents)}) != "
                f"embedding rows ({embeddings.shape[0]})"
            )

        self._index.add(embeddings) # type: ignore
        self._documents.extend(documents)

    # Retrieval
    def search(self, query_vector: np.ndarray, top_k: int = 3,) -> List[SearchResult]:
        if self._index.ntotal == 0:
            return []
          
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(
            query_vector.reshape(1, -1).astype(np.float32), k
        )  # type: ignore

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]), start=1
        ):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    document=self._documents[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return results
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Persist index and document metadata to disk."""
        save_path = path or self._index_path
        if save_path is None:
            raise ValueError("No path provided for saving the index.")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(save_path))

        meta_path = save_path.with_suffix(".docs.json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump([doc.to_dict() for doc in self._documents], fh, indent=2)
        return save_path

    @classmethod
    def load(cls, path: Path) -> "FAISSVectorStore":
        """Restore a previously saved index."""
        path = Path(path)
        index = faiss.read_index(str(path))
        store = cls(embedding_dim=index.d, index_path=path)
        store._index = index

        meta_path = path.with_suffix(".docs.json")
        with meta_path.open(encoding="utf-8") as fh:
            raw_docs = json.load(fh)

        for raw in raw_docs:
            meta = raw.get("metadata", {})
            store._documents.append(
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
        return store


    @property
    def document_count(self) -> int:
        return self._index.ntotal
