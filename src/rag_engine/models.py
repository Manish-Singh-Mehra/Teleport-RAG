from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DocumentMetadata:
    """Flexible bag of metadata attached to every ingested document."""
    source: str = ""
    tags: List[str] = field(default_factory=list)
    page: Optional[int] = None

    def to_dict(self) -> dict:
        return {"source": self.source, "tags": self.tags, "page": self.page}

@dataclass
class Document:
    id: str
    title: str
    content: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    def to_embedding_text(self) -> str:
        return f"{self.title}: {self.content}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class SearchResult:

    document: Document
    score: float
    rank: int

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"id={self.document.id!r}, title={self.document.title!r})"
        )
       
@dataclass
class QueryResult:

    original_query: str
    expanded_queries: List[str]
    results: List[SearchResult]
    answer: str
    strategy: str  # 'A' | 'B'