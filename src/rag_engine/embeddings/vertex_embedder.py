from __future__ import annotations
from typing import Sequence
import numpy as np

class VertexAIEmbedder:
    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model_name: str = "textembedding-gecko@003",
        normalise: bool = True
    ) -> None:

        from vertexai.language_models import TextEmbeddingModel  # type: ignore
        import vertexai  # type: ignore

        vertexai.init(project=project, location=location)
        self._model = TextEmbeddingModel.from_pretrained(model_name)
        self.normalise = normalise
        self.model_name = model_name

    @property
    def embedding_dim(self)-> int:
        # gecko@003 emits 768-d vectors by default.
        return 768
    
    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        from vertexai.language_models import TextEmbeddingInput 

        model_texts: list[str | TextEmbeddingInput] = list(texts)
        embeddings_response = self._model.get_embeddings(texts=model_texts)
        vectors = np.array(
            [r.values for r in embeddings_response], dtype= np.float32
        )
        if self.normalise:
            norms = np.linalg.norm(vectors, axis= 1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors
    
    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]