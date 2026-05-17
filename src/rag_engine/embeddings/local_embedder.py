from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbedder:
    def __init__(self, 
                model_name: str = "all-MiniLM-L6-v2",
                normalise: bool = True,)->None:
        self.model = SentenceTransformer(model_name)
        self.normalise = normalise
        self.model_name = model_name

    @property 
    def embedding_dim(self) -> int:
        getter = getattr(
            self.model,
            "get_embedding_dimension",
            None
        ) or getattr(self.model, "get_sentence_embedding_dimension")

        return getter()
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vector = self.model.encode(texts, 
                                   convert_to_numpy=True,
                                   normalize_embeddings= self.normalise,
                                   show_progress_bar=True,)
        
        return vector.astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]
    
