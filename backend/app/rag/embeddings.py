from threading import Lock
from typing import Protocol

import numpy as np


class EmbeddingService(Protocol):
    dimension: int

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbeddings:
    """Lazily loads one process-wide SentenceTransformer model."""

    _models: dict[str, object] = {}
    _lock = Lock()

    def __init__(self, model_name: str, dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension

    @property
    def model(self):
        if self.model_name not in self._models:
            with self._lock:
                if self.model_name not in self._models:
                    from sentence_transformers import SentenceTransformer

                    self._models[self.model_name] = SentenceTransformer(self.model_name)
        return self._models[self.model_name]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        values = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(values, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
