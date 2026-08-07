"""
Wraps sentence-transformers so the rest of the app never touches the model
directly. Uses a small (~80MB) model — cheap enough to run on a free-tier
host with no GPU.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    # Cached so the model loads once per process, not once per request.
    return Embedder(settings.embedding_model)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already normalized by sentence-transformers, so this
    # is just a dot product — cheap enough to do in a Python loop at
    # small (portfolio-project) scale.
    return float(np.dot(a, b))
