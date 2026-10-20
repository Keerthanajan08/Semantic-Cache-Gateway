"""
Thin wrapper around sentence-transformers so the rest of the app never
touches the model directly. all-MiniLM-L6-v2 is ~80MB and 384-dimensional:
cheap enough to run on a free-tier CPU host with ~10ms per query.
"""

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

from app.config import settings

Vector = npt.NDArray[np.float32]


class Embedder:
    def __init__(self, model_name: str) -> None:
        # Imported lazily: sentence-transformers pulls in torch, which takes
        # seconds to import. Modules that only need cosine_similarity (cache,
        # rag, the fast test suite) stay lightweight.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model: Any = SentenceTransformer(model_name)

    def encode(self, text: str) -> Vector:
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide embedder; the model loads once, not once per request."""
    return Embedder(settings.embedding_model)


def cosine_similarity(a: Vector, b: Vector) -> float:
    # Embeddings are L2-normalized at encode time, so cosine == dot product.
    return float(np.dot(a, b))
