import sys
import types

import numpy as np
import pytest

import app.embeddings as embeddings_module
from app.config import settings
from app.embeddings import Embedder, cosine_similarity, get_embedder


@pytest.fixture
def stub_sentence_transformers(monkeypatch):
    """Stand-in for the sentence_transformers package (no torch, no download)."""
    created = []

    class SentenceTransformer:
        def __init__(self, name):
            self.name = name
            self.encode_kwargs = []
            created.append(self)

        def encode(self, texts, normalize_embeddings=False):
            self.encode_kwargs.append({"normalize_embeddings": normalize_embeddings})
            if isinstance(texts, str):
                return [3.0, 4.0]
            return [[3.0, 4.0] for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    get_embedder.cache_clear()
    yield created
    get_embedder.cache_clear()


class TestCosineSimilarity:
    def test_identical_unit_vectors(self):
        v = np.array([0.6, 0.8], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0

    def test_opposite(self):
        v = np.array([0.6, 0.8])
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_returns_python_float(self):
        result = cosine_similarity(np.array([1.0], np.float32), np.array([1.0], np.float32))
        assert type(result) is float


class TestEmbedder:
    def test_encode_requests_normalized_float32(self, stub_sentence_transformers):
        embedder = Embedder("some-model")
        vec = embedder.encode("hello")
        assert vec.dtype == np.float32
        assert vec.shape == (2,)
        assert stub_sentence_transformers[0].name == "some-model"
        assert stub_sentence_transformers[0].encode_kwargs == [{"normalize_embeddings": True}]

    def test_encode_batch_returns_matrix(self, stub_sentence_transformers):
        vecs = Embedder("m").encode_batch(["a", "b", "c"])
        assert vecs.shape == (3, 2)
        assert vecs.dtype == np.float32

    def test_get_embedder_loads_configured_model_once(self, stub_sentence_transformers, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", "configured-model")
        assert get_embedder() is get_embedder()
        assert len(stub_sentence_transformers) == 1
        assert stub_sentence_transformers[0].name == "configured-model"


def test_importing_embeddings_does_not_load_torch():
    """Keeps the cache/RAG modules (and the fast test suite) lightweight."""
    assert "sentence_transformers" not in vars(embeddings_module)
