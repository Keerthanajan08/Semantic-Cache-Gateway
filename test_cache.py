import numpy as np
import pytest

from app.cache import InMemoryCache


def unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_empty_cache_is_a_miss():
    cache = InMemoryCache(threshold=0.9)
    result = cache.lookup(unit_vector(0))
    assert result.hit is False


def test_identical_vector_is_a_hit():
    cache = InMemoryCache(threshold=0.9)
    vec = unit_vector(1)
    cache.store("how do I reset my password", "Here's how...", vec)

    result = cache.lookup(vec)
    assert result.hit is True
    assert result.similarity == pytest.approx(1.0, abs=1e-5)
    assert result.entry.answer == "Here's how..."


def test_dissimilar_vector_is_a_miss():
    cache = InMemoryCache(threshold=0.9)
    cache.store("how do I reset my password", "Here's how...", unit_vector(1))

    # A very different random vector should not cross the threshold.
    result = cache.lookup(unit_vector(999))
    assert result.hit is False


def test_size_tracks_stored_entries():
    cache = InMemoryCache(threshold=0.9)
    assert cache.size() == 0
    cache.store("q1", "a1", unit_vector(1))
    cache.store("q2", "a2", unit_vector(2))
    assert cache.size() == 2
