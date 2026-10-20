"""
Cache contract tests. Every test in CacheContract runs against both
backends (in-memory and Redis via fakeredis), since README promises they
have identical semantics behind BaseCache.
"""
import time

import numpy as np
import pytest

from app.cache import BaseCache, InMemoryCache, RedisCache, get_cache
from app.config import settings
from app.embeddings import cosine_similarity
from tests.conftest import unit_vector


@pytest.fixture(params=["memory", "redis"])
def make_cache(request):
    if request.param == "memory":
        return lambda threshold=0.9: InMemoryCache(threshold=threshold)
    request.getfixturevalue("redis_server")
    return lambda threshold=0.9: RedisCache("redis://fake", threshold=threshold)


class TestCacheContract:
    def test_empty_cache_is_a_miss(self, make_cache):
        result = make_cache().lookup(unit_vector(0))
        assert result.hit is False
        assert result.entry is None
        assert result.similarity == 0.0

    def test_identical_vector_is_a_hit(self, make_cache):
        cache = make_cache()
        vec = unit_vector(1)
        cache.store("how do I reset my password", "Here's how...", vec)

        result = cache.lookup(vec)
        assert result.hit is True
        assert result.similarity == pytest.approx(1.0, abs=1e-5)
        assert result.entry.query == "how do I reset my password"
        assert result.entry.answer == "Here's how..."

    def test_dissimilar_vector_is_a_miss(self, make_cache):
        cache = make_cache()
        cache.store("q", "a", unit_vector(1))
        assert cache.lookup(unit_vector(999)).hit is False

    def test_similarity_exactly_at_threshold_is_a_hit(self, make_cache):
        stored, probe = unit_vector(1), unit_vector(2)
        threshold = cosine_similarity(probe, stored)
        cache = make_cache(threshold=threshold)
        cache.store("q", "a", stored)
        assert cache.lookup(probe).hit is True

    def test_similarity_just_below_threshold_is_a_miss(self, make_cache):
        stored, probe = unit_vector(1), unit_vector(2)
        threshold = cosine_similarity(probe, stored) + 1e-4
        cache = make_cache(threshold=threshold)
        cache.store("q", "a", stored)
        assert cache.lookup(probe).hit is False

    def test_returns_the_best_match_not_the_first(self, make_cache):
        target = unit_vector(10)
        near = target + 0.05 * unit_vector(11)
        near /= np.linalg.norm(near)
        cache = make_cache(threshold=0.5)
        cache.store("far", "far answer", unit_vector(12))
        cache.store("near", "near answer", near)
        cache.store("exact", "exact answer", target)

        result = cache.lookup(target)
        assert result.entry.answer == "exact answer"

    def test_miss_reports_best_similarity_seen(self, make_cache):
        stored, probe = unit_vector(1), unit_vector(2)
        cache = make_cache(threshold=0.999)
        cache.store("q", "a", stored)
        result = cache.lookup(probe)
        assert result.hit is False
        assert result.similarity == pytest.approx(max(cosine_similarity(probe, stored), 0.0), abs=1e-5)

    def test_negative_similarity_is_clamped_to_zero_on_miss(self, make_cache):
        vec = unit_vector(3)
        cache = make_cache()
        cache.store("q", "a", vec)
        result = cache.lookup(-vec)
        assert result.hit is False
        assert result.similarity == 0.0

    def test_size_tracks_stored_entries(self, make_cache):
        cache = make_cache()
        assert cache.size() == 0
        cache.store("q1", "a1", unit_vector(1))
        cache.store("q2", "a2", unit_vector(2))
        assert cache.size() == 2

    def test_entry_records_creation_time(self, make_cache):
        cache = make_cache()
        before = time.time()
        cache.store("q", "a", unit_vector(1))
        entry = cache.lookup(unit_vector(1)).entry
        assert before <= entry.created_at <= time.time()

    def test_unicode_query_and_answer_round_trip(self, make_cache):
        cache = make_cache()
        cache.store("पासवर्ड कैसे बदलें? 🔑", "उत्तर ✅", unit_vector(1))
        entry = cache.lookup(unit_vector(1)).entry
        assert entry.query == "पासवर्ड कैसे बदलें? 🔑"
        assert entry.answer == "उत्तर ✅"


class TestRedisCache:
    def test_entries_survive_a_new_client(self, redis_server):
        """The point of Redis: entries persist across process restarts."""
        RedisCache("redis://fake", threshold=0.9).store("q", "a", unit_vector(1))
        fresh = RedisCache("redis://fake", threshold=0.9)
        assert fresh.size() == 1
        assert fresh.lookup(unit_vector(1)).entry.answer == "a"

    def test_keys_are_namespaced(self, redis_server):
        cache = RedisCache("redis://fake", threshold=0.9)
        cache.store("q", "a", unit_vector(1))
        keys = list(cache.client.scan_iter())
        assert len(keys) == 1
        assert keys[0].startswith(RedisCache.KEY_PREFIX)

    def test_ignores_unrelated_keys_in_the_same_db(self, redis_server):
        cache = RedisCache("redis://fake", threshold=0.9)
        cache.client.set("someone-elses-key", "value")
        cache.client.hset("other:hash", mapping={"x": "1"})
        cache.store("q", "a", unit_vector(1))
        assert cache.size() == 1
        assert cache.lookup(unit_vector(1)).hit is True

    def test_vector_round_trips_as_float32(self, redis_server):
        cache = RedisCache("redis://fake", threshold=0.9)
        vec = unit_vector(5)
        cache.store("q", "a", vec)
        entry = cache.lookup(vec).entry
        assert entry.vector.dtype == np.float32
        np.testing.assert_allclose(entry.vector, vec, rtol=1e-6)


class TestGetCache:
    def test_defaults_to_in_memory(self):
        assert isinstance(get_cache(), InMemoryCache)

    def test_uses_redis_when_url_configured(self, redis_server, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "redis://fake")
        assert isinstance(get_cache(), RedisCache)

    def test_uses_configured_threshold(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_similarity_threshold", 0.42)
        assert get_cache().threshold == 0.42

    def test_is_a_singleton(self):
        assert get_cache() is get_cache()


def test_base_cache_is_abstract():
    base = BaseCache()
    with pytest.raises(NotImplementedError):
        base.lookup(unit_vector(0))
    with pytest.raises(NotImplementedError):
        base.store("q", "a", unit_vector(0))
    with pytest.raises(NotImplementedError):
        base.size()
