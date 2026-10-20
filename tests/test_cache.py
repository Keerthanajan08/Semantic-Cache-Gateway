"""
Cache tests. TestCacheContract runs against both backends (in-memory and
Redis via fakeredis), since they promise identical semantics behind
BaseCache. Backend-specific behaviour (LRU, TTL, namespacing) is tested
separately.
"""

import time

import numpy as np
import pytest

from app.cache import BaseCache, InMemoryCache, RedisCache, get_cache
from app.config import settings
from app.embeddings import cosine_similarity
from tests.conftest import unit_vector


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(params=["memory", "redis"])
def make_cache(request):
    if request.param == "memory":
        return lambda threshold=0.9: InMemoryCache(threshold=threshold)
    request.getfixturevalue("redis_server")
    return lambda threshold=0.9: RedisCache("redis://fake", threshold=threshold)


def near(vec: np.ndarray, seed: int, eps: float = 0.05) -> np.ndarray:
    v = vec + eps * unit_vector(seed, dim=vec.shape[0])
    return v / np.linalg.norm(v)


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
        # Computed exactly the way the cache scores (matrix @ vector), so the
        # comparison is at the same float32 value, not a rounding neighbour.
        threshold = float((np.stack([stored]) @ probe)[0])
        cache = make_cache(threshold=threshold)
        cache.store("q", "a", stored)
        assert cache.lookup(probe).hit is True

    def test_similarity_just_below_threshold_is_a_miss(self, make_cache):
        stored, probe = unit_vector(1), unit_vector(2)
        cache = make_cache(threshold=cosine_similarity(probe, stored) + 1e-4)
        cache.store("q", "a", stored)
        assert cache.lookup(probe).hit is False

    def test_returns_the_best_match_not_the_first(self, make_cache):
        target = unit_vector(10)
        cache = make_cache(threshold=0.5)
        cache.store("far", "far answer", unit_vector(12))
        cache.store("near", "near answer", near(target, 11))
        cache.store("exact", "exact answer", target)
        assert cache.lookup(target).entry.answer == "exact answer"

    def test_miss_reports_best_similarity_seen(self, make_cache):
        stored, probe = unit_vector(1), unit_vector(2)
        cache = make_cache(threshold=0.999)
        cache.store("q", "a", stored)
        result = cache.lookup(probe)
        assert result.hit is False
        expected = max(cosine_similarity(probe, stored), 0.0)
        assert result.similarity == pytest.approx(expected, abs=1e-5)

    def test_negative_similarity_is_clamped_to_zero(self, make_cache):
        vec = unit_vector(3)
        cache = make_cache()
        cache.store("q", "a", vec)
        result = cache.lookup(-vec)
        assert result.hit is False
        assert result.similarity == 0.0

    def test_size_and_clear(self, make_cache):
        cache = make_cache()
        assert cache.size() == 0
        cache.store("q1", "a1", unit_vector(1))
        cache.store("q2", "a2", unit_vector(2))
        assert cache.size() == 2
        assert cache.clear() == 2
        assert cache.size() == 0
        assert cache.lookup(unit_vector(1)).hit is False

    def test_same_query_stored_twice_keeps_both(self, make_cache):
        cache = make_cache()
        cache.store("q", "a1", unit_vector(1))
        cache.store("q", "a2", unit_vector(1))
        assert cache.size() == 2

    def test_entry_records_creation_time(self, make_cache):
        cache = make_cache()
        before = time.time()
        cache.store("q", "a", unit_vector(1))
        entry = cache.lookup(unit_vector(1)).entry
        assert before <= entry.created_at <= time.time()

    def test_unicode_round_trips(self, make_cache):
        cache = make_cache()
        cache.store("पासवर्ड कैसे बदलें? 🔑", "उत्तर ✅", unit_vector(1))
        entry = cache.lookup(unit_vector(1)).entry
        assert (entry.query, entry.answer) == ("पासवर्ड कैसे बदलें? 🔑", "उत्तर ✅")


class TestAcceptCallback:
    """The hook the safety guards plug into."""

    def test_vetoed_best_match_falls_through_to_next_best(self, make_cache):
        target = unit_vector(20)
        cache = make_cache(threshold=0.5)
        cache.store("best", "best answer", target)
        cache.store("runner-up", "runner-up answer", near(target, 21))

        result = cache.lookup(target, accept=lambda e: e.query != "best")

        assert result.hit is True
        assert result.entry.answer == "runner-up answer"
        assert [e.query for e in result.rejected] == ["best"]

    def test_all_candidates_vetoed_is_a_miss(self, make_cache):
        vec = unit_vector(30)
        cache = make_cache()
        cache.store("q", "a", vec)

        result = cache.lookup(vec, accept=lambda e: False)

        assert result.hit is False
        assert result.similarity == pytest.approx(1.0, abs=1e-5)
        assert [e.query for e in result.rejected] == ["q"]

    def test_candidates_below_threshold_are_never_offered(self, make_cache):
        offered = []
        cache = make_cache(threshold=0.99)
        cache.store("q", "a", unit_vector(1))
        cache.lookup(unit_vector(2), accept=lambda e: offered.append(e) or True)
        assert offered == []


class TestInMemoryCache:
    def test_entries_expire_after_ttl(self):
        clock = FakeClock()
        cache = InMemoryCache(threshold=0.9, ttl_seconds=60, clock=clock)
        cache.store("q", "a", unit_vector(1))

        clock.now += 59
        assert cache.lookup(unit_vector(1)).hit is True
        clock.now += 1
        assert cache.lookup(unit_vector(1)).hit is False
        assert cache.size() == 0

    def test_ttl_zero_never_expires(self):
        clock = FakeClock()
        cache = InMemoryCache(threshold=0.9, ttl_seconds=0, clock=clock)
        cache.store("q", "a", unit_vector(1))
        clock.now += 10**9
        assert cache.lookup(unit_vector(1)).hit is True

    def test_evicts_least_recently_used_beyond_capacity(self):
        cache = InMemoryCache(threshold=0.9, max_entries=2)
        cache.store("a", "A", unit_vector(1))
        cache.store("b", "B", unit_vector(2))
        cache.lookup(unit_vector(1))  # touch "a": now "b" is least recently used
        cache.store("c", "C", unit_vector(3))

        assert cache.size() == 2
        assert cache.lookup(unit_vector(1)).hit is True
        assert cache.lookup(unit_vector(2)).hit is False
        assert cache.lookup(unit_vector(3)).hit is True

    def test_backend_name(self):
        assert InMemoryCache(threshold=0.9).backend == "memory"


class TestRedisCache:
    def test_entries_survive_a_new_client(self, redis_server):
        RedisCache("redis://fake", threshold=0.9).store("q", "a", unit_vector(1))
        fresh = RedisCache("redis://fake", threshold=0.9)
        assert fresh.size() == 1
        assert fresh.lookup(unit_vector(1)).entry.answer == "a"

    def test_ttl_is_set_on_each_key(self, redis_server):
        cache = RedisCache("redis://fake", threshold=0.9, ttl_seconds=3600)
        cache.store("q", "a", unit_vector(1))
        (key,) = cache.client.scan_iter()
        assert 3590 < cache.client.ttl(key) <= 3600

    def test_no_ttl_when_disabled(self, redis_server):
        cache = RedisCache("redis://fake", threshold=0.9, ttl_seconds=0)
        cache.store("q", "a", unit_vector(1))
        (key,) = cache.client.scan_iter()
        assert cache.client.ttl(key) == -1

    def test_namespaces_are_isolated(self, redis_server):
        """Vectors from different embedding models must never be compared."""
        RedisCache("redis://fake", threshold=0.9, namespace="model-a").store(
            "q", "a", unit_vector(1)
        )
        other = RedisCache("redis://fake", threshold=0.9, namespace="model-b")
        assert other.size() == 0
        assert other.lookup(unit_vector(1)).hit is False

    def test_namespace_is_sanitised_into_the_key(self, redis_server):
        cache = RedisCache("redis://fake", threshold=0.9, namespace="BAAI/bge small*")
        assert cache.key_prefix == "semcache:BAAI_bge_small_:"

    def test_clear_only_touches_own_namespace(self, redis_server):
        mine = RedisCache("redis://fake", threshold=0.9, namespace="mine")
        theirs = RedisCache("redis://fake", threshold=0.9, namespace="theirs")
        mine.store("q", "a", unit_vector(1))
        theirs.store("q", "a", unit_vector(1))
        mine.client.set("unrelated", "value")

        assert mine.clear() == 1
        assert theirs.size() == 1
        assert mine.client.get("unrelated") == "value"

    def test_lookup_fetches_entries_in_one_pipeline(self, redis_server, monkeypatch):
        cache = RedisCache("redis://fake", threshold=0.9)
        for i in range(5):
            cache.store(f"q{i}", f"a{i}", unit_vector(i))

        def per_key_round_trip(*args, **kwargs):
            raise AssertionError("lookup must not issue one HGETALL round trip per key")

        monkeypatch.setattr(cache.client, "hgetall", per_key_round_trip)
        assert cache.lookup(unit_vector(3)).entry.answer == "a3"

    def test_key_expiring_mid_lookup_is_skipped(self, redis_server, monkeypatch):
        cache = RedisCache("redis://fake", threshold=0.9)
        cache.store("q", "a", unit_vector(1))
        real_keys = cache._keys()
        monkeypatch.setattr(cache, "_keys", lambda: [*real_keys, "semcache:default:gone"])
        assert cache.lookup(unit_vector(1)).hit is True

    def test_vector_round_trips_exactly_as_float32(self, redis_server):
        cache = RedisCache("redis://fake", threshold=0.9)
        vec = unit_vector(5, dim=384)
        cache.store("q", "a", vec)
        entry = cache.lookup(vec).entry
        assert entry.vector.dtype == np.float32
        np.testing.assert_array_equal(entry.vector, vec)

    def test_backend_name(self, redis_server):
        assert RedisCache("redis://fake", threshold=0.9).backend == "redis"


class TestGetCache:
    def test_defaults_to_in_memory_with_configured_limits(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_ttl_seconds", 123)
        monkeypatch.setattr(settings, "cache_max_entries", 7)
        cache = get_cache()
        assert isinstance(cache, InMemoryCache)
        assert (cache.ttl_seconds, cache.max_entries) == (123, 7)

    def test_uses_redis_namespaced_by_model_when_url_configured(self, redis_server, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "redis://fake")
        monkeypatch.setattr(settings, "embedding_model", "all-mpnet-base-v2")
        cache = get_cache()
        assert isinstance(cache, RedisCache)
        assert cache.key_prefix == "semcache:all-mpnet-base-v2:"

    def test_uses_resolved_threshold(self, monkeypatch):
        monkeypatch.setattr(settings, "cache_similarity_threshold", 0.42)
        assert get_cache().threshold == 0.42

    def test_is_a_singleton(self):
        assert get_cache() is get_cache()


def test_base_cache_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseCache()  # type: ignore[abstract]
