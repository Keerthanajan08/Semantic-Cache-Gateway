"""
Semantic cache: on a query, embed it and compare against every previously
cached query's embedding. If the closest one is above the similarity
threshold, reuse its answer instead of calling the LLM.

Two backends behind one interface:
  - InMemoryCache: a Python dict. Zero setup, resets on restart. Perfect
    for local dev and for running this project with no external services.
  - RedisCache: persists entries as Redis hashes (query, answer, vector
    stored as JSON). Vector search is done in Python at read time — this
    avoids requiring the RediSearch module, which keeps it compatible with
    any plain Redis instance (including free hosted tiers).

At the scale a portfolio/demo project runs at (hundreds to low thousands
of cached entries), scanning in Python is fast enough that it isn't worth
the added complexity of an ANN index. That tradeoff is worth being able to
explain, not hidden.
"""
import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import settings
from app.embeddings import cosine_similarity


@dataclass
class CacheEntry:
    query: str
    answer: str
    vector: np.ndarray
    created_at: float


@dataclass
class CacheLookupResult:
    hit: bool
    entry: Optional[CacheEntry] = None
    similarity: float = 0.0


class BaseCache:
    def lookup(self, vector: np.ndarray) -> CacheLookupResult:
        raise NotImplementedError

    def store(self, query: str, answer: str, vector: np.ndarray) -> None:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError


class InMemoryCache(BaseCache):
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._entries: list[CacheEntry] = []

    def lookup(self, vector: np.ndarray) -> CacheLookupResult:
        best_entry, best_score = None, -1.0
        for entry in self._entries:
            score = cosine_similarity(vector, entry.vector)
            if score > best_score:
                best_entry, best_score = entry, score

        if best_entry is not None and best_score >= self.threshold:
            return CacheLookupResult(hit=True, entry=best_entry, similarity=best_score)
        return CacheLookupResult(hit=False, similarity=max(best_score, 0.0))

    def store(self, query: str, answer: str, vector: np.ndarray) -> None:
        self._entries.append(CacheEntry(query, answer, vector, time.time()))

    def size(self) -> int:
        return len(self._entries)


class RedisCache(BaseCache):
    """Same semantics as InMemoryCache, backed by Redis hashes."""

    KEY_PREFIX = "semcache:"

    def __init__(self, redis_url: str, threshold: float):
        import redis  # imported lazily so redis isn't required for in-memory mode

        self.client = redis.from_url(redis_url, decode_responses=True)
        self.threshold = threshold

    def _all_keys(self):
        return self.client.scan_iter(match=f"{self.KEY_PREFIX}*")

    def lookup(self, vector: np.ndarray) -> CacheLookupResult:
        best_entry, best_score = None, -1.0
        for key in self._all_keys():
            raw = self.client.hgetall(key)
            if not raw:
                continue
            cached_vec = np.array(json.loads(raw["vector"]), dtype=np.float32)
            score = cosine_similarity(vector, cached_vec)
            if score > best_score:
                best_entry = CacheEntry(
                    query=raw["query"],
                    answer=raw["answer"],
                    vector=cached_vec,
                    created_at=float(raw["created_at"]),
                )
                best_score = score

        if best_entry is not None and best_score >= self.threshold:
            return CacheLookupResult(hit=True, entry=best_entry, similarity=best_score)
        return CacheLookupResult(hit=False, similarity=max(best_score, 0.0))

    def store(self, query: str, answer: str, vector: np.ndarray) -> None:
        key = f"{self.KEY_PREFIX}{int(time.time() * 1000)}:{hash(query) & 0xffffffff}"
        self.client.hset(
            key,
            mapping={
                "query": query,
                "answer": answer,
                "vector": json.dumps(vector.tolist()),
                "created_at": time.time(),
            },
        )

    def size(self) -> int:
        return sum(1 for _ in self._all_keys())


_cache_instance: Optional[BaseCache] = None


def get_cache() -> BaseCache:
    global _cache_instance
    if _cache_instance is None:
        if settings.redis_url:
            _cache_instance = RedisCache(settings.redis_url, settings.cache_similarity_threshold)
        else:
            _cache_instance = InMemoryCache(settings.cache_similarity_threshold)
    return _cache_instance
