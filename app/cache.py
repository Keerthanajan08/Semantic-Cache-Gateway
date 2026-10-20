"""
Semantic cache: embed a query, compare it with every cached query's
embedding, and reuse the stored answer if the best match clears the
similarity threshold.

Two backends behind one interface (BaseCache):

* InMemoryCache: zero setup, resets on restart. Embeddings live in one
  contiguous (N, D) float32 matrix, so a lookup is a single matrix-vector
  product instead of a Python loop. Bounded by TTL and an LRU entry cap.
* RedisCache: persistent and shareable between replicas. Works on any plain
  Redis (no RediSearch module), so free hosted tiers are fine. All entries
  are fetched in ONE pipelined round trip and scored with the same
  vectorised maths. Expiry is native Redis TTL.

Both do exact (brute-force) nearest-neighbour search. At the scale this
project targets (up to ~10k entries) that is a few milliseconds, and it is
exact. Past that, the right move is an ANN index (Redis vector sets, FAISS,
pgvector) behind this same interface; nothing else in the app would change.
"""

from __future__ import annotations

import base64
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from app.config import settings

if TYPE_CHECKING:
    from app.embeddings import Vector

Clock = Callable[[], float]
AcceptFn = Callable[["CacheEntry"], bool]


@dataclass
class CacheEntry:
    query: str
    answer: str
    vector: Vector
    created_at: float


@dataclass
class CacheLookupResult:
    hit: bool
    entry: CacheEntry | None = None
    # Best similarity seen (of the accepted entry on a hit, of any entry on a
    # miss), clamped to [0, 1] for readability.
    similarity: float = 0.0
    # Candidates above the threshold that the `accept` callback vetoed.
    rejected: list[CacheEntry] = field(default_factory=list)


def _select(
    entries: list[CacheEntry],
    scores: npt.NDArray[np.float32],
    threshold: float,
    accept: AcceptFn | None,
) -> tuple[CacheLookupResult, int | None]:
    """
    Pick the best-scoring entry above `threshold` that `accept` allows.
    Returns the result and the chosen entry's index (None on a miss).
    """
    if len(entries) == 0:
        return CacheLookupResult(hit=False), None

    best = float(np.max(scores))
    rejected: list[CacheEntry] = []
    for idx in np.argsort(-scores):
        score = float(scores[idx])
        if score < threshold:
            break
        entry = entries[int(idx)]
        if accept is None or accept(entry):
            hit = CacheLookupResult(hit=True, entry=entry, similarity=score, rejected=rejected)
            return hit, int(idx)
        rejected.append(entry)
    return CacheLookupResult(hit=False, similarity=max(best, 0.0), rejected=rejected), None


class BaseCache(ABC):
    threshold: float

    @abstractmethod
    def lookup(self, vector: Vector, accept: AcceptFn | None = None) -> CacheLookupResult:
        """Best entry with similarity >= threshold for which accept(entry) is true."""

    @abstractmethod
    def store(self, query: str, answer: str, vector: Vector) -> None: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def clear(self) -> int:
        """Remove every entry; returns how many were removed."""

    @property
    @abstractmethod
    def backend(self) -> str: ...


class InMemoryCache(BaseCache):
    def __init__(
        self,
        threshold: float,
        ttl_seconds: int = 0,
        max_entries: int = 10_000,
        clock: Clock = time.time,
    ) -> None:
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.RLock()
        # Insertion/recency order for LRU eviction (most recently used last).
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        self._next_id = 0
        # Matrix rows are rebuilt only when the set of entries changes, not on
        # every hit, so LRU bookkeeping stays O(1).
        self._row_ids: list[int] = []
        self._matrix: npt.NDArray[np.float32] | None = None

    @property
    def backend(self) -> str:
        return "memory"

    def _expire(self) -> None:
        if not self.ttl_seconds:
            return
        cutoff = self._clock() - self.ttl_seconds
        expired = [k for k, e in self._entries.items() if e.created_at <= cutoff]
        for key in expired:
            del self._entries[key]
        if expired:
            self._matrix = None

    def _rows(self) -> tuple[list[int], npt.NDArray[np.float32]]:
        if self._matrix is None:
            self._row_ids = list(self._entries)
            self._matrix = (
                np.stack([self._entries[k].vector for k in self._row_ids])
                if self._row_ids
                else np.zeros((0, 0), dtype=np.float32)
            )
        return self._row_ids, self._matrix

    def lookup(self, vector: Vector, accept: AcceptFn | None = None) -> CacheLookupResult:
        with self._lock:
            self._expire()
            row_ids, matrix = self._rows()
            if not row_ids:
                return CacheLookupResult(hit=False)
            scores = matrix @ vector
            entries = [self._entries[k] for k in row_ids]
            result, idx = _select(entries, scores, self.threshold, accept)
            if idx is not None:
                self._entries.move_to_end(row_ids[idx])  # mark as recently used
            return result

    def store(self, query: str, answer: str, vector: Vector) -> None:
        with self._lock:
            self._entries[self._next_id] = CacheEntry(query, answer, vector, self._clock())
            self._next_id += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)  # least recently used
            self._matrix = None

    def size(self) -> int:
        with self._lock:
            self._expire()
            return len(self._entries)

    def clear(self) -> int:
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._matrix = None
            return removed


class RedisCache(BaseCache):
    """
    Entries are Redis hashes under `semcache:<namespace>:<uuid>`. The
    namespace is the embedding model name: vectors from different models
    live in different spaces and must never be compared, so swapping
    EMBEDDING_MODEL can't serve answers matched in the wrong space.
    """

    def __init__(
        self,
        redis_url: str,
        threshold: float,
        ttl_seconds: int = 0,
        namespace: str = "default",
        client: Any = None,
    ) -> None:
        if client is None:
            import redis  # lazy: not needed for the in-memory backend

            client = redis.from_url(redis_url, decode_responses=True)
        self.client = client
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        safe_ns = re.sub(r"[^A-Za-z0-9._-]", "_", namespace)
        self.key_prefix = f"semcache:{safe_ns}:"

    @property
    def backend(self) -> str:
        return "redis"

    @staticmethod
    def _encode_vector(vector: Vector) -> str:
        return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")

    @staticmethod
    def _decode_vector(raw: str) -> Vector:
        return np.frombuffer(base64.b64decode(raw), dtype=np.float32).copy()

    def _keys(self) -> list[str]:
        return list(self.client.scan_iter(match=f"{self.key_prefix}*", count=500))

    def _load_entries(self) -> list[CacheEntry]:
        keys = self._keys()
        if not keys:
            return []
        pipe = self.client.pipeline(transaction=False)
        for key in keys:
            pipe.hgetall(key)
        entries = []
        for raw in pipe.execute():  # one network round trip for every entry
            if not raw:  # expired between SCAN and HGETALL
                continue
            entries.append(
                CacheEntry(
                    query=raw["query"],
                    answer=raw["answer"],
                    vector=self._decode_vector(raw["vector"]),
                    created_at=float(raw["created_at"]),
                )
            )
        return entries

    def lookup(self, vector: Vector, accept: AcceptFn | None = None) -> CacheLookupResult:
        entries = self._load_entries()
        if not entries:
            return CacheLookupResult(hit=False)
        scores = np.stack([e.vector for e in entries]) @ vector
        result, _ = _select(entries, scores, self.threshold, accept)
        return result

    def store(self, query: str, answer: str, vector: Vector) -> None:
        key = f"{self.key_prefix}{uuid.uuid4().hex}"
        pipe = self.client.pipeline(transaction=True)
        pipe.hset(
            key,
            mapping={
                "query": query,
                "answer": answer,
                "vector": self._encode_vector(vector),
                "created_at": time.time(),
            },
        )
        if self.ttl_seconds:
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def size(self) -> int:
        return len(self._keys())

    def clear(self) -> int:
        keys = self._keys()
        if keys:
            self.client.delete(*keys)
        return len(keys)


_cache_instance: BaseCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> BaseCache:
    global _cache_instance
    with _cache_lock:
        if _cache_instance is None:
            if settings.redis_url:
                _cache_instance = RedisCache(
                    settings.redis_url,
                    threshold=settings.similarity_threshold,
                    ttl_seconds=settings.cache_ttl_seconds,
                    namespace=settings.embedding_model,
                )
            else:
                _cache_instance = InMemoryCache(
                    threshold=settings.similarity_threshold,
                    ttl_seconds=settings.cache_ttl_seconds,
                    max_entries=settings.cache_max_entries,
                )
        return _cache_instance
