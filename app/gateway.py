"""
The request flow, independent of HTTP:

  1. Embed the query.
  2. Look up the semantic cache. Candidates above the similarity threshold
     must also pass the safety guards (app/guards.py).
       -> HIT:  return the cached answer. No retrieval, no LLM call.
  3. MISS: retrieve top-k knowledge-base chunks and build a grounded prompt.
  4. Call the configured LLM provider.
  5. Store (query, answer) so the next similar question is a hit.

Concurrent identical misses are coalesced ("single-flight"): if ten
requests for the same new question arrive together, one calls the LLM and
the other nine wait and are served from the cache it just filled.
"""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from app.cache import CacheEntry, get_cache
from app.config import settings
from app.embeddings import get_embedder
from app.guards import veto_reason
from app.llm_providers import LLMProviderError, get_llm_provider
from app.rag import retrieve_and_build_prompt
from app.schemas import SourceChunk
from app.stats import stats

CacheStatus = Literal["hit", "miss", "bypass"]


@dataclass
class GatewayResult:
    answer: str
    cache_status: CacheStatus
    provider: str
    latency_ms: float
    matched_query: str | None = None
    similarity: float | None = None
    guard_vetoes: list[str] = field(default_factory=list)
    sources: list[SourceChunk] = field(default_factory=list)

    @property
    def cache_hit(self) -> bool:
        return self.cache_status == "hit"


class _SingleFlight:
    """Per-key locks that are created on demand and dropped when unused."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, tuple[threading.Lock, int]] = {}

    @contextmanager
    def acquire(self, key: str) -> Iterator[bool]:
        """Yields True if another request held this key and we had to wait."""
        with self._guard:
            lock, users = self._locks.get(key, (threading.Lock(), 0))
            self._locks[key] = (lock, users + 1)
        waited = not lock.acquire(blocking=False)
        if waited:
            lock.acquire()
        try:
            yield waited
        finally:
            lock.release()
            with self._guard:
                lock, users = self._locks[key]
                if users == 1:
                    del self._locks[key]
                else:
                    self._locks[key] = (lock, users - 1)


_single_flight = _SingleFlight()


def _vetoes(query: str, rejected: list[CacheEntry]) -> list[str]:
    reasons = (veto_reason(query, e.query) for e in rejected)
    return sorted({r for r in reasons if r})


class _Request:
    """State for one query as it moves through the flow."""

    def __init__(self, query: str, use_cache: bool) -> None:
        self.query = query
        self.use_cache = use_cache
        self.start = time.perf_counter()
        self.cache = get_cache()
        self.vector = get_embedder().encode(query)
        self.similarity: float | None = None
        self.vetoes: list[str] = []
        self.vetoed_candidates = 0

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.start) * 1000, 2)

    def _accept(self, entry: CacheEntry) -> bool:
        return not settings.cache_guards or veto_reason(self.query, entry.query) is None

    def try_cache(self) -> GatewayResult | None:
        lookup = self.cache.lookup(self.vector, accept=self._accept)
        self.similarity = round(lookup.similarity, 4)
        self.vetoes = _vetoes(self.query, lookup.rejected)
        self.vetoed_candidates = len(lookup.rejected)
        if not lookup.hit or lookup.entry is None:
            return None
        latency = self.elapsed_ms()
        stats.record_hit(latency, guard_vetoes=self.vetoed_candidates)
        return GatewayResult(
            answer=lookup.entry.answer,
            cache_status="hit",
            provider="cache",
            latency_ms=latency,
            matched_query=lookup.entry.query,
            similarity=self.similarity,
            guard_vetoes=self.vetoes,
        )

    def generate(self) -> GatewayResult:
        prompt, sources = retrieve_and_build_prompt(self.query, self.vector)
        llm = get_llm_provider()
        try:
            answer = llm.generate(prompt)
        except LLMProviderError:
            stats.record_llm_error()
            raise  # never cache a failure
        self.cache.store(self.query, answer, self.vector)

        latency = self.elapsed_ms()
        stats.record_miss(latency, guard_vetoes=self.vetoed_candidates, bypassed=not self.use_cache)
        return GatewayResult(
            answer=answer,
            cache_status="miss" if self.use_cache else "bypass",
            provider=llm.name,
            latency_ms=latency,
            similarity=self.similarity,
            guard_vetoes=self.vetoes,
            sources=sources,
        )


def handle_query(query: str, use_cache: bool = True) -> GatewayResult:
    req = _Request(query, use_cache)
    if use_cache and (hit := req.try_cache()):
        return hit

    with _single_flight.acquire(query.lower()) as waited:
        # Another request was generating this exact question; its answer is
        # in the cache now, so check again before paying for another call.
        if use_cache and waited and (hit := req.try_cache()):
            return hit
        return req.generate()


def warm_up() -> None:
    """Load the model, knowledge base and provider before the first request,
    so request #1 isn't charged seconds of start-up time (and a bad API key
    or unreachable Redis fails the deploy instead of the first user)."""
    from app.rag import get_knowledge_base

    get_embedder().encode("warm-up")
    get_knowledge_base()
    get_llm_provider()
    get_cache().size()
