"""
In-process gateway metrics.

Thread-safe (FastAPI runs sync endpoints in a thread pool) and bounded:
latencies are kept in fixed-size windows, so memory stays flat no matter
how long the process runs. Percentiles come from the window; counters are
lifetime. A multi-replica deployment would export these to Prometheus or
OpenTelemetry instead; the interface would stay the same.
"""

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

LATENCY_WINDOW = 1_000


@dataclass(frozen=True)
class LatencySummary:
    avg: float
    p50: float
    p95: float


def _summarize(samples: deque[float]) -> LatencySummary:
    if not samples:
        return LatencySummary(0.0, 0.0, 0.0)
    arr = np.fromiter(samples, dtype=np.float64)
    return LatencySummary(
        avg=round(float(arr.mean()), 2),
        p50=round(float(np.percentile(arr, 50)), 2),
        p95=round(float(np.percentile(arr, 95)), 2),
    )


class GatewayStats:
    def __init__(self, window: int = LATENCY_WINDOW) -> None:
        self._lock = threading.Lock()
        self._window = window
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.total_queries = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_bypassed = 0
            self.guard_vetoes = 0
            self.llm_errors = 0
            self._hit_ms: deque[float] = deque(maxlen=self._window)
            self._miss_ms: deque[float] = deque(maxlen=self._window)

    def record_hit(self, latency_ms: float, guard_vetoes: int = 0) -> None:
        with self._lock:
            self.total_queries += 1
            self.cache_hits += 1
            self.guard_vetoes += guard_vetoes
            self._hit_ms.append(latency_ms)

    def record_miss(self, latency_ms: float, guard_vetoes: int = 0, bypassed: bool = False) -> None:
        with self._lock:
            self.total_queries += 1
            self.cache_misses += 1
            self.guard_vetoes += guard_vetoes
            self.cache_bypassed += int(bypassed)
            self._miss_ms.append(latency_ms)

    def record_llm_error(self) -> None:
        with self._lock:
            self.total_queries += 1
            self.llm_errors += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            hit = _summarize(self._hit_ms)
            miss = _summarize(self._miss_ms)
            answered = self.cache_hits + self.cache_misses
            return {
                "total_queries": self.total_queries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_bypassed": self.cache_bypassed,
                "llm_errors": self.llm_errors,
                "guard_vetoes": self.guard_vetoes,
                "hit_rate": round(self.cache_hits / answered, 4) if answered else 0.0,
                "estimated_llm_calls_saved": self.cache_hits,
                "avg_hit_latency_ms": hit.avg,
                "p50_hit_latency_ms": hit.p50,
                "p95_hit_latency_ms": hit.p95,
                "avg_miss_latency_ms": miss.avg,
                "p50_miss_latency_ms": miss.p50,
                "p95_miss_latency_ms": miss.p95,
            }


stats = GatewayStats()
