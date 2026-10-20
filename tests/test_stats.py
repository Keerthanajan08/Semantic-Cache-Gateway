import threading

import pytest

from app.stats import GatewayStats


def test_zero_state():
    snap = GatewayStats().snapshot()
    assert snap["total_queries"] == 0
    assert snap["hit_rate"] == 0.0
    assert snap["p95_hit_latency_ms"] == 0.0


def test_counts_and_rates():
    s = GatewayStats()
    s.record_miss(100.0)
    s.record_hit(10.0)
    s.record_hit(20.0, guard_vetoes=1)
    s.record_miss(120.0, guard_vetoes=2, bypassed=True)

    snap = s.snapshot()
    assert snap["total_queries"] == 4
    assert (snap["cache_hits"], snap["cache_misses"]) == (2, 2)
    assert snap["cache_bypassed"] == 1
    assert snap["guard_vetoes"] == 3
    assert snap["hit_rate"] == 0.5
    assert snap["estimated_llm_calls_saved"] == 2
    assert snap["avg_hit_latency_ms"] == 15.0
    assert snap["avg_miss_latency_ms"] == 110.0


def test_percentiles():
    s = GatewayStats()
    for ms in range(1, 101):
        s.record_hit(float(ms))
    snap = s.snapshot()
    assert snap["p50_hit_latency_ms"] == pytest.approx(50.5)
    assert snap["p95_hit_latency_ms"] == pytest.approx(95.05)


def test_llm_errors_count_as_queries_but_not_in_hit_rate():
    s = GatewayStats()
    s.record_hit(5.0)
    s.record_llm_error()
    snap = s.snapshot()
    assert snap["total_queries"] == 2
    assert snap["llm_errors"] == 1
    assert snap["hit_rate"] == 1.0


def test_latency_window_is_bounded():
    """Memory stays flat however long the process runs."""
    s = GatewayStats(window=10)
    for i in range(1_000):
        s.record_hit(float(i))
    assert len(s._hit_ms) == 10
    assert s.snapshot()["avg_hit_latency_ms"] == pytest.approx(994.5)
    assert s.snapshot()["cache_hits"] == 1_000  # lifetime counters still exact


def test_reset():
    s = GatewayStats()
    s.record_hit(1.0)
    s.reset()
    assert s.snapshot()["total_queries"] == 0


def test_thread_safe_counting():
    s = GatewayStats()

    def worker():
        for _ in range(2_000):
            s.record_hit(1.0)
            s.record_miss(2.0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = s.snapshot()
    assert snap["cache_hits"] == snap["cache_misses"] == 16_000
    assert snap["total_queries"] == 32_000
