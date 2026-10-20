"""
The request flow (app/gateway.py) without HTTP: cache hits/misses, guards,
bypass, failure handling and request coalescing.
"""

import threading

import pytest

import app.llm_providers as llm_module
from app.cache import get_cache
from app.config import settings
from app.gateway import handle_query, warm_up
from app.llm_providers import LLMProviderError
from app.stats import stats
from tests.conftest import CountingProvider

pytestmark = pytest.mark.usefixtures("fake_embedder")

Q = "how do i reset my password"


def test_miss_then_hit(llm):
    miss = handle_query(Q)
    hit = handle_query(Q)

    assert (miss.cache_status, hit.cache_status) == ("miss", "hit")
    assert miss.cache_hit is False
    assert hit.cache_hit is True
    assert hit.answer == miss.answer
    assert hit.provider == "cache"
    assert hit.matched_query == Q
    assert len(llm.prompts) == 1


def test_guard_veto_forces_a_miss_and_is_reported(llm):
    handle_query("how do i reset my password")
    result = handle_query("how do i not reset my password")  # fake similarity ~0.93

    assert result.cache_status == "miss"
    assert result.guard_vetoes == ["negation"]
    assert result.similarity >= settings.similarity_threshold
    assert len(llm.prompts) == 2
    assert stats.snapshot()["guard_vetoes"] == 1


def test_guards_can_be_disabled(llm, monkeypatch):
    monkeypatch.setattr(settings, "cache_guards", False)
    handle_query("how do i reset my password")
    result = handle_query("how do i not reset my password")
    assert result.cache_status == "hit"
    assert result.guard_vetoes == []


def test_bypass_skips_lookup_but_refreshes_cache(llm):
    handle_query(Q)
    result = handle_query(Q, use_cache=False)

    assert result.cache_status == "bypass"
    assert result.provider == "counting"
    assert result.answer == "answer #2"
    assert len(llm.prompts) == 2
    assert get_cache().size() == 2
    assert stats.snapshot()["cache_bypassed"] == 1


def test_llm_failure_is_not_cached_and_is_counted(monkeypatch):
    monkeypatch.setattr(llm_module, "_provider_instance", CountingProvider(fail=True))
    with pytest.raises(LLMProviderError):
        handle_query(Q)
    assert get_cache().size() == 0
    snap = stats.snapshot()
    assert (snap["llm_errors"], snap["cache_misses"]) == (1, 0)


def test_failure_then_recovery(monkeypatch):
    failing = CountingProvider(fail=True)
    monkeypatch.setattr(llm_module, "_provider_instance", failing)
    with pytest.raises(LLMProviderError):
        handle_query(Q)

    failing.fail = False
    assert handle_query(Q).cache_status == "miss"
    assert handle_query(Q).cache_status == "hit"


def test_concurrent_identical_misses_call_the_llm_once(monkeypatch):
    """Single-flight: a burst of the same new question costs one LLM call."""
    slow = CountingProvider(delay_s=0.2)
    monkeypatch.setattr(llm_module, "_provider_instance", slow)
    results = []
    barrier = threading.Barrier(8)

    def ask():
        barrier.wait()
        results.append(handle_query(Q))

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(slow.prompts) == 1
    assert sorted(r.cache_status for r in results) == ["hit"] * 7 + ["miss"]
    assert len({r.answer for r in results}) == 1


def test_coalescing_is_per_question(monkeypatch):
    slow = CountingProvider(delay_s=0.1)
    monkeypatch.setattr(llm_module, "_provider_instance", slow)
    questions = ["reset password", "cancel subscription", "update card"]
    threads = [threading.Thread(target=handle_query, args=(q,)) for q in questions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(slow.prompts) == 3


def test_concurrent_bypass_requests_each_call_the_llm(monkeypatch):
    slow = CountingProvider(delay_s=0.05)
    monkeypatch.setattr(llm_module, "_provider_instance", slow)
    threads = [
        threading.Thread(target=handle_query, args=(Q,), kwargs={"use_cache": False})
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(slow.prompts) == 3


def test_warm_up_loads_everything(llm, fake_embedder):
    from app.rag import get_knowledge_base

    warm_up()
    assert fake_embedder.encode_calls == 1
    assert get_knowledge_base.cache_info().currsize == 1
