"""
End-to-end tests of the HTTP API through FastAPI's TestClient, with the
embedder and LLM replaced by fast deterministic fakes (see conftest.py).
"""
import re

import pytest
from fastapi.testclient import TestClient

import app.cache as cache_module
import app.llm_providers as llm_module
from app.config import settings
from app.main import app
from tests.conftest import CountingProvider

PASSWORD_Q = "how do i reset my password"
PASSWORD_PARAPHRASE = "how do i reset my password please"  # fake similarity ~0.93
CANCEL_Q = "how do i cancel my subscription"  # fake similarity to PASSWORD_Q ~0.67


def ask(client, query):
    response = client.post("/query", json={"query": query})
    assert response.status_code == 200, response.text
    return response.json()


class TestHealth:
    def test_reports_ok_and_provider(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "llm_provider": "mock"}

    def test_reflects_configured_provider(self, client, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        assert client.get("/health").json()["llm_provider"] == "groq"


class TestQueryCacheMiss:
    def test_first_query_is_a_miss_answered_by_llm(self, client, llm):
        body = ask(client, PASSWORD_Q)
        assert body["cache_hit"] is False
        assert body["provider"] == "counting"
        assert body["answer"] == "answer #1"
        assert body["matched_query"] is None
        assert len(llm.prompts) == 1

    def test_miss_returns_rag_sources(self, client):
        body = ask(client, PASSWORD_Q)
        sources = body["sources"]
        assert len(sources) == settings.rag_top_k
        for s in sources:
            assert re.fullmatch(r"(account_and_auth|billing)\.md#\d+", s["doc_id"])
            assert s["text"]
        scores = [s["score"] for s in sources]
        assert scores == sorted(scores, reverse=True)

    def test_llm_receives_augmented_prompt(self, client, llm):
        body = ask(client, PASSWORD_Q)
        prompt = llm.prompts[0]
        assert f"Question: {PASSWORD_Q}" in prompt
        for source in body["sources"]:
            assert f"[{source['doc_id']}] {source['text']}" in prompt

    def test_miss_stores_answer_in_cache(self, client):
        ask(client, PASSWORD_Q)
        assert cache_module.get_cache().size() == 1

    def test_unrelated_query_is_a_miss(self, client, llm):
        ask(client, PASSWORD_Q)
        body = ask(client, CANCEL_Q)
        assert body["cache_hit"] is False
        assert 0 < body["similarity"] < settings.cache_similarity_threshold
        assert len(llm.prompts) == 2

    def test_latency_is_reported(self, client):
        assert ask(client, PASSWORD_Q)["latency_ms"] > 0


class TestQueryCacheHit:
    def test_repeat_query_is_a_hit_without_llm_call(self, client, llm):
        first = ask(client, PASSWORD_Q)
        second = ask(client, PASSWORD_Q)

        assert second["cache_hit"] is True
        assert second["provider"] == "cache"
        assert second["answer"] == first["answer"]
        assert second["matched_query"] == PASSWORD_Q
        assert second["similarity"] == pytest.approx(1.0, abs=1e-3)
        assert second["sources"] == []
        assert len(llm.prompts) == 1

    def test_paraphrase_above_threshold_is_a_hit(self, client, llm):
        ask(client, PASSWORD_Q)
        body = ask(client, PASSWORD_PARAPHRASE)
        assert body["cache_hit"] is True
        assert body["matched_query"] == PASSWORD_Q
        assert body["similarity"] >= settings.cache_similarity_threshold
        assert len(llm.prompts) == 1

    def test_hit_does_not_grow_the_cache(self, client):
        ask(client, PASSWORD_Q)
        ask(client, PASSWORD_Q)
        ask(client, PASSWORD_PARAPHRASE)
        assert cache_module.get_cache().size() == 1

    @pytest.mark.parametrize("threshold, expect_hit", [(0.95, False), (0.60, True)])
    def test_threshold_controls_hit_or_miss(self, client, monkeypatch, threshold, expect_hit):
        monkeypatch.setattr(settings, "cache_similarity_threshold", threshold)
        ask(client, PASSWORD_Q)
        # ~0.93 for the paraphrase; ~0.67 for CANCEL_Q
        probe = PASSWORD_PARAPHRASE if threshold > 0.9 else CANCEL_Q
        assert ask(client, probe)["cache_hit"] is expect_hit

    def test_embeds_each_query_exactly_once(self, client, fake_embedder):
        ask(client, PASSWORD_Q)  # miss (KB load uses encode_batch, not encode)
        ask(client, PASSWORD_Q)  # hit
        assert fake_embedder.encode_calls == 2


class TestQueryWithRedisBackend:
    def test_hit_and_miss_flow_through_redis(self, client, llm, redis_server, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "redis://fake")
        assert ask(client, PASSWORD_Q)["cache_hit"] is False
        assert ask(client, PASSWORD_Q)["cache_hit"] is True
        assert isinstance(cache_module.get_cache(), cache_module.RedisCache)
        assert len(llm.prompts) == 1

    def test_cache_survives_gateway_restart(self, client, llm, redis_server, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "redis://fake")
        ask(client, PASSWORD_Q)
        cache_module._cache_instance = None  # simulate a process restart
        assert ask(client, PASSWORD_Q)["cache_hit"] is True


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [{}, {"query": ""}, {"query": None}, {"q": "wrong field"}],
        ids=["missing", "empty", "null", "wrong-field"],
    )
    def test_rejects_invalid_payloads(self, client, llm, payload):
        response = client.post("/query", json=payload)
        assert response.status_code == 422
        assert llm.prompts == []

    def test_rejects_non_json_body(self, client):
        response = client.post("/query", content="not json", headers={"content-type": "application/json"})
        assert response.status_code == 422

    def test_get_on_query_not_allowed(self, client):
        assert client.get("/query").status_code == 405


class TestLLMFailure:
    def test_llm_error_returns_500_and_caches_nothing(self, fake_embedder, monkeypatch):
        monkeypatch.setattr(llm_module, "_provider_instance", CountingProvider(fail=True))
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/query", json={"query": PASSWORD_Q})

        assert response.status_code == 500
        assert cache_module.get_cache().size() == 0


class TestStats:
    def test_zero_state(self, client):
        assert client.get("/stats").json() == {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_rate": 0.0,
            "estimated_llm_calls_saved": 0,
            "avg_hit_latency_ms": 0.0,
            "avg_miss_latency_ms": 0.0,
        }

    def test_counts_hits_and_misses(self, client):
        ask(client, PASSWORD_Q)  # miss
        ask(client, PASSWORD_Q)  # hit
        ask(client, PASSWORD_PARAPHRASE)  # hit
        ask(client, CANCEL_Q)  # miss

        stats = client.get("/stats").json()
        assert stats["total_queries"] == 4
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["estimated_llm_calls_saved"] == 2
        assert stats["avg_hit_latency_ms"] > 0
        assert stats["avg_miss_latency_ms"] > 0

    def test_hit_rate_is_rounded(self, client):
        ask(client, PASSWORD_Q)
        ask(client, PASSWORD_Q)
        ask(client, CANCEL_Q)
        assert client.get("/stats").json()["hit_rate"] == 0.3333

    def test_invalid_requests_are_not_counted(self, client):
        client.post("/query", json={"query": ""})
        assert client.get("/stats").json()["total_queries"] == 0

    def test_hits_are_faster_than_misses_with_slow_llm(self, fake_embedder, monkeypatch):
        """The whole point of the gateway, measured against a real MockProvider (50ms)."""
        monkeypatch.setattr(llm_module, "_provider_instance", llm_module.MockProvider())
        client = TestClient(app)
        ask(client, PASSWORD_Q)
        ask(client, PASSWORD_Q)
        stats = client.get("/stats").json()
        assert stats["avg_hit_latency_ms"] < stats["avg_miss_latency_ms"]
        assert stats["avg_miss_latency_ms"] >= 50
