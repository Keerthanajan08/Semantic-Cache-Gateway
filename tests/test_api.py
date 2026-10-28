"""
HTTP API tests through FastAPI's TestClient, with the embedder and LLM
replaced by fast deterministic fakes (see conftest.py).
"""

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.cache as cache_module
import app.llm_providers as llm_module
import app.main as main_module
from app.config import settings
from app.main import app
from tests.conftest import CountingProvider

PASSWORD_Q = "how do i reset my password"
PASSWORD_PARAPHRASE = "how do i reset my password please"  # fake similarity ~0.93
CANCEL_Q = "how do i cancel my subscription"  # fake similarity to PASSWORD_Q ~0.67


def ask(client, query, **extra):
    response = client.post("/query", json={"query": query, **extra})
    assert response.status_code == 200, response.text
    return response.json()


class TestHealth:
    def test_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["llm_provider"] == "mock"
        assert body["cache_backend"] == "memory"
        assert body["cache_reachable"] is True
        assert body["version"]

    def test_reports_degraded_when_cache_unreachable(self, client, monkeypatch):
        cache = cache_module.get_cache()

        def boom():
            raise ConnectionError("redis down")

        monkeypatch.setattr(cache, "size", boom)
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["cache_reachable"] is False

    def test_root_redirects_to_docs_without_a_dashboard_build(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main_module, "UI_DIR", tmp_path)
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/docs"

    def test_root_redirects_to_dashboard_when_built(self, client, monkeypatch, tmp_path):
        (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
        monkeypatch.setattr(main_module, "UI_DIR", tmp_path)
        response = client.get("/", follow_redirects=False)
        assert response.headers["location"] == "/ui/"


class TestDashboard:
    def test_serves_a_build(self, tmp_path):
        (tmp_path / "index.html").write_text("<h1>dashboard</h1>", encoding="utf-8")
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        target = FastAPI()
        assert main_module.mount_ui(target, tmp_path) is True
        ui = TestClient(target)
        assert "dashboard" in ui.get("/ui/").text
        assert ui.get("/ui/assets/app.js").status_code == 200

    def test_skips_mounting_without_a_build(self, tmp_path):
        target = FastAPI()
        assert main_module.mount_ui(target, tmp_path) is False
        assert TestClient(target).get("/ui/").status_code == 404


class TestQuery:
    def test_miss_answers_from_llm_with_sources(self, client, llm):
        body = ask(client, PASSWORD_Q)
        assert body["cache_status"] == "miss"
        assert body["cache_hit"] is False
        assert body["provider"] == "counting"
        assert body["answer"] == "answer #1"
        assert len(body["sources"]) == settings.rag_top_k
        for s in body["sources"]:
            assert re.fullmatch(r"(account_and_auth|billing)\.md#\d+", s["doc_id"])
        assert len(llm.prompts) == 1

    def test_llm_receives_grounded_prompt(self, client, llm):
        body = ask(client, PASSWORD_Q)
        prompt = llm.prompts[0]
        assert f"Question: {PASSWORD_Q}" in prompt
        for source in body["sources"]:
            assert f"[{source['doc_id']}] {source['text']}" in prompt

    def test_repeat_is_a_hit_without_llm_call(self, client, llm):
        first = ask(client, PASSWORD_Q)
        second = ask(client, PASSWORD_Q)
        assert second["cache_status"] == "hit"
        assert second["provider"] == "cache"
        assert second["answer"] == first["answer"]
        assert second["matched_query"] == PASSWORD_Q
        assert second["similarity"] == pytest.approx(1.0, abs=1e-3)
        assert second["sources"] == []
        assert len(llm.prompts) == 1

    def test_paraphrase_above_threshold_hits(self, client, llm):
        ask(client, PASSWORD_Q)
        body = ask(client, PASSWORD_PARAPHRASE)
        assert body["cache_status"] == "hit"
        assert body["similarity"] >= settings.similarity_threshold
        assert len(llm.prompts) == 1

    def test_unrelated_question_misses(self, client, llm):
        ask(client, PASSWORD_Q)
        body = ask(client, CANCEL_Q)
        assert body["cache_status"] == "miss"
        assert 0 < body["similarity"] < settings.similarity_threshold

    @pytest.mark.parametrize(("threshold", "expect_hit"), [(0.95, False), (0.60, True)])
    def test_threshold_controls_hit_or_miss(self, client, monkeypatch, threshold, expect_hit):
        monkeypatch.setattr(settings, "cache_similarity_threshold", threshold)
        ask(client, PASSWORD_Q)
        probe = PASSWORD_PARAPHRASE if threshold > 0.9 else CANCEL_Q
        assert ask(client, probe)["cache_hit"] is expect_hit

    def test_negated_question_is_vetoed(self, client, llm):
        ask(client, PASSWORD_Q)
        body = ask(client, "how do i not reset my password")
        assert body["cache_status"] == "miss"
        assert body["guard_vetoes"] == ["negation"]

    def test_use_cache_false_bypasses(self, client, llm):
        ask(client, PASSWORD_Q)
        body = ask(client, PASSWORD_Q, use_cache=False)
        assert body["cache_status"] == "bypass"
        assert len(llm.prompts) == 2

    def test_whitespace_is_normalised(self, client, llm):
        ask(client, PASSWORD_Q)
        body = ask(client, "  how do i   reset\tmy password \n")
        assert body["cache_status"] == "hit"
        assert body["similarity"] == pytest.approx(1.0, abs=1e-3)

    def test_headers(self, client):
        miss = client.post("/query", json={"query": PASSWORD_Q})
        hit = client.post("/query", json={"query": PASSWORD_Q})
        assert miss.headers["X-Cache"] == "MISS"
        assert hit.headers["X-Cache"] == "HIT"
        assert float(hit.headers["X-Response-Time-Ms"]) > 0
        assert re.fullmatch(r"[0-9a-f]{32}", hit.headers["X-Request-ID"])
        assert hit.json()["request_id"] == hit.headers["X-Request-ID"]

    def test_propagates_caller_request_id(self, client):
        response = client.post(
            "/query", json={"query": PASSWORD_Q}, headers={"X-Request-ID": "trace-abc-123"}
        )
        assert response.headers["X-Request-ID"] == "trace-abc-123"
        assert response.json()["request_id"] == "trace-abc-123"

    def test_works_with_redis_backend(self, client, llm, redis_server, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "redis://fake")
        assert ask(client, PASSWORD_Q)["cache_status"] == "miss"
        cache_module._cache_instance = None  # simulate a restart: Redis persists
        assert ask(client, PASSWORD_Q)["cache_status"] == "hit"
        assert len(llm.prompts) == 1


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"query": ""},
            {"query": "   "},
            {"query": "\n\t"},
            {"query": None},
            {"query": 42},
            {"q": "wrong field"},
            {"query": "ok", "use_cache": "maybe"},
        ],
        ids=[
            "missing",
            "empty",
            "spaces",
            "newline-tab",
            "null",
            "number",
            "wrong-field",
            "bad-flag",
        ],
    )
    def test_rejects_invalid_payloads(self, client, llm, payload):
        response = client.post("/query", json=payload)
        assert response.status_code == 422
        assert llm.prompts == []

    def test_rejects_overlong_query(self, client, llm, monkeypatch):
        monkeypatch.setattr(settings, "max_query_chars", 20)
        response = client.post("/query", json={"query": "x" * 21})
        assert response.status_code == 422
        assert "at most 20" in response.text

    def test_rejects_non_json_body(self, client):
        response = client.post(
            "/query", content="not json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 422

    def test_invalid_requests_are_not_counted(self, client):
        client.post("/query", json={"query": "  "})
        assert client.get("/stats").json()["total_queries"] == 0


class TestErrors:
    def test_llm_failure_is_a_clean_502(self, fake_embedder, monkeypatch):
        monkeypatch.setattr(llm_module, "_provider_instance", CountingProvider(fail=True))
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/query", json={"query": PASSWORD_Q}, headers={"X-Request-ID": "req-1"}
        )

        assert response.status_code == 502
        assert response.json() == {
            "detail": "Upstream LLM provider failed.",
            "request_id": "req-1",
        }
        assert cache_module.get_cache().size() == 0

    def test_misconfigured_provider_is_a_503(self, fake_embedder, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "")
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/query", json={"query": PASSWORD_Q})

        assert response.status_code == 503
        assert "GROQ_API_KEY" in response.json()["detail"]


class TestStats:
    def test_zero_state(self, client):
        body = client.get("/stats").json()
        assert body["total_queries"] == 0
        assert body["hit_rate"] == 0.0
        assert body["cache_size"] == 0
        assert body["cache_backend"] == "memory"
        assert body["similarity_threshold"] == 0.90

    def test_counts_hits_misses_and_size(self, client):
        ask(client, PASSWORD_Q)  # miss
        ask(client, PASSWORD_Q)  # hit
        ask(client, PASSWORD_PARAPHRASE)  # hit
        ask(client, CANCEL_Q)  # miss

        body = client.get("/stats").json()
        assert (body["total_queries"], body["cache_hits"], body["cache_misses"]) == (4, 2, 2)
        assert body["hit_rate"] == 0.5
        assert body["estimated_llm_calls_saved"] == 2
        assert body["cache_size"] == 2
        assert body["p95_hit_latency_ms"] >= body["p50_hit_latency_ms"] > 0

    def test_hits_are_faster_than_misses_with_a_slow_llm(self, fake_embedder, monkeypatch):
        monkeypatch.setattr(llm_module, "_provider_instance", CountingProvider(delay_s=0.05))
        client = TestClient(app)
        ask(client, PASSWORD_Q)
        ask(client, PASSWORD_Q)
        body = client.get("/stats").json()
        assert body["avg_hit_latency_ms"] < body["avg_miss_latency_ms"]
        assert body["avg_miss_latency_ms"] >= 50


class TestClearCache:
    def test_disabled_without_admin_token(self, client):
        assert client.delete("/cache").status_code == 403

    def test_rejects_wrong_token(self, client, monkeypatch):
        monkeypatch.setattr(settings, "admin_token", "s3cret")
        assert client.delete("/cache", headers={"X-Admin-Token": "nope"}).status_code == 401
        assert client.delete("/cache").status_code == 401

    def test_clears_with_correct_token(self, client, llm, monkeypatch):
        monkeypatch.setattr(settings, "admin_token", "s3cret")
        ask(client, PASSWORD_Q)
        ask(client, CANCEL_Q)

        response = client.delete("/cache", headers={"X-Admin-Token": "s3cret"})

        assert response.json() == {"removed": 2}
        assert ask(client, PASSWORD_Q)["cache_status"] == "miss"


class TestLifespan:
    def test_warms_up_on_startup(self, fake_embedder, llm, monkeypatch):
        monkeypatch.setattr(settings, "warmup_on_startup", True)
        with TestClient(app):
            assert fake_embedder.encode_calls == 1  # warmed before any request

    def test_bad_provider_config_fails_startup(self, fake_embedder, monkeypatch):
        """A missing API key should fail the deploy, not the first user."""
        monkeypatch.setattr(settings, "warmup_on_startup", True)
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        monkeypatch.setattr(settings, "gemini_api_key", "")
        with pytest.raises(llm_module.LLMConfigurationError), TestClient(app):
            pass


def test_openapi_schema_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/query", "/stats", "/cache"} <= paths.keys()
