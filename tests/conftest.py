"""
Shared fixtures.

Most tests swap the real sentence-transformers model for FakeEmbedder: a
deterministic bag-of-words embedding. It loads instantly, needs no network,
and makes similarity scores predictable — two texts with identical words
score 1.0, texts sharing most words score high, unrelated texts score ~0.

Tests that exercise the real model live in test_integration_real_model.py
and are marked `slow` (run them with `pytest -m slow`).
"""
import hashlib
import re

import numpy as np
import pytest

import app.cache as cache_module
import app.llm_providers as llm_module
import app.main as main_module
import app.rag as rag_module
from app.config import settings
from app.llm_providers import BaseLLMProvider

DIM = 512


def fake_embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeEmbedder:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, text: str) -> np.ndarray:
        self.encode_calls += 1
        return fake_embed(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([fake_embed(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)


class CountingProvider(BaseLLMProvider):
    """Records every prompt it receives so tests can assert on LLM usage."""

    name = "counting"

    def __init__(self, fail: bool = False):
        self.prompts: list[str] = []
        self.fail = fail

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("upstream LLM unavailable")
        return f"answer #{len(self.prompts)}"


def unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    """
    The app keeps several process-wide singletons (cache, LLM provider,
    knowledge base, stats counters). Reset all of them around every test and
    pin settings to known defaults so a developer's local .env can't leak in.
    """
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "redis_url", "")
    monkeypatch.setattr(settings, "cache_similarity_threshold", 0.90)
    monkeypatch.setattr(settings, "rag_top_k", 3)

    def reset():
        cache_module._cache_instance = None
        llm_module._provider_instance = None
        rag_module.get_knowledge_base.cache_clear()
        main_module._stats.update(
            total_queries=0,
            cache_hits=0,
            cache_misses=0,
            hit_latencies_ms=[],
            miss_latencies_ms=[],
        )

    reset()
    yield
    reset()


@pytest.fixture
def fake_embedder(monkeypatch):
    embedder = FakeEmbedder()
    monkeypatch.setattr(main_module, "get_embedder", lambda: embedder)
    monkeypatch.setattr(rag_module, "get_embedder", lambda: embedder)
    return embedder


@pytest.fixture
def llm(monkeypatch):
    provider = CountingProvider()
    monkeypatch.setattr(llm_module, "_provider_instance", provider)
    return provider


@pytest.fixture
def redis_server(monkeypatch):
    """Route redis.from_url to an in-process fakeredis server."""
    fakeredis = pytest.importorskip("fakeredis")
    import redis

    server = fakeredis.FakeServer()

    def from_url(url, **kwargs):
        return fakeredis.FakeRedis(server=server, **kwargs)

    monkeypatch.setattr(redis, "from_url", from_url)
    return server


@pytest.fixture
def client(fake_embedder, llm):
    from fastapi.testclient import TestClient

    return TestClient(main_module.app)
