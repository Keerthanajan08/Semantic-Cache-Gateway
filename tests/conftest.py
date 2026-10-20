"""
Shared fixtures.

Most tests swap the real embedding model for FakeEmbedder: a deterministic
bag-of-words embedding. It loads instantly, needs no network, and makes
similarity predictable: identical word sets score 1.0, texts sharing most
words score high, unrelated texts score ~0.

Tests that exercise the real model live in test_integration_real_model.py
and are marked `slow` (run them with `pytest -m slow`).
"""

import hashlib
import re
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.cache as cache_module
import app.gateway as gateway_module
import app.llm_providers as llm_module
import app.rag as rag_module
from app.config import settings
from app.llm_providers import BaseLLMProvider, LLMProviderError
from app.stats import stats

DIM = 512


def fake_embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeEmbedder:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text: str) -> np.ndarray:
        self.encode_calls += 1
        return fake_embed(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([fake_embed(t) for t in texts])


class CountingProvider(BaseLLMProvider):
    """Records every prompt so tests can assert exactly when the LLM is called."""

    name = "counting"

    def __init__(self, fail: bool = False, delay_s: float = 0.0) -> None:
        self.prompts: list[str] = []
        self.fail = fail
        self.delay_s = delay_s

    def generate(self, prompt: str) -> str:
        import time

        self.prompts.append(prompt)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise LLMProviderError("upstream LLM unavailable")
        return f"answer #{len(self.prompts)}"


def unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    The app keeps process-wide singletons (cache, provider, knowledge base,
    stats). Reset them around every test and pin settings to known values so
    a developer's local .env can't leak in.
    """
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "redis_url", "")
    monkeypatch.setattr(settings, "cache_similarity_threshold", 0.90)
    monkeypatch.setattr(settings, "cache_ttl_seconds", 0)
    monkeypatch.setattr(settings, "cache_max_entries", 10_000)
    monkeypatch.setattr(settings, "cache_guards", True)
    monkeypatch.setattr(settings, "rag_top_k", 3)
    monkeypatch.setattr(settings, "warmup_on_startup", False)
    monkeypatch.setattr(settings, "admin_token", "")

    def reset() -> None:
        cache_module._cache_instance = None
        llm_module._provider_instance = None
        rag_module.get_knowledge_base.cache_clear()
        stats.reset()

    reset()
    yield
    reset()


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    embedder = FakeEmbedder()
    monkeypatch.setattr(gateway_module, "get_embedder", lambda: embedder)
    monkeypatch.setattr(rag_module, "get_embedder", lambda: embedder)
    return embedder


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> CountingProvider:
    provider = CountingProvider()
    monkeypatch.setattr(llm_module, "_provider_instance", provider)
    return provider


@pytest.fixture
def redis_server(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Route redis.from_url to an in-process fakeredis server."""
    fakeredis = pytest.importorskip("fakeredis")
    import redis

    server = fakeredis.FakeServer()

    def from_url(url: str, **kwargs):  # type: ignore[no-untyped-def]
        return fakeredis.FakeRedis(server=server, **kwargs)

    monkeypatch.setattr(redis, "from_url", from_url)
    return server


@pytest.fixture
def client(fake_embedder: FakeEmbedder, llm: CountingProvider) -> TestClient:
    from app.main import app

    return TestClient(app)
