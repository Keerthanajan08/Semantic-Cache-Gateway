"""
Integration tests with the real all-MiniLM-L6-v2 model. Slow (loads torch,
downloads ~80MB on first run), so they're excluded by default.

    pytest -m slow        # only these
    pytest -m ""          # everything
"""
import pytest
from fastapi.testclient import TestClient

import app.llm_providers as llm_module
from app.main import app
from tests.conftest import CountingProvider

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_embedder():
    pytest.importorskip("sentence_transformers")
    from app.embeddings import get_embedder

    try:
        return get_embedder()
    except Exception as exc:  # offline, HF hub unreachable, etc.
        pytest.skip(f"embedding model unavailable: {exc}")


@pytest.fixture
def client(real_embedder, monkeypatch):
    llm = CountingProvider()
    monkeypatch.setattr(llm_module, "_provider_instance", llm)
    return TestClient(app), llm


def ask(client, q):
    return client.post("/query", json={"query": q}).json()


def test_real_embeddings_are_normalized(real_embedder):
    import numpy as np

    vec = real_embedder.encode("hello world")
    assert vec.shape == (384,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "variant",
    ["How do I reset my password?", "how do i reset my password", "How can I reset my password?"],
)
def test_near_identical_phrasings_hit_at_default_threshold(client, variant):
    c, llm = client
    ask(c, "How do I reset my password?")
    body = ask(c, variant)
    assert body["cache_hit"] is True
    assert len(llm.prompts) == 1


def test_unrelated_question_misses(client):
    c, llm = client
    ask(c, "How do I reset my password?")
    body = ask(c, "Do you offer refunds?")
    assert body["cache_hit"] is False
    assert body["similarity"] < 0.5
    assert len(llm.prompts) == 2


@pytest.mark.parametrize(
    "question, expected_top_doc",
    [
        ("How do I reset my password?", "account_and_auth.md#0"),
        ("Why is my account locked?", "account_and_auth.md#1"),
        ("How do I enable two-factor authentication?", "account_and_auth.md#2"),
        ("How do I cancel my subscription?", "billing.md#1"),
        ("Do you offer refunds?", "billing.md#2"),
        ("How do I update my credit card?", "billing.md#3"),
    ],
)
def test_rag_retrieves_the_right_paragraph(client, question, expected_top_doc):
    c, _ = client
    body = ask(c, question)
    assert body["cache_hit"] is False
    assert body["sources"][0]["doc_id"] == expected_top_doc


@pytest.mark.xfail(
    strict=True,
    reason="Embedding similarity ignores negation: 'NOT reset' scores ~0.93 and is served the reset answer",
)
def test_negated_question_is_not_served_cached_answer(client):
    c, _ = client
    ask(c, "How do I reset my password?")
    assert ask(c, "How do I NOT reset my password?")["cache_hit"] is False
