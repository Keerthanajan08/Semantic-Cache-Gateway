"""
Integration tests with the real embedding model (EMBEDDING_MODEL, default
all-mpnet-base-v2) at its evaluated default threshold. Slow (loads torch,
downloads the model on first run), so they're excluded by default.

    pytest -m slow        # only these
    pytest -m ""          # everything
"""

import pytest
from fastapi.testclient import TestClient

import app.llm_providers as llm_module
from app.config import settings
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
    monkeypatch.setattr(settings, "cache_similarity_threshold", None)  # use the evaluated default
    llm = CountingProvider()
    monkeypatch.setattr(llm_module, "_provider_instance", llm)
    return TestClient(app), llm


def ask(client, q):
    return client.post("/query", json={"query": q}).json()


def test_real_embeddings_are_normalized(real_embedder):
    import numpy as np

    vec = real_embedder.encode("hello world")
    assert vec.ndim == 1
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "variant",
    [
        "how do i reset my password",
        "How can I reset my password?",
        "I forgot my password, how can I reset it",
    ],
)
def test_paraphrases_hit_at_default_threshold(client, variant):
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
    ("question", "expected_top_doc"),
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


@pytest.mark.parametrize(
    ("cached", "trap", "guard"),
    [
        ("How do I reset my password?", "How do I NOT reset my password?", "negation"),
        (
            "How do I enable two-factor authentication?",
            "How do I disable two-factor authentication?",
            "opposite_action",
        ),
        ("What does the 10 seat plan cost?", "What does the 50 seat plan cost?", "numbers"),
    ],
)
def test_guards_block_high_similarity_traps(client, cached, trap, guard):
    """Each trap clears the threshold on similarity alone; only the guard stops it."""
    c, llm = client
    ask(c, cached)
    body = ask(c, trap)
    assert body["similarity"] >= settings.similarity_threshold
    assert body["cache_status"] == "miss"
    assert body["guard_vetoes"] == [guard]
    assert len(llm.prompts) == 2
