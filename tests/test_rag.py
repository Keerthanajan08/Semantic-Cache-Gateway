import pytest

from app import rag
from app.config import settings
from app.rag import KB_DIR, KnowledgeBase, answer_with_rag, build_augmented_prompt, get_knowledge_base
from app.schemas import SourceChunk
from tests.conftest import fake_embed


@pytest.fixture
def kb_dir(tmp_path):
    (tmp_path / "a_auth.md").write_text(
        "Reset your password from the login page.\n\n"
        "Accounts lock after five failed login attempts.",
        encoding="utf-8",
    )
    (tmp_path / "b_billing.md").write_text(
        "Refunds take five to seven business days.", encoding="utf-8"
    )
    (tmp_path / "notes.txt").write_text("not markdown, must be ignored", encoding="utf-8")
    return tmp_path


class TestKnowledgeBaseLoading:
    def test_splits_markdown_files_into_paragraph_chunks(self, fake_embedder, kb_dir):
        kb = KnowledgeBase(kb_dir)
        assert [c.doc_id for c in kb.chunks] == ["a_auth.md#0", "a_auth.md#1", "b_billing.md#0"]
        assert kb.chunks[0].text == "Reset your password from the login page."

    def test_ignores_non_markdown_files(self, fake_embedder, kb_dir):
        kb = KnowledgeBase(kb_dir)
        assert all("must be ignored" not in c.text for c in kb.chunks)

    def test_skips_blank_paragraphs(self, fake_embedder, tmp_path):
        (tmp_path / "doc.md").write_text("first\n\n   \n\nsecond\n\n\n\n", encoding="utf-8")
        kb = KnowledgeBase(tmp_path)
        assert [c.text for c in kb.chunks] == ["first", "second"]

    def test_empty_directory_gives_empty_kb(self, fake_embedder, tmp_path):
        kb = KnowledgeBase(tmp_path)
        assert kb.chunks == []
        assert kb.retrieve(fake_embed("anything"), top_k=3) == []

    def test_chunks_carry_embeddings(self, fake_embedder, kb_dir):
        kb = KnowledgeBase(kb_dir)
        for chunk in kb.chunks:
            assert chunk.vector.shape == fake_embed("x").shape

    def test_bundled_knowledge_base_loads(self, fake_embedder):
        kb = KnowledgeBase(KB_DIR)
        files = {c.doc_id.split("#")[0] for c in kb.chunks}
        assert files == {"account_and_auth.md", "billing.md"}
        assert len(kb.chunks) == 10


class TestRetrieval:
    def test_most_relevant_chunk_ranks_first(self, fake_embedder, kb_dir):
        kb = KnowledgeBase(kb_dir)
        results = kb.retrieve(fake_embed("how long do refunds take"), top_k=3)
        assert results[0].doc_id == "b_billing.md#0"

    def test_results_sorted_by_descending_score(self, fake_embedder, kb_dir):
        kb = KnowledgeBase(kb_dir)
        scores = [r.score for r in kb.retrieve(fake_embed("password login"), top_k=3)]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("top_k, expected", [(1, 1), (2, 2), (3, 3), (10, 3)])
    def test_respects_top_k(self, fake_embedder, kb_dir, top_k, expected):
        kb = KnowledgeBase(kb_dir)
        assert len(kb.retrieve(fake_embed("password"), top_k=top_k)) == expected

    def test_scores_rounded_to_four_places(self, fake_embedder, kb_dir):
        kb = KnowledgeBase(kb_dir)
        for r in kb.retrieve(fake_embed("password login page"), top_k=3):
            assert r.score == round(r.score, 4)


class TestPromptBuilding:
    def test_prompt_contains_question_context_and_citations(self):
        sources = [
            SourceChunk(doc_id="billing.md#2", text="Refunds take 5-7 days.", score=0.8),
            SourceChunk(doc_id="billing.md#1", text="Cancel from Billing settings.", score=0.5),
        ]
        prompt = build_augmented_prompt("Can I get a refund?", sources)
        assert "Question: Can I get a refund?" in prompt
        assert "[billing.md#2] Refunds take 5-7 days." in prompt
        assert "[billing.md#1] Cancel from Billing settings." in prompt
        assert prompt.index("billing.md#2") < prompt.index("billing.md#1")
        assert prompt.rstrip().endswith("Answer:")

    def test_prompt_instructs_model_not_to_guess(self):
        prompt = build_augmented_prompt("q", [])
        assert "only the context" in prompt
        assert "instead of guessing" in prompt


class TestAnswerWithRag:
    def test_uses_configured_top_k(self, fake_embedder, monkeypatch):
        monkeypatch.setattr(settings, "rag_top_k", 2)
        prompt, sources = answer_with_rag("reset password", fake_embed("reset password"))
        assert len(sources) == 2
        assert "Question: reset password" in prompt

    def test_sources_appear_in_prompt_in_rank_order(self, fake_embedder):
        q = "how do I reset my password"
        prompt, sources = answer_with_rag(q, fake_embed(q))
        positions = [prompt.index(f"[{s.doc_id}]") for s in sources]
        assert positions == sorted(positions)

    def test_knowledge_base_is_loaded_once(self, fake_embedder, monkeypatch):
        loads = []
        original_load = KnowledgeBase._load
        monkeypatch.setattr(
            KnowledgeBase, "_load", lambda self, d: (loads.append(d), original_load(self, d))
        )
        for _ in range(3):
            answer_with_rag("q", fake_embed("q"))
        assert len(loads) == 1
        assert get_knowledge_base() is get_knowledge_base()
        assert rag.KB_DIR.name == "knowledge_base"
