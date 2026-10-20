import pytest
from pydantic import ValidationError

from app.config import FALLBACK_THRESHOLD, RECOMMENDED_THRESHOLDS, Settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Settings read env vars and .env; isolate both."""
    monkeypatch.chdir(tmp_path)
    for var in ("LLM_PROVIDER", "CACHE_SIMILARITY_THRESHOLD", "EMBEDDING_MODEL", "REDIS_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.usefixtures("clean_env")
class TestSettings:
    def test_zero_setup_defaults(self):
        s = Settings()
        assert s.llm_provider == "mock"
        assert s.redis_url == ""
        assert s.embedding_model == "all-mpnet-base-v2"
        assert s.cache_guards is True

    def test_typo_in_provider_fails_at_startup(self, monkeypatch):
        """Regression: 'grok' used to silently fall back to the mock LLM."""
        monkeypatch.setenv("LLM_PROVIDER", "grok")
        with pytest.raises(ValidationError, match="llm_provider"):
            Settings()

    @pytest.mark.parametrize("bad", ["1.5", "-0.1", "abc"])
    def test_threshold_must_be_a_probability(self, monkeypatch, bad):
        monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", bad)
        with pytest.raises(ValidationError):
            Settings()

    def test_reads_dotenv_file(self, tmp_path):
        (tmp_path / ".env").write_text("LLM_PROVIDER=gemini\nRAG_TOP_K=5\n")
        s = Settings()
        assert (s.llm_provider, s.rag_top_k) == ("gemini", 5)


@pytest.mark.usefixtures("clean_env")
class TestThresholdResolution:
    def test_uses_evaluated_threshold_for_the_configured_model(self):
        s = Settings(embedding_model="all-MiniLM-L6-v2")
        assert s.similarity_threshold == RECOMMENDED_THRESHOLDS["all-MiniLM-L6-v2"]

    def test_default_model_has_an_evaluated_threshold(self):
        s = Settings()
        assert s.embedding_model in RECOMMENDED_THRESHOLDS

    def test_explicit_threshold_wins(self):
        assert Settings(cache_similarity_threshold=0.5).similarity_threshold == 0.5

    def test_unknown_model_gets_conservative_fallback(self):
        s = Settings(embedding_model="some/unevaluated-model")
        assert s.similarity_threshold == FALLBACK_THRESHOLD
