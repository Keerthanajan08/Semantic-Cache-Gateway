"""
Centralized, validated configuration.

Everything is env-driven, so the same code runs locally (in-memory cache,
mock LLM) and in production (Redis, real LLM) with zero code changes.
Invalid values (a typo in LLM_PROVIDER, a threshold of 1.5) fail at startup
instead of surfacing as confusing behaviour at request time.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["mock", "groq", "gemini"]

# Per-model defaults from the leave-one-out evaluation in eval/results.md:
# the most cache reuse at which at most 1% of queries get a wrong answer.
# Similarity scales differ a lot between models (bge/gte score everything
# high), so a threshold tuned for one model is meaningless for another.
RECOMMENDED_THRESHOLDS: dict[str, float] = {
    "all-mpnet-base-v2": 0.70,
    "BAAI/bge-small-en-v1.5": 0.86,
    "all-MiniLM-L6-v2": 0.74,
    "sentence-transformers/all-MiniLM-L12-v2": 0.76,
    "BAAI/bge-base-en-v1.5": 0.86,
}
FALLBACK_THRESHOLD = 0.85  # conservative for models we haven't evaluated


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    llm_provider: LLMProviderName = "mock"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_output_tokens: int = Field(default=500, gt=0)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # --- Cache ---
    redis_url: str = ""
    # Unset = use the evaluated default for EMBEDDING_MODEL (see RECOMMENDED_THRESHOLDS).
    cache_similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=86_400, ge=0, description="0 disables expiry")
    cache_max_entries: int = Field(default=10_000, gt=0, description="In-memory backend only")
    # Lexical safety checks (negation, numbers) that veto risky semantic matches.
    cache_guards: bool = True

    # --- Embeddings ---
    # Lowest expected latency per request in eval/results.md. On a 512MB host,
    # all-MiniLM-L6-v2 (~90MB, threshold 0.74) is the lightweight alternative.
    embedding_model: str = "all-mpnet-base-v2"

    # --- RAG ---
    rag_top_k: int = Field(default=3, gt=0)

    # --- API ---
    max_query_chars: int = Field(default=2_000, gt=0)
    warmup_on_startup: bool = True
    # DELETE /cache is disabled unless this is set; callers send it as X-Admin-Token.
    admin_token: str = ""
    cors_allow_origins: list[str] = ["*"]
    log_level: str = "INFO"

    @property
    def similarity_threshold(self) -> float:
        if self.cache_similarity_threshold is not None:
            return self.cache_similarity_threshold
        return RECOMMENDED_THRESHOLDS.get(self.embedding_model, FALLBACK_THRESHOLD)


settings = Settings()
