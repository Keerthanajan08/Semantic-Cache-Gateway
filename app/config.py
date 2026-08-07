"""
Centralized configuration. Everything is env-driven so the same code runs
locally (in-memory cache, mock LLM) and in production (Redis, real LLM)
with zero code changes.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider: "mock" | "groq" | "gemini"
    llm_provider: str = "mock"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Cache
    redis_url: str = ""
    cache_similarity_threshold: float = 0.90

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG
    rag_top_k: int = 3


settings = Settings()
