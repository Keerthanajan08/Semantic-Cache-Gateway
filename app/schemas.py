from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"query": "How do I reset my password?"}]}
    )

    query: str = Field(..., description="Natural-language question")
    use_cache: bool = Field(
        default=True,
        description="Set false to force a fresh LLM answer (the result still refreshes the cache).",
    )

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        value = " ".join(value.split())  # trim and collapse internal whitespace
        if not value:
            raise ValueError("query must contain non-whitespace characters")
        if len(value) > settings.max_query_chars:
            raise ValueError(f"query must be at most {settings.max_query_chars} characters")
        return value


class SourceChunk(BaseModel):
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    cache_status: Literal["hit", "miss", "bypass"]
    cache_hit: bool
    matched_query: str | None = None
    similarity: float | None = Field(
        default=None, description="Similarity to the closest cached query (0-1)."
    )
    guard_vetoes: list[str] = Field(
        default_factory=list,
        description="Guards that rejected a cached match above the threshold, e.g. 'negation'.",
    )
    latency_ms: float
    sources: list[SourceChunk] = Field(default_factory=list)
    provider: str
    request_id: str


class StatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_misses: int
    cache_bypassed: int
    llm_errors: int
    guard_vetoes: int
    hit_rate: float
    estimated_llm_calls_saved: int
    avg_hit_latency_ms: float
    p50_hit_latency_ms: float
    p95_hit_latency_ms: float
    avg_miss_latency_ms: float
    p50_miss_latency_ms: float
    p95_miss_latency_ms: float
    cache_size: int
    cache_backend: str
    similarity_threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_provider: str
    cache_backend: str
    cache_reachable: bool
    embedding_model: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class ClearCacheResponse(BaseModel):
    removed: int
