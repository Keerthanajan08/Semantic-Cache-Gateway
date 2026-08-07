from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question")


class SourceChunk(BaseModel):
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity: Optional[float] = None
    latency_ms: float
    sources: list[SourceChunk] = []
    provider: str


class StatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    estimated_llm_calls_saved: int
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float
