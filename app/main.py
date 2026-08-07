"""
Semantic Cache Gateway
======================

Flow for every incoming query:

  1. Embed the query.
  2. Check the semantic cache for a past query with cosine similarity
     above CACHE_SIMILARITY_THRESHOLD.
       -> HIT: return the cached answer immediately. No LLM call.
       -> MISS: continue to step 3.
  3. Retrieve relevant chunks from the knowledge base (RAG) and build an
     augmented prompt.
  4. Call the LLM provider (mock/Groq/Gemini — config-driven).
  5. Store the (query, answer) pair in the cache for future hits.
  6. Return the answer either way, with cache_hit / similarity / latency
     so the caller (and your resume metrics) can see exactly what
     happened.

Run it:
    uvicorn app.main:app --reload
"""
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.cache import get_cache
from app.config import settings
from app.embeddings import get_embedder
from app.llm_providers import get_llm_provider
from app.rag import answer_with_rag
from app.schemas import QueryRequest, QueryResponse, StatsResponse

app = FastAPI(
    title="Semantic Cache Gateway",
    description="An LLM gateway that caches semantically similar queries to cut cost and latency.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-process counters for the /stats endpoint. Good enough for a
# single-instance demo; a production version would push these to a real
# metrics backend (Prometheus, etc.) instead.
_stats = {
    "total_queries": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "hit_latencies_ms": [],
    "miss_latencies_ms": [],
}


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.perf_counter()

    embedder = get_embedder()
    cache = get_cache()
    query_vector = embedder.encode(request.query)

    lookup = cache.lookup(query_vector)
    _stats["total_queries"] += 1

    if lookup.hit:
        latency_ms = (time.perf_counter() - start) * 1000
        _stats["cache_hits"] += 1
        _stats["hit_latencies_ms"].append(latency_ms)
        return QueryResponse(
            answer=lookup.entry.answer,
            cache_hit=True,
            matched_query=lookup.entry.query,
            similarity=round(lookup.similarity, 4),
            latency_ms=round(latency_ms, 2),
            sources=[],
            provider="cache",
        )

    # Cache miss -> genuine RAG path -> LLM call -> cache the result.
    prompt, sources = answer_with_rag(request.query, query_vector)
    llm = get_llm_provider()
    answer = llm.generate(prompt)
    cache.store(request.query, answer, query_vector)

    latency_ms = (time.perf_counter() - start) * 1000
    _stats["cache_misses"] += 1
    _stats["miss_latencies_ms"].append(latency_ms)

    return QueryResponse(
        answer=answer,
        cache_hit=False,
        similarity=round(lookup.similarity, 4),
        latency_ms=round(latency_ms, 2),
        sources=sources,
        provider=llm.name,
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    total = _stats["total_queries"]
    hits = _stats["cache_hits"]
    hit_latencies = _stats["hit_latencies_ms"]
    miss_latencies = _stats["miss_latencies_ms"]

    return StatsResponse(
        total_queries=total,
        cache_hits=hits,
        cache_misses=_stats["cache_misses"],
        hit_rate=round(hits / total, 4) if total else 0.0,
        estimated_llm_calls_saved=hits,
        avg_hit_latency_ms=round(sum(hit_latencies) / len(hit_latencies), 2) if hit_latencies else 0.0,
        avg_miss_latency_ms=round(sum(miss_latencies) / len(miss_latencies), 2) if miss_latencies else 0.0,
    )
