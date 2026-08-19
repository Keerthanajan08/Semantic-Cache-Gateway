# Semantic Cache Gateway

An LLM gateway that sits in front of your model calls and caches
**semantically similar** queries — not just exact-match ones — to cut
LLM API cost and latency. Falls back to a genuine RAG pipeline on a
cache miss.

## The problem

Production LLM features (support bots, internal Q&A tools) get hit with
the same underlying question phrased a dozen different ways:

- "How do I reset my password?"
- "I forgot my password, how can I reset it"
- "forgot password help"

Exact-match caching misses all of these. Every single one re-triggers a
full LLM call — real money and real latency, for a question you already
answered five minutes ago.

## How it works

```
query
  │
  ▼
embed (sentence-transformers, local, free)
  │
  ▼
semantic cache lookup (cosine similarity vs past queries)
  │
  ├── similarity ≥ threshold → return cached answer (no LLM call)
  │
  └── below threshold (cache miss)
        │
        ▼
      retrieve top-k relevant chunks from knowledge base (RAG)
        │
        ▼
      build augmented prompt (query + retrieved context)
        │
        ▼
      call LLM provider (pluggable: mock / Groq / Gemini)
        │
        ▼
      store (query, answer) in cache for next time
        │
        ▼
      return answer
```

The cache and the RAG layer are deliberately separate stages. The cache
is a shortcut that skips generation entirely on a hit. RAG is what runs
when there's no shortcut and a real answer has to be generated. Keeping
them distinct is what makes each one easy to reason about (and to explain
in an interview).

## Design decisions worth knowing

- **Cosine similarity via a Python scan, not RediSearch/ANN index.** At
  the scale a demo or early-stage product runs at (hundreds to low
  thousands of cached entries), a linear scan is fast enough, and it
  keeps the project compatible with *any* plain Redis instance, including
  free hosted tiers that don't ship the RediSearch module. This is a
  conscious tradeoff, not an oversight — a real production system with
  millions of cached entries would swap this for an ANN index (e.g.
  Redis Vector Sets, FAISS, or a managed vector DB) without touching any
  other part of the code, because retrieval is isolated behind one
  interface (`BaseCache`).
- **Pluggable LLM providers.** `LLM_PROVIDER=mock|groq|gemini` in `.env`
  is the only thing that changes which model answers a query. This is
  the "AI gateway / multi-provider routing" pattern in miniature — useful
  in real deployments for failover, cost-based routing, or A/B testing
  models.
- **Mock provider by default.** The whole project builds, runs, and
  demos for $0 with no API key at all. Swap in a real free-tier key
  (Groq or Gemini both have one) when you want real generations.
- **In-memory cache by default, Redis optional.** Same reasoning — you
  can run and test the whole thing with zero external services, then
  point `REDIS_URL` at a real instance for something you'd actually
  deploy.

## Demo

### API

The gateway exposes a simple FastAPI interface for health checks, querying,
and cache statistics.

<img width="903" height="509" alt="image" src="https://github.com/user-attachments/assets/1f1a7d86-25da-4421-90b5-fa01267693ca" />


- `GET /health` — health check and active LLM provider
- `POST /query` — submit a query and receive a cached or generated response
- `GET /stats` — view cache performance statistics

### Semantic Cache Hit

Semantically similar queries can be served directly from the cache without
invoking the LLM.



<img width="940" height="819" alt="image" src="https://github.com/user-attachments/assets/932d8d5f-5e45-4c0f-a8af-c2154c809be8" />

<img width="940" height="514" alt="image" src="https://github.com/user-attachments/assets/5c7b66cb-c449-43c3-8fde-203d139c58ec" />


A cache hit returns the matched query, similarity score, response latency,
and `provider: cache`.

### Cache Miss + RAG

When no sufficiently similar cached query exists, the gateway falls back to
the RAG pipeline, retrieves relevant knowledge-base chunks, and generates a
response through the configured LLM provider.

<img width="738" height="668" alt="image" src="https://github.com/user-attachments/assets/18a679bf-c8ff-4164-a303-391d74862698" />

<img width="963" height="476" alt="image" src="https://github.com/user-attachments/assets/e6fdc97f-4b22-4b0d-8d09-b149a875cd27" />


The response includes the retrieved sources and their relevance scores.

### Benchmark

The demo also reports cache hit rate, cache latency, miss latency, and
estimated LLM calls avoided across paraphrased queries.

<img width="773" height="782" alt="image" src="https://github.com/user-attachments/assets/a5213dfe-db58-4150-8f9d-9483cbab560f" />

## Setup

```bash
git clone <your-repo-url>
cd semantic-cache-gateway
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # defaults to mock LLM + in-memory cache — works immediately
```

Run it:

```bash
uvicorn app.main:app --reload
```

In another terminal, run the demo (fires paraphrased queries, prints hit
rate and latency numbers):

```bash
python /run_demo.py
```

Run tests:

```bash
pytest
```

### Try it manually

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I reset my password"}'

curl http://127.0.0.1:8000/stats
```

## Using a real LLM

Pick one, both have usable free tiers:

- **Groq** — https://console.groq.com — fast inference, hosts Llama/Mixtral
- **Gemini** — https://aistudio.google.com/apikey — Google's free-tier API

Set in `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

## Deploying

| Piece | Where | Why |
|---|---|---|
| FastAPI backend | [Render](https://render.com) or [Railway](https://railway.app) | Free tier, deploys straight from GitHub |
| Redis | [Upstash](https://upstash.com) | Serverless, generous free tier, pay-per-request beyond that |
| Embedding model | Runs in the same backend process | `all-MiniLM-L6-v2` is ~80MB, fine on a free-tier instance |
| LLM | Groq or Gemini | Free tier, no server needed |

Set the same `.env` values as environment variables in your host's
dashboard — no code changes required.

## What the numbers mean

Run `/run_demo.py` against your own instance and you'll get real,
reportable metrics:

- **Cache hit rate** — % of queries answered without an LLM call
- **Avg latency: cache hit vs miss** — the concrete speedup number
- **Estimated LLM calls saved** — directly maps to $ saved at any given
  per-call cost

These are the numbers worth putting on a resume — they're generated by
running the project, not estimated.

## Project structure

```
app/
  main.py            FastAPI app — the request flow described above
  cache.py           BaseCache / InMemoryCache / RedisCache
  embeddings.py       sentence-transformers wrapper
  llm_providers.py    Mock / Groq / Gemini, one interface
  rag.py              knowledge base loading + retrieval + prompt building
  schemas.py          request/response models
  config.py           env-driven settings
  knowledge_base/      sample FAQ docs used for the RAG demo

run_demo.py         fires paraphrased queries, prints hit-rate metrics
test_cache.py       unit tests for cache hit/miss logic
```

## Benchmark Results

Tested semantic cache performance across 15 paraphrased support queries
covering password reset, account lockout, 2FA, subscription cancellation,
and refunds.

| Similarity Threshold | Cache Hit Rate | Avg Cache Latency | Avg Miss Latency | LLM Calls Avoided |
|---|---|---|---|---|
| 0.90 | 6.7% | 13.0 ms | 64.9 ms | 1/15 |
| 0.85 | 6.7% | 13.8 ms | 65.2 ms | 1/15 |
| 0.80 | 6.7% | 15.8 ms | 66.2 ms | 1/15 |
| 0.75 | 20.0% | 17.3 ms | 66.8 ms | 3/15 |
| 0.70 | 40.0% | 13.6 ms | 65.4 ms | 6/15 |

A lower similarity threshold improves cache reuse but increases the risk of
incorrect semantic matches. This demonstrates the tradeoff between cache
coverage and precision when tuning semantic caching systems.
