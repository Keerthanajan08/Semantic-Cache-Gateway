import type { EvalResults, QueryResponse, StatsResponse, SweepRow } from "../types";

const row = (threshold: number, guards: boolean, hit: number, wrong: number): SweepRow => ({
  threshold,
  guards,
  hit_rate: hit,
  wrong_rate: wrong,
  precision: hit ? (hit - wrong) / hit : 1,
});

export const evalResults: EvalResults = {
  n_queries: 106,
  n_intents: 14,
  max_wrong: 0.01,
  llm_ms: 800,
  best_model: "all-mpnet-base-v2",
  models: [
    {
      model: "all-mpnet-base-v2",
      embed_ms: 51.2,
      expected_latency_ms: 346,
      chosen: row(0.7, true, 0.632, 0.009),
      sweep: [
        row(0.68, true, 0.66, 0.019),
        row(0.7, true, 0.632, 0.009),
        row(0.72, true, 0.594, 0),
        row(0.68, false, 0.717, 0.057),
        row(0.7, false, 0.679, 0.028),
        row(0.72, false, 0.632, 0.019),
      ],
    },
    {
      model: "all-MiniLM-L6-v2",
      embed_ms: 13.2,
      expected_latency_ms: 406,
      chosen: row(0.72, true, 0.5, 0.005),
      sweep: [row(0.7, true, 0.55, 0.02), row(0.72, true, 0.5, 0.005)],
    },
  ],
  guard_probes: [],
};

export const stats: StatsResponse = {
  total_queries: 4,
  cache_hits: 3,
  cache_misses: 1,
  cache_bypassed: 0,
  llm_errors: 0,
  guard_vetoes: 1,
  hit_rate: 0.75,
  estimated_llm_calls_saved: 3,
  avg_hit_latency_ms: 14,
  p50_hit_latency_ms: 13.7,
  p95_hit_latency_ms: 18.2,
  avg_miss_latency_ms: 65,
  p50_miss_latency_ms: 64.9,
  p95_miss_latency_ms: 70.1,
  cache_size: 1,
  cache_backend: "memory",
  similarity_threshold: 0.7,
};

export const hit: QueryResponse = {
  answer: "Go to Settings → Security → Reset password.",
  cache_status: "hit",
  cache_hit: true,
  matched_query: "How do I reset my password?",
  similarity: 0.88,
  guard_vetoes: [],
  latency_ms: 13.7,
  sources: [],
  provider: "mock",
  request_id: "req-1",
};

export const vetoed: QueryResponse = {
  ...hit,
  answer: "Generated answer",
  cache_status: "miss",
  cache_hit: false,
  similarity: 0.734,
  guard_vetoes: ["negation"],
  latency_ms: 112.6,
  sources: [{ doc_id: "account_and_auth.md#2", text: "…", score: 0.61 }],
  request_id: "req-2",
};

/** Replace fetch with a function returning the given JSON bodies in order. */
export function mockFetch(...bodies: Array<{ status?: number; body: unknown }>) {
  const fn = vi.fn();
  for (const { status = 200, body } of bodies) {
    fn.mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );
  }
  vi.stubGlobal("fetch", fn);
  return fn;
}
