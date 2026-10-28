// Mirrors app/schemas.py. Keep the two in sync.

export type CacheStatus = "hit" | "miss" | "bypass";

export interface SourceChunk {
  doc_id: string;
  text: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  cache_status: CacheStatus;
  cache_hit: boolean;
  matched_query: string | null;
  similarity: number | null;
  guard_vetoes: string[];
  latency_ms: number;
  sources: SourceChunk[];
  provider: string;
  request_id: string;
}

export interface StatsResponse {
  total_queries: number;
  cache_hits: number;
  cache_misses: number;
  cache_bypassed: number;
  llm_errors: number;
  guard_vetoes: number;
  hit_rate: number;
  estimated_llm_calls_saved: number;
  avg_hit_latency_ms: number;
  p50_hit_latency_ms: number;
  p95_hit_latency_ms: number;
  avg_miss_latency_ms: number;
  p50_miss_latency_ms: number;
  p95_miss_latency_ms: number;
  cache_size: number;
  cache_backend: string;
  similarity_threshold: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  llm_provider: string;
  cache_backend: string;
  cache_reachable: boolean;
  embedding_model: string;
  version: string;
}

// eval/results.json, written by scripts/evaluate_threshold.py.
export interface SweepRow {
  threshold: number;
  guards: boolean;
  hit_rate: number;
  wrong_rate: number;
  precision: number;
}

export interface ModelEval {
  model: string;
  embed_ms: number;
  expected_latency_ms: number;
  chosen: SweepRow;
  sweep: SweepRow[];
}

export interface GuardProbe {
  cached: string;
  incoming: string;
  similarity: number;
  would_hit: boolean;
  blocked_by: string | null;
}

export interface EvalResults {
  n_queries: number;
  n_intents: number;
  max_wrong: number;
  llm_ms: number;
  best_model: string;
  models: ModelEval[];
  guard_probes: GuardProbe[];
}
