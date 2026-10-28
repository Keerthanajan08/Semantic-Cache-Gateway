import { guardLabel, ms, similarity } from "../format";
import type { QueryResponse } from "../types";

interface Props {
  result: QueryResponse;
  threshold: number | null;
}

const STATUS_TEXT = {
  hit: { icon: "✓", label: "HIT", note: "Served from cache, no LLM call" },
  miss: { icon: "→", label: "MISS", note: "Answered by RAG + LLM, then cached" },
  bypass: { icon: "↻", label: "BYPASS", note: "Cache skipped on request; answer refreshed" },
} as const;

export function ResultCard({ result, threshold }: Props) {
  const status = STATUS_TEXT[result.cache_status];
  const sim = result.similarity;

  return (
    <section className="result" aria-label="Query result" aria-live="polite">
      <div className="result-head">
        <span className={`badge ${result.cache_status}`}>
          <span aria-hidden="true">{status.icon}</span>
          {status.label}
        </span>
        <span className="kv">{status.note}</span>
        <span className="kv">
          Latency <strong>{ms(result.latency_ms)}</strong>
        </span>
      </div>

      {sim !== null && (
        <SimilarityMeter value={sim} threshold={threshold} />
      )}

      {result.matched_query && (
        <p className="matched">
          {result.cache_hit ? "Matched cached question: " : "Closest cached question: "}
          <strong>“{result.matched_query}”</strong>
        </p>
      )}

      {result.guard_vetoes.length > 0 && (
        <div className="vetoes" aria-label="Guard vetoes">
          {result.guard_vetoes.map((g) => (
            <span className="veto" key={g}>
              <span aria-hidden="true">⚠</span>
              Blocked by guard: {guardLabel(g)}
            </span>
          ))}
        </div>
      )}

      <p className="answer">{result.answer}</p>

      {result.sources.length > 0 && (
        <ul className="sources" aria-label="Retrieved sources">
          {result.sources.map((s, i) => (
            <li key={`${s.doc_id}-${i}`}>
              <span>{s.doc_id}</span>
              <span>score {similarity(s.score)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function SimilarityMeter({ value, threshold }: { value: number; threshold: number | null }) {
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  return (
    <div className="meter">
      <div
        className="meter-track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(value.toFixed(3))}
        aria-label="Similarity to closest cached question"
      >
        <div className="meter-fill" style={{ width: `${clamp(value) * 100}%` }} />
        {threshold !== null && (
          <div
            className="meter-mark"
            style={{ left: `calc(${clamp(threshold) * 100}% - 1px)` }}
            title={`Threshold ${similarity(threshold)}`}
          />
        )}
      </div>
      <div className="meter-legend">
        <span>
          Similarity <strong>{similarity(value)}</strong>
        </span>
        {threshold !== null && <span>Cache threshold {similarity(threshold)} (marker)</span>}
      </div>
    </div>
  );
}
