import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | null;
  error: string | null;
}

export function HealthBar({ health, error }: Props) {
  if (error) {
    return (
      <div className="health" role="status">
        <span className="pill">
          <span className="dot bad" aria-hidden="true" />
          <strong>Gateway unreachable</strong>
        </span>
      </div>
    );
  }
  if (!health) {
    return (
      <div className="health" role="status">
        <span className="pill">Connecting…</span>
      </div>
    );
  }
  const ok = health.status === "ok";
  return (
    <div className="health" role="status" aria-label="Gateway status">
      <span className="pill">
        <span className={`dot ${ok ? "ok" : "warn"}`} aria-hidden="true" />
        <strong>{ok ? "Healthy" : "Degraded: cache unreachable"}</strong>
      </span>
      <span className="pill">
        LLM <strong>{health.llm_provider}</strong>
      </span>
      <span className="pill">
        Cache <strong>{health.cache_backend}</strong>
      </span>
      <span className="pill">
        Model <strong>{health.embedding_model}</strong>
      </span>
      <span className="pill">v{health.version}</span>
    </div>
  );
}
