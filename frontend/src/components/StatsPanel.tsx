import { type FormEvent, useState } from "react";

import { api } from "../api";
import { ms, pct, similarity } from "../format";
import type { StatsResponse } from "../types";

interface Props {
  stats: StatsResponse | null;
  error: string | null;
  onCleared: () => void;
}

export function StatsPanel({ stats, error, onCleared }: Props) {
  return (
    <div className="card">
      <h2>Live cache stats</h2>
      <p className="sub">Since the gateway started. Refreshes every 5 seconds.</p>
      {error && !stats && <p className="empty">Couldn't load stats: {error}</p>}
      {stats && (
        <>
          <div className="tiles">
            <Tile
              label="Hit rate"
              value={stats.total_queries ? pct(stats.hit_rate) : "–"}
              note={`${stats.cache_hits} of ${stats.total_queries} queries`}
            />
            <Tile
              label="LLM calls saved"
              value={String(stats.estimated_llm_calls_saved)}
              note="answered from cache"
            />
            <Tile
              label="Hit latency (p50)"
              value={stats.cache_hits ? ms(stats.p50_hit_latency_ms) : "–"}
              note={stats.cache_hits ? `p95 ${ms(stats.p95_hit_latency_ms)}` : "no hits yet"}
            />
            <Tile
              label="Miss latency (p50)"
              value={stats.cache_misses ? ms(stats.p50_miss_latency_ms) : "–"}
              note={stats.cache_misses ? `p95 ${ms(stats.p95_miss_latency_ms)}` : "no misses yet"}
            />
            <Tile label="Guard vetoes" value={String(stats.guard_vetoes)} note="risky matches blocked" />
            <Tile
              label="Cached answers"
              value={String(stats.cache_size)}
              note={`${stats.cache_backend} backend`}
            />
          </div>
          <div className="card-foot">
            <span>Similarity threshold {similarity(stats.similarity_threshold)}</span>
            {stats.llm_errors > 0 && <span>{stats.llm_errors} LLM errors</span>}
          </div>
          <ClearCache onCleared={onCleared} />
        </>
      )}
    </div>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      <div className="note">{note}</div>
    </div>
  );
}

function ClearCache({ onCleared }: { onCleared: () => void }) {
  const [token, setToken] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const { removed } = await api.clearCache(token);
      setMessage(`Cleared ${removed} cached answers.`);
      onCleared();
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <details style={{ marginTop: 12 }}>
      <summary>Clear the cache (admin)</summary>
      <form className="ask" onSubmit={submit} style={{ marginTop: 8 }}>
        <label htmlFor="admin-token" className="sr-only">
          Admin token
        </label>
        <input
          id="admin-token"
          type="text"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="ADMIN_TOKEN"
          autoComplete="off"
        />
        <button className="btn secondary" type="submit" disabled={busy || !token}>
          Clear
        </button>
      </form>
      {message && (
        <p className="kv" role="status">
          {message}
        </p>
      )}
    </details>
  );
}
