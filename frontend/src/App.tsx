import { useCallback, useEffect, useState } from "react";

import evalResults from "../../eval/results.json";
import { api } from "./api";
import { History, type HistoryEntry } from "./components/History";
import { HealthBar } from "./components/HealthBar";
import { QueryPlayground } from "./components/QueryPlayground";
import { StatsPanel } from "./components/StatsPanel";
import { ThresholdExplorer } from "./components/ThresholdExplorer";
import type { EvalResults, HealthResponse, QueryResponse, StatsResponse } from "./types";

const EVAL = evalResults as EvalResults;
const POLL_MS = 5000;
const MAX_HISTORY = 50;

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const refresh = useCallback(async () => {
    const [h, s] = await Promise.allSettled([api.health(), api.stats()]);
    if (h.status === "fulfilled") {
      setHealth(h.value);
      setHealthError(null);
    } else {
      setHealthError(String(h.reason));
    }
    if (s.status === "fulfilled") {
      setStats(s.value);
      setStatsError(null);
    } else {
      setStatsError(s.reason instanceof Error ? s.reason.message : String(s.reason));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => {
      // Don't poll from a background tab.
      if (document.visibilityState === "visible") void refresh();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const onResult = useCallback(
    (query: string, result: QueryResponse) => {
      setHistory((prev) =>
        [{ id: result.request_id, query, result }, ...prev].slice(0, MAX_HISTORY),
      );
      void refresh();
    },
    [refresh],
  );

  return (
    <main className="app">
      <header className="header">
        <div>
          <h1>Semantic Cache Gateway</h1>
          <p>Serve paraphrased questions from cache, without serving the wrong answer.</p>
        </div>
        <HealthBar health={health} error={healthError} />
      </header>

      <div className="grid">
        <div className="stack">
          <QueryPlayground threshold={stats?.similarity_threshold ?? null} onResult={onResult} />
          <History entries={history} />
        </div>
        <div className="stack">
          <StatsPanel stats={stats} error={statsError} onCleared={() => void refresh()} />
          <ThresholdExplorer
            data={EVAL}
            liveThreshold={stats?.similarity_threshold ?? null}
            liveModel={health?.embedding_model ?? null}
          />
        </div>
      </div>
    </main>
  );
}
