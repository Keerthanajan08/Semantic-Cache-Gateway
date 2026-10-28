import { type FormEvent, useState } from "react";

import { api } from "../api";
import type { QueryResponse } from "../types";
import { ResultCard } from "./ResultCard";

// Same questions as scripts/run_demo.py: paraphrases that should hit once the
// first is cached, and traps that look similar but must never share an answer.
export const PARAPHRASES = [
  "How do I reset my password?",
  "I forgot my password, how can I reset it",
  "How do I enable two-factor authentication?",
  "how to turn on 2FA",
  "Do you offer refunds?",
  "can I get my money back",
];
export const TRAPS = [
  "How do I NOT reset my password?",
  "How do I disable two-factor authentication?",
  "Why can't I get a refund?",
];

interface Props {
  threshold: number | null;
  onResult: (query: string, result: QueryResponse) => void;
}

export function QueryPlayground({ threshold, onResult }: Props) {
  const [query, setQuery] = useState("");
  const [useCache, setUseCache] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setQuery(trimmed);
    setLoading(true);
    setError(null);
    try {
      const response = await api.query(trimmed, useCache);
      setResult(response);
      onResult(trimmed, response);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(query);
  }

  return (
    <div className="card">
      <h2>Ask the gateway</h2>
      <p className="sub">
        Ask a question, then ask it again in different words. A paraphrase is served from the cache;
        a look-alike with a different meaning is blocked by the guards.
      </p>

      <form className="ask" onSubmit={submit}>
        <label htmlFor="query" className="sr-only">
          Question
        </label>
        <input
          id="query"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. How do I reset my password?"
          autoComplete="off"
        />
        <button className="btn" type="submit" disabled={loading || !query.trim()}>
          {loading ? "Asking…" : "Ask"}
        </button>
      </form>

      <div className="row">
        <label>
          <input
            type="checkbox"
            checked={useCache}
            onChange={(e) => setUseCache(e.target.checked)}
          />
          Use cache
        </label>
        {!useCache && <span>Forces a fresh LLM answer (the result still refreshes the cache).</span>}
      </div>

      <div className="chips">
        <span className="chips-label">Paraphrases</span>
        {PARAPHRASES.map((q) => (
          <button key={q} type="button" className="chip" onClick={() => void ask(q)}>
            {q}
          </button>
        ))}
      </div>
      <div className="chips">
        <span className="chips-label">Traps: similar words, different meaning</span>
        {TRAPS.map((q) => (
          <button key={q} type="button" className="chip" onClick={() => void ask(q)}>
            {q}
          </button>
        ))}
      </div>

      {error && (
        <div className="error" role="alert">
          <strong>Request failed:</strong> {error}
        </div>
      )}
      {result && <ResultCard result={result} threshold={threshold} />}
    </div>
  );
}
