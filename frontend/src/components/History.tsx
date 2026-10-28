import { guardLabel, ms, similarity } from "../format";
import type { QueryResponse } from "../types";

export interface HistoryEntry {
  id: string;
  query: string;
  result: QueryResponse;
}

export function History({ entries }: { entries: HistoryEntry[] }) {
  return (
    <div className="card">
      <h2>This session</h2>
      <p className="sub">Every question you asked here, newest first.</p>
      {entries.length === 0 ? (
        <p className="empty">No questions yet.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Question</th>
                <th>Result</th>
                <th className="num">Similarity</th>
                <th className="num">Latency</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(({ id, query, result }) => (
                <tr key={id}>
                  <td className="query">
                    {query}
                    {result.guard_vetoes.length > 0 && (
                      <div className="kv">
                        Blocked: {result.guard_vetoes.map(guardLabel).join(", ")}
                      </div>
                    )}
                  </td>
                  <td>{result.cache_status.toUpperCase()}</td>
                  <td className="num">
                    {result.similarity === null ? "–" : similarity(result.similarity)}
                  </td>
                  <td className="num">{ms(result.latency_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
