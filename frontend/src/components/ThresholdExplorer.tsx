import { type MouseEvent, useCallback, useMemo, useState } from "react";

import { pct, similarity } from "../format";
import type { EvalResults, ModelEval, SweepRow } from "../types";

// Chart geometry is in real pixels: the SVG is drawn at its container's width
// (measured below), so text stays readable on phones instead of shrinking.
const H = 280;
const M = { top: 22, right: 16, bottom: 44, left: 44 };
const X_MIN = 0.49;
const X_MAX = 0.99;
const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

interface Scale {
  w: number;
  x: (t: number) => number;
  y: (v: number) => number;
  xInverse: (px: number) => number;
}

function makeScale(w: number): Scale {
  const plotW = w - M.left - M.right;
  return {
    w,
    x: (t) => M.left + ((t - X_MIN) / (X_MAX - X_MIN)) * plotW,
    y: (v) => M.top + (1 - v) * (H - M.top - M.bottom),
    xInverse: (px) => X_MIN + ((px - M.left) / plotW) * (X_MAX - X_MIN),
  };
}

/** Width of an element, kept up to date as it resizes. */
function useWidth(fallback: number): [(el: HTMLDivElement | null) => void, number] {
  const [width, setWidth] = useState(fallback);
  const ref = useCallback((el: HTMLDivElement | null) => {
    if (!el) return;
    const measure = () => setWidth(Math.max(280, Math.round(el.getBoundingClientRect().width)));
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect(); // React 19 runs this when the element unmounts
  }, []);
  return [ref, width];
}

function path(rows: SweepRow[], value: (r: SweepRow) => number, { x, y }: Scale): string {
  return rows
    .map((r, i) => `${i === 0 ? "M" : "L"}${x(r.threshold).toFixed(1)},${y(value(r)).toFixed(1)}`)
    .join("");
}

function nearest(rows: SweepRow[], t: number): SweepRow | undefined {
  let best: SweepRow | undefined;
  for (const r of rows) {
    if (!best || Math.abs(r.threshold - t) < Math.abs(best.threshold - t)) best = r;
  }
  return best;
}

interface Props {
  data: EvalResults;
  /** Threshold the running gateway uses, if known. */
  liveThreshold: number | null;
  liveModel: string | null;
}

export function ThresholdExplorer({ data, liveThreshold, liveModel }: Props) {
  const initialModel =
    data.models.find((m) => m.model === liveModel)?.model ?? data.best_model;
  const [modelName, setModelName] = useState(initialModel);
  const [guards, setGuards] = useState(true);
  const model: ModelEval = data.models.find((m) => m.model === modelName) ?? data.models[0]!;
  const [selected, setSelected] = useState(model.chosen.threshold);
  const [hover, setHover] = useState<SweepRow | null>(null);
  const [chartRef, width] = useWidth(560);
  const scale = makeScale(width);
  const { x, y, xInverse } = scale;
  const W = width;
  const xTicks = width < 420 ? [0.5, 0.7, 0.9] : [0.5, 0.6, 0.7, 0.8, 0.9];

  const rows = useMemo(
    () =>
      model.sweep
        .filter((r) => r.guards === guards)
        .sort((a, b) => a.threshold - b.threshold),
    [model, guards],
  );
  const current = nearest(rows, selected);
  const chosen = model.chosen;

  function changeModel(name: string) {
    setModelName(name);
    const next = data.models.find((m) => m.model === name);
    if (next) setSelected(next.chosen.threshold);
  }

  function onMove(event: MouseEvent<SVGRectElement>) {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;
    const box = svg.getBoundingClientRect();
    const px = event.clientX - box.left;
    setHover(nearest(rows, xInverse(px)) ?? null);
  }

  const first = rows[0];
  const isLive = liveModel === model.model;

  return (
    <div className="card">
      <h2>How similar is "similar enough"?</h2>
      <p className="sub">
        Offline evaluation on {data.n_queries} hand-labelled questions across {data.n_intents}{" "}
        intents: how many queries a threshold serves from cache, and how many of those get someone
        else's answer. Drag the slider to explore the trade-off.
      </p>

      <div className="controls">
        <label>
          Embedding model{" "}
          <select value={modelName} onChange={(e) => changeModel(e.target.value)}>
            {data.models.map((m) => (
              <option key={m.model} value={m.model}>
                {m.model}
                {m.model === data.best_model ? " (best)" : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          <input type="checkbox" checked={guards} onChange={(e) => setGuards(e.target.checked)} />{" "}
          Safety guards on
        </label>
      </div>

      <div className="legend" aria-hidden="true">
        <span>
          <i className="swatch" style={{ background: "var(--series-1)" }} />
          Answered from cache
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--series-2)" }} />
          Wrong answers
        </span>
      </div>

      <div className="chart" ref={chartRef}>
        <svg
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Share of queries answered from cache and share of wrong answers versus similarity threshold for ${model.model}, guards ${guards ? "on" : "off"}. Use the slider or the table below for exact values.`}
        >
          <g className="axis">
            {Y_TICKS.map((v) => (
              <g key={v}>
                <line className="gridline" x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} />
                <text x={M.left - 8} y={y(v)} textAnchor="end" dominantBaseline="middle">
                  {pct(v, 0)}
                </text>
              </g>
            ))}
            {xTicks.map((t) => (
              <text key={t} x={x(t)} y={H - M.bottom + 18} textAnchor="middle">
                {t.toFixed(1)}
              </text>
            ))}
            <text x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
              Similarity threshold (stricter →)
            </text>
          </g>

          {/* Default threshold chosen by the evaluation */}
          <line
            x1={x(chosen.threshold)}
            x2={x(chosen.threshold)}
            y1={M.top}
            y2={H - M.bottom}
            stroke="var(--muted)"
            strokeWidth={1}
            strokeDasharray="4 4"
          />
          <text
            className="label"
            x={x(chosen.threshold) + 6}
            y={M.top - 8}
            style={{ fill: "var(--text-2)", fontWeight: 500 }}
          >
            default {similarity(chosen.threshold)}
          </text>

          <path d={path(rows, (r) => r.hit_rate, scale)} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />
          <path d={path(rows, (r) => r.wrong_rate, scale)} fill="none" stroke="var(--series-2)" strokeWidth={2} strokeLinejoin="round" />

          {/* Direct labels near the left end of each line */}
          {first && (
            <>
              <text className="label" x={x(first.threshold) + 4} y={Math.max(M.top + 12, y(first.hit_rate) - 8)}>
                Answered from cache
              </text>
              <text className="label" x={x(first.threshold) + 4} y={y(first.wrong_rate) - 8}>
                Wrong answers
              </text>
            </>
          )}

          {/* Selected threshold */}
          {current && (
            <g>
              <line
                x1={x(current.threshold)}
                x2={x(current.threshold)}
                y1={M.top}
                y2={H - M.bottom}
                stroke="var(--text)"
                strokeWidth={1}
              />
              <circle cx={x(current.threshold)} cy={y(current.hit_rate)} r={5} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
              <circle cx={x(current.threshold)} cy={y(current.wrong_rate)} r={5} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}

          {/* Hover crosshair */}
          {hover && hover !== current && (
            <line
              x1={x(hover.threshold)}
              x2={x(hover.threshold)}
              y1={M.top}
              y2={H - M.bottom}
              stroke="var(--muted)"
              strokeWidth={1}
            />
          )}

          {/* Hit target over the whole plot: hover shows values, click selects. */}
          <rect
            x={M.left}
            y={M.top}
            width={W - M.left - M.right}
            height={H - M.top - M.bottom}
            fill="transparent"
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
            onClick={() => hover && setSelected(hover.threshold)}
            style={{ cursor: "crosshair" }}
          />
        </svg>

        {hover && (
          <div
            className="tooltip"
            style={{
              left: `${(x(hover.threshold) / W) * 100}%`,
              top: `${(y(Math.max(hover.hit_rate, hover.wrong_rate)) / H) * 100}%`,
            }}
          >
            <div className="t-head">Threshold {similarity(hover.threshold)}</div>
            <div className="t-row">
              <span>
                <i className="swatch" style={{ background: "var(--series-1)" }} />
                From cache
              </span>
              {pct(hover.hit_rate)}
            </div>
            <div className="t-row">
              <span>
                <i className="swatch" style={{ background: "var(--series-2)" }} />
                Wrong
              </span>
              {pct(hover.wrong_rate)}
            </div>
            <div className="t-row">
              <span>Precision</span>
              {pct(hover.precision)}
            </div>
          </div>
        )}
      </div>

      <label className="slider">
        <span>
          Threshold <strong>{current ? similarity(current.threshold) : "–"}</strong>
        </span>
        <input
          type="range"
          min={rows[0]?.threshold ?? 0.5}
          max={rows[rows.length - 1]?.threshold ?? 0.98}
          step={0.02}
          value={current?.threshold ?? selected}
          onChange={(e) => setSelected(Number(e.target.value))}
          aria-valuetext={
            current
              ? `${similarity(current.threshold)}: ${pct(current.hit_rate)} from cache, ${pct(current.wrong_rate)} wrong`
              : undefined
          }
        />
      </label>

      {current && (
        <div className="readout">
          <div className="tile">
            <div className="label">From cache</div>
            <div className="value">{pct(current.hit_rate)}</div>
          </div>
          <div className="tile">
            <div className="label">Wrong answers</div>
            <div className="value">{pct(current.wrong_rate)}</div>
          </div>
          <div className="tile">
            <div className="label">Precision</div>
            <div className="value">{pct(current.precision)}</div>
          </div>
        </div>
      )}

      <p className="callout">
        Default <strong>{similarity(chosen.threshold)}</strong> is the lowest threshold where at
        most {pct(data.max_wrong, 0)} of queries get a wrong answer (guards on), there and at every
        stricter threshold.
        {isLive && liveThreshold !== null && (
          <> The running gateway uses {similarity(liveThreshold)}.</>
        )}{" "}
        {current && current.threshold !== chosen.threshold && (
          <button className="link-btn" type="button" onClick={() => setSelected(chosen.threshold)}>
            Reset to default
          </button>
        )}
      </p>

      <details>
        <summary>Show as a table</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="num">Threshold</th>
                <th className="num">From cache</th>
                <th className="num">Wrong</th>
                <th className="num">Precision</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.threshold}>
                  <td className="num">
                    {similarity(r.threshold)}
                    {r.threshold === chosen.threshold ? " (default)" : ""}
                  </td>
                  <td className="num">{pct(r.hit_rate)}</td>
                  <td className="num">{pct(r.wrong_rate)}</td>
                  <td className="num">{pct(r.precision)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <ModelTable data={data} />
    </div>
  );
}

function ModelTable({ data }: { data: EvalResults }) {
  return (
    <details style={{ marginTop: 8 }}>
      <summary>Compare embedding models</summary>
      <p className="callout">
        Ranked by expected latency per request: every request pays for an embedding, only misses
        pay for the LLM call (assumed {data.llm_ms} ms).
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Threshold</th>
              <th className="num">From cache</th>
              <th className="num">Wrong</th>
              <th className="num">Embed</th>
              <th className="num">Expected</th>
            </tr>
          </thead>
          <tbody>
            {data.models.map((m) => (
              <tr key={m.model}>
                <td className="query">
                  {m.model}
                  {m.model === data.best_model ? " (best)" : ""}
                </td>
                <td className="num">{similarity(m.chosen.threshold)}</td>
                <td className="num">{pct(m.chosen.hit_rate)}</td>
                <td className="num">{pct(m.chosen.wrong_rate)}</td>
                <td className="num">{m.embed_ms.toFixed(1)} ms</td>
                <td className="num">{m.expected_latency_ms.toFixed(0)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
