"""
Pick the cache similarity threshold from data instead of guessing.

Uses a hand-labelled set of support questions (eval/intents.json) and the
real cache code. Evaluation is leave-one-out on a warm cache: for each
question, every OTHER question is already cached, and we check what the
gateway would serve. That models a long-running gateway, and unlike
replaying random orderings it's deterministic, so results are reproducible.

A hit is CORRECT if the matched question has the same intent, WRONG if it
doesn't (the user gets someone else's answer). Per threshold, with and
without the safety guards:
  * hit rate   - share of questions answered from cache (LLM calls saved)
  * precision  - share of hits that were correct
  * wrong rate - share of ALL questions answered with a wrong cached answer

Then separately checks eval/guard_probes.json: near-identical pairs that
must never share an answer (negations, different numbers).

Usage:
    python scripts/evaluate_threshold.py                      # configured model
    python scripts/evaluate_threshold.py --models all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5

With several models it also compares them (hit rate at the wrong-answer
budget, per-query embedding latency) and details the best one.
Writes eval/results.md, eval/results.json (for the dashboard) and eval/threshold_sweep.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cache import InMemoryCache  # noqa: E402
from app.config import RECOMMENDED_THRESHOLDS, settings  # noqa: E402
from app.embeddings import Embedder, Vector, cosine_similarity  # noqa: E402
from app.guards import veto_reason  # noqa: E402

EVAL_DIR = ROOT / "eval"
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.50, 0.981, 0.02)]


@dataclass
class SweepRow:
    threshold: float
    guards: bool
    hit_rate: float
    wrong_rate: float
    precision: float


@dataclass
class ModelResult:
    model: str
    embed_ms: float
    rows: list[SweepRow]
    chosen: SweepRow


def load_intents() -> dict[str, list[str]]:
    data = json.loads((EVAL_DIR / "intents.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_loo_caches(queries: list[str], vectors: dict[str, Vector]) -> list[InMemoryCache]:
    """caches[i] holds every question except queries[i]."""
    caches = []
    for i in range(len(queries)):
        cache = InMemoryCache(threshold=1.0, max_entries=len(queries))
        for j, q in enumerate(queries):
            if j != i:
                cache.store(q, "answer", vectors[q])
        caches.append(cache)
    return caches


def sweep(intents: dict[str, list[str]], vectors: dict[str, Vector]) -> list[SweepRow]:
    labelled = [(q, intent) for intent, qs in intents.items() for q in qs]
    queries = [q for q, _ in labelled]
    intent_of = dict(labelled)
    caches = build_loo_caches(queries, vectors)

    rows = []
    for guards in (False, True):
        for t in THRESHOLDS:
            hits = wrong = 0
            for (query, intent), cache in zip(labelled, caches, strict=True):
                cache.threshold = t
                accept = (lambda e, q=query: veto_reason(q, e.query) is None) if guards else None
                result = cache.lookup(vectors[query], accept=accept)
                if result.hit and result.entry is not None:
                    hits += 1
                    wrong += intent_of[result.entry.query] != intent
            rows.append(
                SweepRow(
                    threshold=t,
                    guards=guards,
                    hit_rate=hits / len(labelled),
                    wrong_rate=wrong / len(labelled),
                    precision=(hits - wrong) / hits if hits else 1.0,
                )
            )
    return rows


def choose(rows: list[SweepRow], max_wrong: float) -> SweepRow:
    """
    Lowest threshold (= most cache reuse) with guards on whose wrong-answer
    rate is within budget there AND at every stricter threshold, so the
    default can't sit on a lucky dip just before things get worse.
    """
    guarded = sorted((r for r in rows if r.guards), key=lambda r: r.threshold)
    for i, row in enumerate(guarded):
        if all(r.wrong_rate <= max_wrong for r in guarded[i:]):
            return row
    return guarded[-1]


def probe_guards(
    embedder: Embedder, threshold: float
) -> list[tuple[str, str, float, bool, str | None]]:
    data = json.loads((EVAL_DIR / "guard_probes.json").read_text(encoding="utf-8"))
    out = []
    for a, b in data["pairs"]:
        sim = cosine_similarity(embedder.encode(a), embedder.encode(b))
        reason = veto_reason(b, a)
        out.append((a, b, sim, sim >= threshold, reason))
    return out


def plot(rows: list[SweepRow], chosen: SweepRow, model: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for guards, style in ((False, "--"), (True, "-")):
        sel = [r for r in rows if r.guards == guards]
        x = [r.threshold for r in sel]
        label = "guards on" if guards else "guards off"
        ax.plot(
            x, [r.hit_rate * 100 for r in sel], style, color="#2563eb", label=f"hit rate ({label})"
        )
        ax.plot(
            x,
            [r.wrong_rate * 100 for r in sel],
            style,
            color="#dc2626",
            label=f"wrong answers ({label})",
        )
    ax.axvline(chosen.threshold, color="#6b7280", linewidth=1)
    ax.annotate(
        f"default {chosen.threshold}",
        (chosen.threshold, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        color="#374151",
        fontsize=9,
    )
    ax.set_xlabel("cosine similarity threshold")
    ax.set_ylabel("% of all queries")
    ax.set_title(f"Semantic cache: reuse vs. wrong answers ({model})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_report(
    results: list[ModelResult],
    best: ModelResult,
    probes: list[tuple[str, str, float, bool, str | None]],
    n_queries: int,
    n_intents: int,
    max_wrong: float,
    llm_ms: float,
    has_plot: bool,
) -> str:
    rows, chosen = best.rows, best.chosen
    lines = [
        "# Threshold evaluation",
        "",
        f"Generated by `scripts/evaluate_threshold.py`: "
        f"{n_queries} hand-labelled questions across {n_intents} intents "
        "(`eval/intents.json`), evaluated leave-one-out on a warm cache: for each "
        "question, all the others are already cached.",
        "",
        "A hit is **wrong** when the matched question belongs to a different intent, "
        "i.e. the user receives someone else's answer. *Precision* is the share of hits "
        "that were right; *wrong* is out of all queries. Several intents are deliberate "
        "near-misses (reset vs. change password, enable vs. disable 2FA, cancel plan vs. "
        "delete account).",
        "",
    ]
    if len(results) > 1:
        lines += [
            "## Embedding models",
            "",
            "Each model at its own default threshold (most reuse with guards on while at "
            f"most {max_wrong:.0%} of queries get a wrong answer). Latency is one "
            "query on CPU. Models are ranked by **expected latency per request**: every "
            "request pays for an embedding, but only misses pay for the LLM call "
            f"(assumed {llm_ms:.0f} ms). A slower model that avoids more LLM calls can "
            "be faster overall, and cheaper.",
            "",
            "| Model | Threshold | Hit rate | Precision | Wrong answers | Embed latency "
            "| Expected latency |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sorted(results, key=lambda r: expected_latency_ms(r, llm_ms)):
            name = f"**{r.model}**" if r is best else r.model
            lines.append(
                f"| {name} | {r.chosen.threshold} | {r.chosen.hit_rate:.1%} | "
                f"{r.chosen.precision:.1%} | {r.chosen.wrong_rate:.1%} | {r.embed_ms:.1f} ms | "
                f"{expected_latency_ms(r, llm_ms):.0f} ms |"
            )
        lines.append("")
    lines += [
        f"## Threshold sweep: `{best.model}`",
        "",
        f"**Chosen default: `{chosen.threshold}`**: the lowest threshold where at most "
        f"{max_wrong:.0%} of queries get a wrong answer (guards on), there and at every "
        "stricter threshold.",
        "",
    ]
    if has_plot:
        lines += ["![threshold sweep](threshold_sweep.png)", ""]
    lines += [
        "| Threshold | Hit rate (guards off) | Wrong (guards off) | Hit rate (guards on) "
        "| Wrong (guards on) | Precision (guards on) |",
        "|---|---|---|---|---|---|",
    ]
    by_t = {(r.threshold, r.guards): r for r in rows}
    for t in THRESHOLDS:
        off, on = by_t[(t, False)], by_t[(t, True)]
        mark = " ⬅ default" if t == chosen.threshold else ""
        lines.append(
            f"| {t:.3f}{mark} | {off.hit_rate:.1%} | {off.wrong_rate:.1%} | "
            f"{on.hit_rate:.1%} | {on.wrong_rate:.1%} | {on.precision:.1%} |"
        )
    lines += [
        "",
        "## Guard probes",
        "",
        "Pairs that must never share an answer (`eval/guard_probes.json`). "
        f'"Would hit" means similarity clears the {chosen.threshold} threshold, so without '
        "guards the second question would be served the first one's answer.",
        "",
        "| Cached question | Incoming question | Similarity | Would hit | Guard |",
        "|---|---|---|---|---|",
    ]
    for a, b, sim, would_hit, reason in probes:
        verdict = (
            f"blocked ({reason})" if reason else ("**served wrong answer**" if would_hit else "-")
        )
        lines.append(f"| {a} | {b} | {sim:.3f} | {'yes' if would_hit else 'no'} | {verdict} |")
    lines.append("")
    return "\n".join(lines)


def evaluate_model(
    model: str, intents: dict[str, list[str]], max_wrong: float
) -> tuple[ModelResult, Embedder]:
    print(f"evaluating {model} ...", file=sys.stderr)
    embedder = Embedder(model)
    queries = [q for qs in intents.values() for q in qs]
    vectors = dict(zip(queries, embedder.encode_batch(queries), strict=True))

    embedder.encode("warm-up")
    t0 = time.perf_counter()
    for q in queries:
        embedder.encode(q)
    embed_ms = (time.perf_counter() - t0) * 1000 / len(queries)

    rows = sweep(intents, vectors)
    return ModelResult(model, embed_ms, rows, choose(rows, max_wrong)), embedder


def expected_latency_ms(result: ModelResult, llm_ms: float) -> float:
    """Every request pays for an embedding; only misses pay for the LLM."""
    return result.embed_ms + (1 - result.chosen.hit_rate) * llm_ms


def pick_best(
    evaluated: list[tuple[ModelResult, Embedder]], llm_ms: float
) -> tuple[ModelResult, Embedder]:
    return min(evaluated, key=lambda pair: expected_latency_ms(pair[0], llm_ms))


def results_json(
    results: list[ModelResult],
    best: ModelResult,
    probes: list[tuple[str, str, float, bool, str | None]],
    n_queries: int,
    n_intents: int,
    max_wrong: float,
    llm_ms: float,
) -> dict[str, object]:
    """Same data as results.md, machine-readable (the dashboard's threshold explorer reads it)."""

    def row(r: SweepRow) -> dict[str, object]:
        return {
            "threshold": r.threshold,
            "guards": r.guards,
            "hit_rate": round(r.hit_rate, 4),
            "wrong_rate": round(r.wrong_rate, 4),
            "precision": round(r.precision, 4),
        }

    return {
        "n_queries": n_queries,
        "n_intents": n_intents,
        "max_wrong": max_wrong,
        "llm_ms": llm_ms,
        "best_model": best.model,
        "models": [
            {
                "model": r.model,
                "embed_ms": round(r.embed_ms, 1),
                "expected_latency_ms": round(expected_latency_ms(r, llm_ms), 1),
                "chosen": row(r.chosen),
                "sweep": [row(s) for s in r.rows],
            }
            for r in sorted(results, key=lambda r: expected_latency_ms(r, llm_ms))
        ],
        "guard_probes": [
            {
                "cached": a,
                "incoming": b,
                "similarity": round(sim, 3),
                "would_hit": would_hit,
                "blocked_by": reason,
            }
            for a, b, sim, would_hit, reason in probes
        ],
    }


def check_recommended(models: list[str], intents: dict[str, list[str]], max_wrong: float) -> int:
    failures = 0
    for model in models:
        result, _ = evaluate_model(model, intents, max_wrong)
        shipped = RECOMMENDED_THRESHOLDS.get(model)
        ok = shipped == result.chosen.threshold
        failures += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {model}: evaluated {result.chosen.threshold}, "
            f"shipped {shipped} (hit rate {result.chosen.hit_rate:.1%}, "
            f"wrong {result.chosen.wrong_rate:.1%})"
        )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=[settings.embedding_model])
    parser.add_argument(
        "--max-wrong",
        type=float,
        default=0.01,
        help="share of ALL queries allowed to receive a wrong cached answer",
    )
    parser.add_argument(
        "--llm-ms",
        type=float,
        default=800.0,
        help="typical latency of one LLM call, used to rank models by expected latency",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI mode: verify RECOMMENDED_THRESHOLDS still matches the evaluation; write nothing",
    )
    args = parser.parse_args()

    intents = load_intents()
    if args.check:
        sys.exit(check_recommended(args.models, intents, args.max_wrong))
    n_queries = sum(len(v) for v in intents.values())
    evaluated = [evaluate_model(m, intents, args.max_wrong) for m in args.models]
    results = [r for r, _ in evaluated]
    best, best_embedder = pick_best(evaluated, args.llm_ms)

    probes = probe_guards(best_embedder, best.chosen.threshold)
    has_plot = plot(best.rows, best.chosen, best.model, EVAL_DIR / "threshold_sweep.png")
    report = write_report(
        results,
        best,
        probes,
        n_queries,
        len(intents),
        args.max_wrong,
        args.llm_ms,
        has_plot,
    )
    (EVAL_DIR / "results.md").write_text(report, encoding="utf-8")
    data = results_json(results, best, probes, n_queries, len(intents), args.max_wrong, args.llm_ms)
    (EVAL_DIR / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
