"""
Live demo against a running gateway: paraphrased questions that should be
served from cache, plus "trap" questions that look similar but must NOT be
(the safety guards catch them).

Usage:
    uvicorn app.main:app            # terminal 1 (or: docker compose up)
    python scripts/run_demo.py      # terminal 2
    python scripts/run_demo.py --url https://your-deployment.example.com
"""

import argparse
import sys

import httpx

# Each group: the first question seeds the cache, the rest are paraphrases.
PARAPHRASE_GROUPS = [
    [
        "How do I reset my password?",
        "I forgot my password, how can I reset it",
        "forgot password help",
    ],
    [
        "Why is my account locked?",
        "my account got locked after failed logins",
        "locked out after too many login attempts",
    ],
    [
        "How do I enable two-factor authentication?",
        "how to turn on 2FA",
        "set up two factor auth on my account",
    ],
    [
        "How do I cancel my subscription?",
        "I want to cancel my plan",
        "cancel my membership",
    ],
    [
        "Do you offer refunds?",
        "can I get my money back",
        "what is your refund policy",
    ],
]

# Similar wording, different meaning: must miss even though similarity is high.
TRAPS = [
    "How do I disable two-factor authentication?",
    "Why can't I get a refund?",
    "How do I NOT reset my password?",
]


def show(client: httpx.Client, url: str, query: str) -> None:
    resp = client.post(f"{url}/query", json={"query": query})
    resp.raise_for_status()
    data = resp.json()
    status = data["cache_status"].upper().ljust(4)
    sim = data.get("similarity")
    sim_str = f"sim={sim:.3f}" if sim is not None else ""
    veto = f"  vetoed by: {', '.join(data['guard_vetoes'])}" if data["guard_vetoes"] else ""
    print(f"  [{status}] {sim_str:10} {data['latency_ms']:7.1f}ms  {query!r}{veto}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    with httpx.Client(timeout=60) as client:
        try:
            health = client.get(f"{url}/health").json()
        except httpx.HTTPError as exc:
            sys.exit(f"Could not reach the gateway at {url}: {exc}")
        print(
            f"Connected: provider={health['llm_provider']} cache={health['cache_backend']} "
            f"model={health['embedding_model']}\n"
        )

        for i, group in enumerate(PARAPHRASE_GROUPS, start=1):
            print(f"--- Group {i} ---")
            for q in group:
                show(client, url, q)
            print()

        print("--- Traps (similar wording, different meaning) ---")
        for q in TRAPS:
            show(client, url, q)
        print()

        s = client.get(f"{url}/stats").json()
        print("=== Summary ===")
        print(f"Queries:              {s['total_queries']}")
        print(f"Hit rate:             {s['hit_rate'] * 100:.1f}%")
        print(f"LLM calls saved:      {s['estimated_llm_calls_saved']}")
        print(f"Guard vetoes:         {s['guard_vetoes']}")
        hit = f"{s['p50_hit_latency_ms']:.1f} / {s['p95_hit_latency_ms']:.1f} ms"
        miss = f"{s['p50_miss_latency_ms']:.1f} / {s['p95_miss_latency_ms']:.1f} ms"
        print(f"Latency hit  p50/p95: {hit}")
        print(f"Latency miss p50/p95: {miss}")


if __name__ == "__main__":
    main()
