"""
Fires a batch of queries at a running instance of the gateway —
several semantically-distinct questions, each phrased multiple different
ways — and prints a summary. This is what generates the "X% cache hit
rate / Yms average latency saved" numbers worth putting on a resume.

Usage:
    1. In one terminal: uvicorn app.main:app --reload
    2. In another:       python scripts/run_demo.py
"""
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"

# Each inner list is paraphrases of the *same* underlying question.
# First hit in each group is a guaranteed cache miss; the rest should
# increasingly hit the cache.
QUERY_GROUPS = [
    [
        "How do I reset my password?",
        "I forgot my password, how can I reset it",
        "forgot password help",
        "what's the process to change a forgotten password",
    ],
    [
        "Why am I locked out of my account?",
        "my account got locked after failed logins",
        "locked out too many login attempts, what now",
    ],
    [
        "How do I enable two-factor authentication?",
        "how to turn on 2FA",
        "setting up two factor auth on my account",
    ],
    [
        "How do I cancel my subscription?",
        "I want to cancel my plan",
        "how to stop auto-renewal",
    ],
    [
        "Do you offer refunds?",
        "can I get my money back",
        "refund policy question",
    ],
]


def run():
    with httpx.Client(timeout=30) as client:
        health = client.get(f"{BASE_URL}/health").json()
        print(f"Connected. LLM provider: {health['llm_provider']}\n")

        for group_idx, group in enumerate(QUERY_GROUPS, start=1):
            print(f"--- Group {group_idx} ---")
            for q in group:
                t0 = time.perf_counter()
                resp = client.post(f"{BASE_URL}/query", json={"query": q}).json()
                elapsed = (time.perf_counter() - t0) * 1000
                hit_marker = "HIT " if resp["cache_hit"] else "MISS"
                sim = resp.get("similarity")
                sim_str = f"sim={sim:.3f}" if sim is not None else ""
                print(f"  [{hit_marker}] {sim_str:12} {elapsed:6.1f}ms  \"{q}\"")
            print()

        stats = client.get(f"{BASE_URL}/stats").json()
        print("=== Summary ===")
        print(f"Total queries:            {stats['total_queries']}")
        print(f"Cache hits:               {stats['cache_hits']}")
        print(f"Cache misses:             {stats['cache_misses']}")
        print(f"Hit rate:                 {stats['hit_rate'] * 100:.1f}%")
        print(f"Avg latency on hit:       {stats['avg_hit_latency_ms']:.1f}ms")
        print(f"Avg latency on miss:      {stats['avg_miss_latency_ms']:.1f}ms")
        if stats["avg_miss_latency_ms"]:
            speedup = stats["avg_miss_latency_ms"] / max(stats["avg_hit_latency_ms"], 0.01)
            print(f"Speedup on cache hit:     {speedup:.1f}x")
        print(f"Estimated LLM calls saved: {stats['estimated_llm_calls_saved']}")


if __name__ == "__main__":
    run()
