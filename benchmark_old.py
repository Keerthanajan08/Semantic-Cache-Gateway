import os
import subprocess
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"

queries = [
    "How do I reset my password?",
    "I forgot my password, how can I reset it",
    "forgot password help",
    "what's the process to change a forgotten password",
]

thresholds = [0.90, 0.85, 0.80, 0.75, 0.70]

print("Make sure no uvicorn server is running before starting this benchmark.")
input("Press Enter to start...")

for threshold in thresholds:
    print("\n" + "=" * 60)
    print(f"Testing threshold: {threshold}")
    print("=" * 60)

    # Create a fresh .env for this run.
    with open(".env", "w") as f:
        f.write(f"LLM_PROVIDER=mock\n")
        f.write(f"CACHE_SIMILARITY_THRESHOLD={threshold}\n")

    # Start a fresh server so cache + configuration are reset.
    server = subprocess.Popen(
        [
            "venv\\Scripts\\python.exe",
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            "8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server.
        for _ in range(30):
            try:
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(f"{BASE_URL}/health")
                    if response.status_code == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            print("Server failed to start.")
            continue

        with httpx.Client(timeout=60.0) as client:

            # Warm up embedding model.
            client.post(
                f"{BASE_URL}/query",
                json={"query": "warmup query"},
            )

            # Seed cache with the canonical question.
            client.post(
                f"{BASE_URL}/query",
                json={"query": queries[0]},
            )

            hits = 0
            misses = 0
            hit_latencies = []
            miss_latencies = []

            # Test paraphrases against the SAME single cached answer.
            for query in queries[1:]:

                start = time.perf_counter()

                response = client.post(
                    f"{BASE_URL}/query",
                    json={"query": query},
                )

                latency = (time.perf_counter() - start) * 1000
                data = response.json()

                if data["cache_hit"]:
                    hits += 1
                    hit_latencies.append(latency)
                    status = "HIT "
                else:
                    misses += 1
                    miss_latencies.append(latency)
                    status = "MISS"

                print(
                    f"[{status}] "
                    f"similarity={data['similarity']:.3f} "
                    f"latency={latency:.1f} ms | {query}"
                )

            total = hits + misses

            print()
            print(f"Hits:              {hits}")
            print(f"Misses:            {misses}")
            print(f"Hit rate:          {hits / total * 100:.1f}%")

            if hit_latencies:
                print(
                    f"Avg hit latency:   "
                    f"{sum(hit_latencies) / len(hit_latencies):.1f} ms"
                )

            if miss_latencies:
                print(
                    f"Avg miss latency:  "
                    f"{sum(miss_latencies) / len(miss_latencies):.1f} ms"
                )

            print(f"LLM calls saved:    {hits}")

    finally:
        server.terminate()
        server.wait()

print("\n" + "=" * 60)
print("Benchmark complete.")
print("=" * 60)