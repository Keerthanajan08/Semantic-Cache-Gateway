import subprocess
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"

tests = [
    {
        "intent": "Password reset",
        "seed": "How do I reset my password?",
        "queries": [
            "I forgot my password, how can I reset it",
            "password recovery help",
            "I cannot remember my login password",
        ],
    },
    {
        "intent": "Account lockout",
        "seed": "Why is my account locked?",
        "queries": [
            "I am locked out after failed logins",
            "too many login attempts locked me out",
            "my account access is blocked",
        ],
    },
    {
        "intent": "Two factor authentication",
        "seed": "How do I enable two-factor authentication?",
        "queries": [
            "how do I turn on 2FA",
            "setup authenticator app",
            "enable two step verification",
        ],
    },
    {
        "intent": "Cancel subscription",
        "seed": "How do I cancel my subscription?",
        "queries": [
            "I want to stop my plan",
            "how do I disable auto renewal",
            "cancel my membership",
        ],
    },
    {
        "intent": "Refunds",
        "seed": "Do you offer refunds?",
        "queries": [
            "can I get my money back",
            "what is your refund policy",
            "I need a refund",
        ],
    },
]

thresholds = [0.90, 0.85, 0.80, 0.75, 0.70]


input(
    "Make sure no uvicorn server is running. Press Enter to start..."
)


for threshold in thresholds:

    print("\n" + "=" * 60)
    print(f"Testing threshold: {threshold}")
    print("=" * 60)

    with open(".env", "w") as f:
        f.write("LLM_PROVIDER=mock\n")
        f.write(f"CACHE_SIMILARITY_THRESHOLD={threshold}\n")

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
        for _ in range(30):
            try:
                with httpx.Client(timeout=2) as client:
                    if client.get(f"{BASE_URL}/health").status_code == 200:
                        break
            except Exception:
                time.sleep(1)

        hits = 0
        misses = 0
        total = 0

        hit_latency = []
        miss_latency = []

        with httpx.Client(timeout=60) as client:

            client.post(
                f"{BASE_URL}/query",
                json={"query": "warmup"},
            )

            for test in tests:

                # seed cache
                client.post(
                    f"{BASE_URL}/query",
                    json={"query": test["seed"]},
                )

                print(f"\n{test['intent']}")

                for query in test["queries"]:

                    start = time.perf_counter()

                    response = client.post(
                        f"{BASE_URL}/query",
                        json={"query": query},
                    )

                    latency = (
                        time.perf_counter() - start
                    ) * 1000

                    data = response.json()

                    total += 1

                    if data["cache_hit"]:
                        hits += 1
                        hit_latency.append(latency)
                        status = "HIT "
                    else:
                        misses += 1
                        miss_latency.append(latency)
                        status = "MISS"

                    print(
                        f"{status} "
                        f"similarity={data['similarity']:.3f} "
                        f"latency={latency:.1f}ms"
                    )

        print("\nSUMMARY")
        print(f"Total queries:       {total}")
        print(f"Hits:                {hits}")
        print(f"Misses:              {misses}")
        print(f"Hit rate:            {hits/total*100:.1f}%")
        print(f"LLM calls avoided:   {hits}")

        if hit_latency:
            print(
                f"Avg hit latency:     "
                f"{sum(hit_latency)/len(hit_latency):.1f} ms"
            )

        if miss_latency:
            print(
                f"Avg miss latency:    "
                f"{sum(miss_latency)/len(miss_latency):.1f} ms"
            )

    finally:
        server.terminate()
        server.wait()


print("\nBenchmark complete.")