# Running on Ubuntu: snap or systemd

Two ways to run the gateway as a proper background service on Ubuntu 24.04+
(including Ubuntu on WSL with systemd enabled):

| | Snap | systemd from a checkout |
|---|---|---|
| Install | `sudo snap install` one file | `./deploy/systemd/install.sh` |
| Dependencies | bundled in the snap (Python venv, CPU torch, the model) | a `.venv` in the checkout |
| Isolation | strict confinement: only network access is granted | systemd hardening (read-only OS, private /tmp) |
| Config | `snap set` | `.env` file |
| Logs | `snap logs` | `journalctl` |
| Updates | `snap refresh` (automatic rollback on failure) | `git pull` + restart |

## Option 1: snap

### Build

```bash
sudo snap install snapcraft --classic
sudo snap install lxd && sudo lxd init --auto    # snapcraft builds inside an LXD container
sudo usermod -aG lxd "$USER" && newgrp lxd
snapcraft                                         # from the repo root; takes 10-20 min the first time
```

The output is `semantic-cache-gateway_1.0.0_amd64.snap`. CI builds and tests
the same snap on every change (`.github/workflows/snap.yml`) and uploads it as
a build artifact.

### Install and use

```bash
sudo snap install --dangerous semantic-cache-gateway_1.0.0_amd64.snap
snap services semantic-cache-gateway              # gateway: enabled, active
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' -d '{"query": "How do I reset my password?"}'
sudo snap logs -f semantic-cache-gateway
```

`--dangerous` is needed because a locally built snap isn't signed by the Snap
Store.

### Configuration

Every change is validated by the `configure` hook (a typo such as
`llm-provider=grok` is refused) and restarts the service.

| Key | Default | Maps to |
|---|---|---|
| `listen-address` | `127.0.0.1` | uvicorn `--host` (use `0.0.0.0` to expose it) |
| `port` | `8000` | uvicorn `--port` |
| `llm-provider` | `mock` | `LLM_PROVIDER` (`mock`, `groq`, `gemini`) |
| `groq-api-key` / `gemini-api-key` | unset | `GROQ_API_KEY` / `GEMINI_API_KEY` |
| `redis-url` | unset (in-memory) | `REDIS_URL` |
| `embedding-model` | `all-mpnet-base-v2` | `EMBEDDING_MODEL` |
| `similarity-threshold` | unset (evaluated default) | `CACHE_SIMILARITY_THRESHOLD` |
| `cache-guards` | `true` | `CACHE_GUARDS` |
| `admin-token` | unset | `ADMIN_TOKEN` |
| `log-level` | `INFO` | `LOG_LEVEL` |

```bash
sudo snap set semantic-cache-gateway redis-url=redis://localhost:6379/0
sudo snap set semantic-cache-gateway llm-provider=groq groq-api-key=gsk_...
sudo snap get semantic-cache-gateway
sudo snap unset semantic-cache-gateway redis-url   # back to the in-memory cache
```

### How the snap is put together

- `snap/snapcraft.yaml`: `core24` base (Ubuntu 24.04 runtime, Python 3.12),
  strict confinement. The `python` plugin creates a venv inside the snap and
  installs `requirements.txt` (CPU-only torch via `PIP_EXTRA_INDEX_URL`) and
  the `app` package. The build then downloads `all-mpnet-base-v2` into the
  snap, so the service starts offline.
- `apps.gateway` is a `daemon: simple`: snapd generates and manages a systemd
  unit for it (`snap.semantic-cache-gateway.gateway.service`), restarting it
  on failure.
- Plugs: `network` (Redis, LLM APIs) and `network-bind` (serving HTTP). Under
  strict confinement nothing else is reachable.
- `snap/local/launch`: reads `snap set` values with `snapctl get` and turns
  them into the environment variables `app/config.py` already validates, then
  starts uvicorn from `$SNAP_DATA`. A non-default embedding model is
  downloaded into `$SNAP_COMMON`, which persists across snap refreshes.
- `snap/hooks/install` records the defaults; `snap/hooks/configure` validates
  each change and restarts the service only if it's already running.

## Option 2: systemd from a checkout

```bash
git clone https://github.com/Keerthanajan08/Semantic-Cache-Gateway.git
cd Semantic-Cache-Gateway
./deploy/systemd/install.sh --redis     # omit --redis for the in-memory cache
```

The script installs packages with apt, creates `.venv`, writes `.env` from
`.env.example`, installs `deploy/systemd/semantic-cache-gateway.service` with
your user and path filled in, starts it, and waits for `/health`.

```bash
systemctl status semantic-cache-gateway
journalctl -u semantic-cache-gateway -f
sudo systemctl restart semantic-cache-gateway   # after editing .env
```

The unit runs as your user (not root), starts after Redis, restarts on
failure, and is hardened with `ProtectSystem=strict`, `PrivateTmp` and
`NoNewPrivileges`.

### WSL

systemd is off by default on WSL. Enable it once:

```bash
printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
```

Then run `wsl --shutdown` in PowerShell and reopen Ubuntu.

## Failure checks worth trying

```bash
sudo systemctl stop redis-server
curl http://127.0.0.1:8000/health          # "status": "degraded", cache_reachable false
sudo systemctl start redis-server

sudo kill -9 "$(pgrep -f 'uvicorn app.main')"
sleep 6; curl http://127.0.0.1:8000/health # systemd restarted it
ss -tlnp | grep 8000                        # which process owns the port
```
