#!/usr/bin/env bash
# Installs the gateway as a systemd service on Ubuntu, from this checkout.
#
#   ./deploy/systemd/install.sh            # mock LLM, in-memory cache
#   ./deploy/systemd/install.sh --redis    # also installs Redis and uses it
#
# Run it as your normal user (it calls sudo where needed), from anywhere.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_NAME="semantic-cache-gateway"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}.service"
USE_REDIS=false
[ "${1:-}" = "--redis" ] && USE_REDIS=true

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as your normal user, not root (the service runs as that user)." >&2
  exit 1
fi
if ! pidof systemd >/dev/null 2>&1 && [ ! -d /run/systemd/system ]; then
  echo "systemd isn't running. On WSL, add '[boot] systemd=true' to /etc/wsl.conf," >&2
  echo "then run 'wsl --shutdown' in PowerShell and reopen Ubuntu." >&2
  exit 1
fi

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
  echo "Needs Python 3.12+ (Ubuntu 24.04 or later); found $(python3 --version)." >&2
  exit 1
fi

echo "==> Installing system packages"
packages=(python3-venv python3-pip curl)
$USE_REDIS && packages+=(redis-server)
sudo apt-get update -qq
sudo apt-get install -y -qq "${packages[@]}"

if $USE_REDIS; then
  echo "==> Starting Redis"
  sudo systemctl enable --now redis-server
  redis-cli ping
fi

echo "==> Creating virtualenv and installing dependencies (CPU-only torch)"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$APP_DIR/requirements.txt"

if [ ! -f "$APP_DIR/.env" ]; then
  echo "==> Creating .env from .env.example"
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi
if $USE_REDIS; then
  sed -i 's|^REDIS_URL=.*|REDIS_URL=redis://localhost:6379/0|' "$APP_DIR/.env"
fi

echo "==> Installing ${UNIT_PATH}"
sed -e "s|@USER@|$(id -un)|g" -e "s|@APP_DIR@|${APP_DIR}|g" \
  "$APP_DIR/deploy/systemd/${UNIT_NAME}.service" | sudo tee "$UNIT_PATH" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT_NAME"
sudo systemctl restart "$UNIT_NAME"

echo "==> Waiting for the service to load the model (first start can take a minute)"
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:8000/health
    echo
    echo "Done. Logs: journalctl -u ${UNIT_NAME} -f"
    exit 0
  fi
  sleep 3
done

echo "The service didn't become healthy in time. Recent logs:" >&2
sudo journalctl -u "$UNIT_NAME" -n 50 --no-pager >&2
exit 1
