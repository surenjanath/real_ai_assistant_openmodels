#!/usr/bin/env bash
# Boot the J.A.R.V.I.S. backend. Env vars override defaults (see app/config.py).
set -euo pipefail
cd "$(dirname "$0")"

HOST="${JARVIS_HOST:-0.0.0.0}"
PORT="${JARVIS_PORT:-8000}"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level warning
