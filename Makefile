# J.A.R.V.I.S. - developer entrypoints
#
#   make setup     one-time: python venv + npm install
#   make backend   fastapi  -> http://localhost:8000
#   make frontend  next.js  -> http://localhost:3000 (proxies /ws/* + /api/*)
#   make smoke     end-to-end check against a running backend
#   make mcp       FastMCP tool server (stdio)
#   make n8n       docker compose n8n automation instance -> :5678

SHELL := /usr/bin/env bash
BACKEND_DIR := backend
FRONTEND_DIR := frontend

# The Kokoro voice stack has no wheels for Python 3.13+ (`kokoro` requires
# <3.13, `kokoro-onnx` <3.14), so a newer interpreter silently degrades the
# assistant to the robotic fallback synth. Pin 3.12 explicitly.
PY_VERSION := 3.12

.PHONY: setup setup-backend setup-frontend backend frontend smoke mcp n8n typecheck build clean-venv

setup: setup-backend setup-frontend

setup-backend:
	@set -euo pipefail; cd $(BACKEND_DIR); \
	if command -v uv >/dev/null 2>&1; then \
	  echo "==> uv: creating python $(PY_VERSION) venv"; \
	  uv venv --python $(PY_VERSION) .venv; \
	  uv pip install --python ./.venv/bin/python -r requirements.txt; \
	else \
	  PY=$$(command -v python$(PY_VERSION) || true); \
	  if [ -z "$$PY" ]; then \
	    echo "ERROR: python$(PY_VERSION) not found and uv is not installed."; \
	    echo "       The Kokoro voice needs Python $(PY_VERSION); newer versions have no wheels."; \
	    echo "       Install uv (https://astral.sh/uv) or 'brew install python@$(PY_VERSION)'."; \
	    exit 1; \
	  fi; \
	  echo "==> $$PY: creating venv"; \
	  "$$PY" -m venv .venv; \
	  ./.venv/bin/pip install -U pip; \
	  ./.venv/bin/pip install -r requirements.txt; \
	fi; \
	./.venv/bin/python -c "import sys; v=sys.version_info; \
	  print(f'backend python {v.major}.{v.minor}.{v.micro}'); \
	  sys.exit(0 if v[:2]==(3,12) else 1)" \
	  || { echo 'ERROR: backend venv is not python $(PY_VERSION)'; exit 1; }
	@echo "==> verifying the Kokoro voice engine imports"
	@cd $(BACKEND_DIR) && ./.venv/bin/python -c "\
import pykokoro, sys; print('pykokoro', pykokoro.__version__, 'OK')" \
	  || echo "WARNING: pykokoro unavailable - the assistant will use the fallback synth"

setup-frontend:
	cd $(FRONTEND_DIR) && npm install

backend:
	cd $(BACKEND_DIR) && ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level warning

frontend:
	cd $(FRONTEND_DIR) && PORT=3000 HOSTNAME=0.0.0.0 node server.mjs

smoke:
	cd $(BACKEND_DIR) && ./.venv/bin/python scripts/smoke.py --url http://127.0.0.1:8000

mcp:
	cd $(BACKEND_DIR) && ./.venv/bin/python tools/mcp_server.py

n8n:
	cd deploy/n8n && docker compose up -d

typecheck:
	cd $(FRONTEND_DIR) && npx tsc --noEmit

build:
	cd $(FRONTEND_DIR) && npm run build

clean-venv:
	rm -rf $(BACKEND_DIR)/.venv
