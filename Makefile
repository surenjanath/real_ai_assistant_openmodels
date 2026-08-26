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

.PHONY: setup backend frontend smoke mcp n8n typecheck build

setup:
	cd $(BACKEND_DIR) && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
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
