#!/usr/bin/env bash
# Local dev without Docker: harness (:8000) + backend (:8100) + UI (:8501) against a host Ollama.
#   python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
#   ollama pull qwen3.5:4b-q4_K_M
#   scripts/dev.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
export OLLAMA_URL=${OLLAMA_URL:-http://127.0.0.1:11434}
export AGENT_URL=${AGENT_URL:-http://127.0.0.1:8000}
export BACKEND_URL=${BACKEND_URL:-http://127.0.0.1:8100}
export PYDANTIC_AI_NO_BANNER=1

trap 'kill 0' EXIT INT TERM
(cd "$ROOT/agent" && exec "$PY" -m uvicorn harness.app:app --port 8000) &
(cd "$ROOT/backend" && exec "$PY" -m uvicorn loopback.api:app --port 8100) &
(cd "$ROOT/frontend" && exec "$PY" -m streamlit run app.py --server.port 8501 --server.headless true) &
echo "UI: http://localhost:8501   backend: $BACKEND_URL/docs   agent: $AGENT_URL/docs"
wait
