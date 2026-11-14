#!/usr/bin/env bash
# Local dev without Docker: harness (:8000) + backend (:8100) + UI (:8501).
#   Apple Silicon: also starts the MLX inference server (:11435) - no Ollama needed.
#   Elsewhere:     talks to a host Ollama on $OLLAMA_URL.
#
#   uv venv --managed-python -p 3.12 .venv && uv pip install -p .venv -r requirements-dev.txt
#   (Apple Silicon: also  uv pip install -p .venv -r requirements-mac.txt)
#   scripts/dev.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}

if [[ -z "${LLM_ENGINE:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then LLM_ENGINE=mlx; else LLM_ENGINE=ollama; fi
fi
export LLM_ENGINE
export OLLAMA_URL=${OLLAMA_URL:-http://127.0.0.1:11434}
export MLX_URL=${MLX_URL:-http://127.0.0.1:11435}
export AGENT_URL=${AGENT_URL:-http://127.0.0.1:8000}
export BACKEND_URL=${BACKEND_URL:-http://127.0.0.1:8100}
export PYDANTIC_AI_NO_BANNER=1
export TOKENIZERS_PARALLELISM=false

trap 'kill 0' EXIT INT TERM
if [[ "$LLM_ENGINE" == "mlx" ]]; then
  (cd "$ROOT/agent" && exec "$PY" -m mlx_server) &
  echo -n "loading Qwen 3.5 4B on MLX "
  for _ in $(seq 1 180); do
    curl -fs "$MLX_URL/health" 2>/dev/null | grep -q '"loaded":true' && break
    echo -n "."; sleep 1
  done
  echo " ready"
fi
(cd "$ROOT/agent" && exec "$PY" -m uvicorn harness.app:app --port 8000 --log-level warning) &
(cd "$ROOT/backend" && exec "$PY" -m uvicorn loopback.api:app --port 8100 --log-level warning) &
(cd "$ROOT/frontend" && exec "$PY" -m streamlit run app.py --server.port 8501 --server.headless true \
   --server.maxUploadSize 2000) &
echo "UI: http://localhost:8501   backend: $BACKEND_URL/docs   agent: $AGENT_URL/docs   engine: $LLM_ENGINE"
wait
