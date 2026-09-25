#!/usr/bin/env bash
# Start Ollama, make sure the model is present (cached in the /root/.ollama volume),
# then serve the harness. Ollama itself is only reachable inside the container.
set -euo pipefail

ollama serve &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -fs "$OLLAMA_URL/api/tags" >/dev/null && break
  sleep 1
done

if ! ollama list | awk '{print $1}' | grep -qx "$LLM_MODEL"; then
  echo "pulling $LLM_MODEL (first start only)..."
  ollama pull "$LLM_MODEL"
fi

# Warm the model so the first real request is not a cold load.
curl -fs "$OLLAMA_URL/api/generate" -d "{\"model\":\"$LLM_MODEL\",\"prompt\":\"ok\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":1}}" >/dev/null || true

exec uvicorn harness.app:app --host 0.0.0.0 --port "$PORT"
