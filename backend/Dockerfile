# Backend: data -> context -> suggest -> adapt -> train -> monitor -> eval -> report.
# Build context is the repo root (it also ships scripts/fetch_datasets.py).
FROM python:3.12-slim

# CPU wheels by default. GPU: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/loopback backend/loopback
COPY scripts scripts

ENV PYTHONPATH=/app/backend \
    LOOPBACK_WORKSPACE=/workspace \
    LOOPBACK_DATASETS=/datasets \
    AGENT_URL=http://agent:8000 \
    YOLO_CONFIG_DIR=/workspace/.ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fs http://127.0.0.1:8100/health || exit 1
CMD ["uvicorn", "loopback.api:app", "--host", "0.0.0.0", "--port", "8100"]
