# Agent container: Ollama (Qwen 3.5 4B Q4_K_M) + PydanticAI harness behind FastAPI.
FROM ollama/ollama:0.34.4

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY harness ./harness
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_URL=http://127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=10m \
    LLM_MODEL=qwen3.5:4b-q4_K_M \
    OLLAMA_CONTEXT_LENGTH=8192 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
  CMD curl -fs http://127.0.0.1:8000/health | grep -q '"ok":true' || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD []
