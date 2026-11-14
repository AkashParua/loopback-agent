"""Inference server: the only way the rest of the system talks to the LLM."""

from __future__ import annotations

import logging
import os

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import llm
from .schemas import TASKS, ChatMessage, ChatRequest, ChatResponse, TaskRequest, output_model

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger("harness")

app = FastAPI(title="Loopback agent harness", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            if llm.ENGINE == "mlx":
                h = (await client.get(f"{llm.MLX_URL}/health")).json()
                return {"ok": bool(h.get("ok")), "engine": "mlx", "model": llm.MODEL_NAME,
                        "loaded": h.get("loaded"), "loading": h.get("loading")}
            tags = (await client.get(f"{llm.OLLAMA_URL}/api/tags")).json()
        names = [m["name"] for m in tags.get("models", [])]
        ready = llm.MODEL_NAME in names
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "engine": llm.ENGINE, "model": llm.MODEL_NAME, "error": str(exc)}
    return {"ok": ready, "engine": "ollama", "model": llm.MODEL_NAME, "models": names}


@app.get("/engine")
async def engine_stats() -> dict:
    """Memory / throughput telemetry of the inference engine (MLX only; Ollama has none)."""
    if llm.ENGINE != "mlx":
        return {"engine": llm.ENGINE, "model": llm.MODEL_NAME, "available": False}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            return {"engine": "mlx", "available": True, **(await client.get(f"{llm.MLX_URL}/stats")).json()}
    except (httpx.HTTPError, ValueError) as exc:
        return {"engine": "mlx", "available": False, "error": str(exc)}


@app.post("/engine/{action}")
async def engine_control(action: str) -> dict:
    if action not in ("load", "unload"):
        raise HTTPException(404, "action is load or unload")
    if llm.ENGINE != "mlx":
        raise HTTPException(400, f"{action} is only supported on the mlx engine")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{llm.MLX_URL}/admin/{action}")
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:300])
    return r.json()


@app.get("/v1/tasks")
async def list_tasks() -> dict:
    return {"tasks": TASKS}


@app.get("/v1/tasks/{task}/schema")
async def task_schema(task: str) -> dict:
    try:
        return output_model(task, {"candidate_id": [0], "columns": ["col"]}).model_json_schema()
    except KeyError:
        raise HTTPException(404, f"unknown task {task!r}; one of {TASKS}")


@app.post("/v1/tasks/{task}")
async def run_task(task: str, req: TaskRequest) -> dict:
    if task not in TASKS:
        raise HTTPException(404, f"unknown task {task!r}; one of {TASKS}")
    try:
        return await llm.run_task(task, req.context, req.choices)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # model unreachable, retries exhausted, ...
        log.error("task_failed", task=task, error=repr(exc))
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        reply = await llm.chat(req.messages, req.context)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log.error("chat_failed", error=repr(exc))
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")
    return ChatResponse(reply=reply, messages=[*req.messages, ChatMessage(role="assistant", content=reply)])


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Plain-text stream of the reply; the client appends it to its own history.

    The whole pydantic-ai stream lives inside the response generator: its context
    manager must be entered and exited in the same task."""
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(422, "last message must be from the user")

    async def body():
        try:
            async for delta in llm.chat_stream(req.messages, req.context):
                yield delta
        except Exception as exc:  # noqa: BLE001 - the stream has already started
            log.error("chat_stream_failed", error=repr(exc))
            yield f"\n\n_agent unavailable: {type(exc).__name__}_"

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
