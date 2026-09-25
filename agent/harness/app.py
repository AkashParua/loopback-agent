"""Inference server: the only way the rest of the system talks to the LLM."""

from __future__ import annotations

import logging
import os

import httpx
import structlog
from fastapi import FastAPI, HTTPException

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
            tags = (await client.get(f"{llm.OLLAMA_URL}/api/tags")).json()
        names = [m["name"] for m in tags.get("models", [])]
        ready = llm.MODEL_NAME in names
    except httpx.HTTPError as exc:
        return {"ok": False, "model": llm.MODEL_NAME, "error": str(exc)}
    return {"ok": ready, "model": llm.MODEL_NAME, "models": names}


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
