"""OpenAI-compatible endpoints over the MLX engine (what pydantic-ai's OpenAIChatModel speaks).

  POST /v1/chat/completions   messages, max_tokens, temperature, response_format, stream
  GET  /v1/models
  GET  /health                 server up + whether the model is resident
  GET  /stats                  memory, throughput history, prefix-cache hits
  POST /admin/load | /admin/unload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import MODEL_ID, MemoryPressure, engine

structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger("mlx_server")

app = FastAPI(title="Loopback MLX server", version="1.0.0")
DEFAULT_MAX_TOKENS = int(os.getenv("MLX_DEFAULT_MAX_TOKENS", "1024"))


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.2
    response_format: Optional[dict[str, Any]] = None
    stream: bool = False
    chat_template_kwargs: Optional[dict[str, Any]] = None

    model_config = {"extra": "allow"}  # reasoning_effort, top_p, ... are accepted and ignored


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _messages(raw: list[dict]) -> list[dict]:
    """Qwen's template allows one system turn, first. pydantic-ai may send several
    (system prompt + output-schema instructions), so they are merged."""
    system, out = [], []
    for m in raw:
        role = m.get("role", "user")
        role = "system" if role == "developer" else role
        role = role if role in ("system", "user", "assistant") else "user"
        if role == "system":
            system.append(_text(m.get("content")))
        else:
            out.append({"role": role, "content": _text(m.get("content"))})
    return ([{"role": "system", "content": "\n\n".join(system)}] if system else []) + out


def _schema(rf: Optional[dict]) -> Optional[dict]:
    if not rf:
        return None
    if rf.get("type") == "json_schema":
        return (rf.get("json_schema") or {}).get("schema") or {"type": "object"}
    if rf.get("type") == "json_object":
        return {"type": "object"}
    return None


async def _stream(req: ChatCompletionRequest):
    """Bridge the MLX thread's generator into an async iterator."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    msgs = _messages(req.messages)
    max_tokens = req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS
    temp = 0.2 if req.temperature is None else float(req.temperature)
    schema = _schema(req.response_format)

    def work():
        try:
            for item in engine.generate(msgs, max_tokens=max_tokens, temperature=temp, schema=schema,
                                        template_kwargs=req.chat_template_kwargs):
                loop.call_soon_threadsafe(q.put_nowait, item)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            loop.call_soon_threadsafe(q.put_nowait, {"error": exc})
        loop.call_soon_threadsafe(q.put_nowait, None)

    engine.submit(work)
    while (item := await q.get()) is not None:
        yield item


def _usage(stats: dict) -> dict:
    return {"prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["gen_tokens"],
            "total_tokens": stats["prompt_tokens"] + stats["gen_tokens"],
            "prompt_tokens_details": {"cached_tokens": stats["cached_tokens"]}}


def _raise(exc: Exception):
    if isinstance(exc, MemoryPressure):
        raise HTTPException(503, str(exc))
    raise HTTPException(500, f"{type(exc).__name__}: {exc}")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = req.model or MODEL_ID

    if not req.stream:
        parts: list[str] = []
        stats: dict = {}
        async for item in _stream(req):
            if "error" in item:
                _raise(item["error"])
            if "text" in item:
                parts.append(item["text"])
            if "done" in item:
                stats = item["done"]
        return {"id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "finish_reason": stats.get("finish", "stop"),
                             "message": {"role": "assistant", "content": "".join(parts)}}],
                "usage": _usage(stats), "x_stats": stats}

    async def sse():
        def chunk(delta: dict, finish: Optional[str] = None, **extra) -> str:
            body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
            return f"data: {json.dumps(body)}\n\n"

        yield chunk({"role": "assistant", "content": ""})
        async for item in _stream(req):
            if "error" in item:
                yield f"data: {json.dumps({'error': {'message': str(item['error'])}})}\n\n"
                break
            if "text" in item:
                yield chunk({"content": item["text"]})
            if "done" in item:
                yield chunk({}, item["done"]["finish"], usage=_usage(item["done"]), x_stats=item["done"])
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mlx-community"}]}


@app.get("/health")
async def health():
    st = engine.state
    return {"ok": True, "engine": "mlx", "model": MODEL_ID, "loaded": st.loaded, "loading": st.loading}


@app.get("/stats")
async def stats():
    return engine.stats()


@app.post("/admin/load")
async def load():
    try:
        await asyncio.wrap_future(engine.submit(engine.ensure_loaded))
    except Exception as exc:  # noqa: BLE001
        _raise(exc)
    return engine.stats()


@app.post("/admin/unload")
async def unload():
    freed = await asyncio.wrap_future(engine.submit(engine.unload, "admin"))
    return {"unloaded": freed, **engine.stats()}


@app.on_event("startup")
async def warm():
    if os.getenv("MLX_PRELOAD", "1") == "1":
        # load in the background (requests queue behind it), then one short generation so
        # Metal kernels are compiled before the first real request
        def preload():
            engine.ensure_loaded()
            for _ in engine.generate([{"role": "user", "content": "Reply with: ready"}], max_tokens=8,
                                     temperature=0.0, schema=None):
                pass
        engine.submit(preload)
