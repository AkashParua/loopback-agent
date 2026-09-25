"""PydanticAI harness around the local Ollama model.

Why this shape works for a 4B model:
  * NativeOutput -> Ollama receives the JSON schema as `response_format` and, on
    engines that support it (Ollama >= 0.34 for qwen3.5), constrains decoding so
    enums / required fields cannot be violated at the token level.
  * The same schema is also put in the prompt (`template`). Older engines ignore
    `response_format` for qwen3.5; the model still sees the schema, PydanticAI
    strips markdown fences, validates, and on failure sends the validation
    error back to the model for a retry (`retries`).
  * Thinking is off (`reasoning_effort=none`): with it on, qwen3.5:4b can spend
    >1500 tokens reasoning before answering. The schema already forces the
    reasoning into short fields.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

import structlog
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.ollama import OllamaProvider

from . import prompts
from .schemas import ChatMessage, output_model

log = structlog.get_logger("harness")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("LLM_MODEL", "qwen3.5:4b-q4_K_M")
RETRIES = int(os.getenv("LLM_RETRIES", "2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))

SCHEMA_TEMPLATE = (
    "Answer with a single JSON object and nothing else (no prose, no code fence). "
    "It must match this JSON schema:\n{schema}"
)


def _settings(temperature: float, max_tokens: int = MAX_TOKENS) -> OpenAIChatModelSettings:
    # Context length is set server-side (OLLAMA_CONTEXT_LENGTH); the OpenAI-compatible
    # endpoint has no per-request num_ctx.
    return OpenAIChatModelSettings(temperature=temperature, max_tokens=max_tokens,
                                   openai_reasoning_effort="none", timeout=TIMEOUT)


@lru_cache(maxsize=1)
def model() -> OpenAIChatModel:
    return OpenAIChatModel(MODEL_NAME, provider=OllamaProvider(base_url=f"{OLLAMA_URL}/v1"))


def _task_agent(task: str, out: type[BaseModel]) -> Agent[None, Any]:
    return Agent(
        model(),
        output_type=NativeOutput(out, template=SCHEMA_TEMPLATE),
        system_prompt=prompts.system_prompt(task),
        retries=RETRIES,
        model_settings=_settings(temperature=0.2),
    )


async def run_task(task: str, context: str, choices: dict[str, list[Any]]) -> dict[str, Any]:
    out = output_model(task, choices)
    agent = _task_agent(task, out)
    user = context
    if choices:
        user += "\n\nAllowed values:\n" + "\n".join(f"- {k}: {v}" for k, v in choices.items())
    t0 = time.perf_counter()
    result = await agent.run(user)
    usage = result.usage
    log.info("task_done", task=task, ms=round((time.perf_counter() - t0) * 1000),
             in_tokens=usage.input_tokens, out_tokens=usage.output_tokens)
    return result.output.model_dump()


_chat_agent = None


def chat_agent() -> Agent[None, str]:
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = Agent(model(), system_prompt=prompts.CHAT, retries=RETRIES,
                            model_settings=_settings(temperature=0.4, max_tokens=700))
    return _chat_agent


def _to_history(messages: list[ChatMessage]) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for m in messages:
        if m.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=m.content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=m.content)]))
    return history


async def chat(messages: list[ChatMessage], context: str | None) -> str:
    """Stateless: history comes from the client every call, nothing is stored here."""
    if not messages or messages[-1].role != "user":
        raise ValueError("last message must be from the user")
    *past, last = messages
    prompt = last.content
    if context:
        prompt = f"Project context:\n{context}\n\nQuestion: {last.content}"
    result = await chat_agent().run(prompt, message_history=_to_history(past))
    return result.output
