"""HTTP client for the agent container, with a rule-based fallback for every call.

The pipeline must never block on the LLM: if the agent is down, slow, or its
answer fails validation, the caller's deterministic fallback is used and the
source is reported as "rules" so the UI can say so.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import httpx
import structlog

from .config import AGENT_TIMEOUT, AGENT_URL

log = structlog.get_logger("agent_client")


class AgentClient:
    def __init__(self, base_url: str = AGENT_URL, timeout: float = AGENT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=5)
            return r.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    def task(self, name: str, context: str, *, choices: Optional[dict[str, list]] = None,
             fallback: Callable[[], dict], validate: Optional[Callable[[dict], Optional[str]]] = None,
             ) -> tuple[dict, str, Optional[str]]:
        """Returns (result, source, note). source is 'llm' or 'rules'."""
        try:
            r = httpx.post(f"{self.base_url}/v1/tasks/{name}",
                           json={"context": context, "choices": choices or {}}, timeout=self.timeout)
            r.raise_for_status()
            out = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("agent_unavailable", task=name, error=str(exc)[:200])
            return fallback(), "rules", f"agent unavailable: {str(exc)[:160]}"
        if validate:
            problem = validate(out)
            if problem:
                log.warning("agent_output_rejected", task=name, problem=problem)
                return fallback(), "rules", f"agent answer rejected: {problem}"
        return out, "llm", None

    def chat(self, messages: list[dict], context: Optional[str]) -> dict:
        r = httpx.post(f"{self.base_url}/v1/chat", json={"messages": messages, "context": context},
                       timeout=self.timeout)
        r.raise_for_status()
        return r.json()


agent = AgentClient()
