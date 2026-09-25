"""Hits a real agent. Run with: LIVE_AGENT_URL=http://127.0.0.1:8000 pytest -m live"""

import os

import httpx
import pytest

URL = os.getenv("LIVE_AGENT_URL")
pytestmark = [pytest.mark.live, pytest.mark.skipif(not URL, reason="set LIVE_AGENT_URL")]


def test_structured_task_returns_valid_enum():
    r = httpx.post(f"{URL}/v1/tasks/suggest", timeout=400, json={
        "context": "DATA tabular 300 rows, classify y (50/50)\nCANDIDATES\n0: logreg\n1: mlp",
        "choices": {"candidate_id": [0, 1]}})
    assert r.status_code == 200, r.text
    assert all(p["candidate_id"] in (0, 1) for p in r.json()["picks"])


def test_chat_is_stateless():
    msgs = [{"role": "user", "content": "Reply with the single word: ready"}]
    r = httpx.post(f"{URL}/v1/chat", json={"messages": msgs}, timeout=400)
    assert r.status_code == 200
    body = r.json()
    assert body["messages"][:1] == msgs and body["messages"][-1]["role"] == "assistant"
