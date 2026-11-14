"""Shared UI plumbing: backend/agent calls, styling, plotly helpers."""

from __future__ import annotations

import os
import time
from typing import Any, Iterator, Optional

import httpx
import plotly.graph_objects as go
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8100").rstrip("/")
AGENT = os.getenv("AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
LLM_TIMEOUT = 400
COLORS = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6",
          "#f97316", "#64748b"]

CSS = """
<style>
  .block-container {padding-top: 1.4rem; max-width: 1360px;}
  .muted {opacity: .7; font-size: .85rem;}
  .src {font-size: .75rem; padding: 2px 9px; border-radius: 999px; border: 1px solid rgba(128,128,128,.45);}
  .hero {padding: 26px 30px; border-radius: 16px; margin-bottom: 18px; color: white;
         background: linear-gradient(120deg, #312e81 0%, #4f46e5 45%, #0ea5e9 100%);}
  .hero h1 {color: white; margin: 0 0 6px 0; font-size: 2.1rem;}
  .hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
  .chip {display: inline-block; padding: 3px 11px; margin: 8px 6px 0 0; border-radius: 999px;
         background: rgba(255,255,255,.18); font-size: .8rem;}
  .kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 12px 16px;}
  .kpi .v {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
  .kpi .l {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em;}
  .ds-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 12px 14px; min-height: 108px;}
  .ds-card b {font-size: 1.02rem;}
  div[data-testid="stSidebar"] button {text-align: left; justify-content: flex-start;}
  section[data-testid="stFileUploaderDropzone"], div[data-testid="stFileUploaderDropzone"] {
         min-height: 150px !important; border: 2px dashed rgba(79,70,229,.6) !important;
         background: rgba(79,70,229,.06) !important; border-radius: 14px !important;}
</style>"""


def api(method: str, path: str, *, timeout: float = 30, quiet: bool = False, **kw) -> Any:
    try:
        r = httpx.request(method, f"{BACKEND}{path}", timeout=timeout, **kw)
    except httpx.HTTPError as exc:
        if not quiet:
            st.error(f"Backend unreachable at {BACKEND}: {exc}")
        return None
    if r.status_code >= 400:
        if not quiet:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:400]
            st.error(f"{method} {path} failed ({r.status_code}): {detail}")
        return None
    ctype = r.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return r.json()
    if ctype.startswith("text/"):
        return r.text
    return r.content


def agent_call(label: str, method: str, path: str, **kw) -> Any:
    t0 = time.perf_counter()
    with st.spinner(f"{label} - Qwen 3.5 4B is thinking..."):
        out = api(method, path, timeout=LLM_TIMEOUT, **kw)
    if out is not None:
        st.session_state["last_agent_s"] = round(time.perf_counter() - t0, 1)
        st.toast(f"🤖 {label}: done in {st.session_state['last_agent_s']} s", icon="⚡")
    return out


def stream_chat(messages: list[dict], context: str) -> Iterator[str]:
    """Yields reply text as the agent streams it."""
    try:
        with httpx.stream("POST", f"{AGENT}/v1/chat/stream", json={"messages": messages, "context": context},
                          timeout=LLM_TIMEOUT) as r:
            if r.status_code >= 400:
                yield f"_agent error {r.status_code}_"
                return
            yield from r.iter_text()
    except httpx.HTTPError as exc:
        yield f"_agent unavailable: {exc}_"


def style(fig: go.Figure, height: int = 300, title: Optional[str] = None, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=14)) if title else None,
                      colorway=COLORS, showlegend=legend, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None))
    fig.update_xaxes(automargin=True)
    return fig


def plot(fig: go.Figure, key: Optional[str] = None) -> None:
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "displayModeBar": False})


def kpi(col, label: str, value: Any, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="muted">{sub}</div></div>', unsafe_allow_html=True)


def gauge(value: float, target: Optional[float], title: str, higher_is_better: bool = True,
          vmax: Optional[float] = None) -> go.Figure:
    vmax = vmax or (1.0 if value <= 1 and (target or 0) <= 1 else max(value, target or 0) * 1.4)
    good = "#10b981" if target is None or (value >= target if higher_is_better else value <= target) else "#f59e0b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if target is not None else "gauge+number", value=value,
        number={"valueformat": ".4f"},
        delta={"reference": target, "increasing": {"color": "#10b981" if higher_is_better else "#ef4444"},
               "decreasing": {"color": "#ef4444" if higher_is_better else "#10b981"}} if target is not None else None,
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, vmax]}, "bar": {"color": good},
               "threshold": {"line": {"color": "#111827", "width": 3}, "value": target} if target is not None else None}))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=40, b=10))
    return fig
