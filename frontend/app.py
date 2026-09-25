"""Loopback UI - one screen per step. Left = facts, right = agent + Approve / Reject.

The user drives every transition; nothing advances on its own. Chat history is
kept client-side (this browser session) and can be downloaded / restored.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import httpx
import pandas as pd
import streamlit as st
import yaml

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8100").rstrip("/")
AGENT = os.getenv("AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
STEPS = ["Data", "Context", "Suggest", "Adapt", "Train", "Watch", "Eval", "Report"]
LLM_TIMEOUT = 400

st.set_page_config(page_title="Loopback", page_icon="🔁", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1280px;}
  .agent-card {border: 1px solid rgba(128,128,128,.35); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;}
  .muted {opacity: .7; font-size: .85rem;}
  .src {font-size: .75rem; padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(128,128,128,.5);}
  div[data-testid="stSidebar"] button {text-align: left; justify-content: flex-start;}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("pid", None)
ss.setdefault("step", 1)
ss.setdefault("chat", [])


# ================================================================== API helpers

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
    with st.spinner(f"{label} - Qwen 3.5 4B is thinking (10-60 s)..."):
        return api(method, path, timeout=LLM_TIMEOUT, **kw)


def goto(step: int) -> None:
    ss.step = step
    st.rerun()


def source_badge(source: Optional[str], note: Optional[str] = None) -> None:
    if source == "llm":
        st.markdown('<span class="src">🤖 Qwen 3.5 4B</span>', unsafe_allow_html=True)
    elif source:
        st.markdown('<span class="src">📏 rule-based fallback</span>', unsafe_allow_html=True)
    if note:
        st.caption(f"Why fallback: {note}")


def run_image(rid: str, name: str, caption: str = "") -> None:
    data = api("GET", f"/runs/{rid}/files/{name}", quiet=True)
    if isinstance(data, (bytes, bytearray)):
        st.image(data, caption=caption or name, width="stretch")


def project() -> Optional[dict]:
    return api("GET", f"/projects/{ss.pid}") if ss.pid else None


# ================================================================== sidebar

def sidebar(p: Optional[dict]) -> None:
    with st.sidebar:
        st.markdown("## 🔁 Loopback")
        health = api("GET", "/health", quiet=True, timeout=6)
        agent_ok = bool(health and health.get("agent", {}).get("ok"))
        st.caption(("🟢" if health else "🔴") + " backend  ·  " + ("🟢" if agent_ok else "🟠") +
                   (" agent (Qwen 3.5 4B)" if agent_ok else " agent offline - rule-based fallbacks"))

        projects = api("GET", "/projects", quiet=True) or []
        ids = [x["id"] for x in projects]
        c1, c2 = st.columns([3, 2])
        with c1:
            if ids:
                label = {x["id"]: f"{x['id']} · {x['data'] or 'no data'}" for x in projects}
                sel = st.selectbox("Project", ids, index=ids.index(ss.pid) if ss.pid in ids else 0,
                                   format_func=label.get, label_visibility="collapsed")
                if sel != ss.pid:
                    ss.pid = sel
                    ss.step = 1
                    st.rerun()
        with c2:
            if st.button("＋ New", width="stretch"):
                new = api("POST", "/projects")
                if new:
                    ss.pid, ss.step = new["id"], 1
                    st.rerun()

        if p:
            st.divider()
            for i, name in enumerate(STEPS, start=1):
                locked = i > p["unlocked"]
                done = i < p["unlocked"]
                icon = "🔒" if locked else ("✅" if done else "▶️") if i != ss.step else "👉"
                if st.button(f"{icon}  {i}. {name}", key=f"nav{i}", disabled=locked, width="stretch"):
                    goto(i)
            chat_panel(p)


def chat_panel(p: dict) -> None:
    st.divider()
    with st.expander("💬 Ask the agent", expanded=False):
        st.caption("History lives in this browser session only.")
        for m in ss.chat[-12:]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        q = st.chat_input("Ask about this project...")
        if q:
            ss.chat.append({"role": "user", "content": q})
            ctx = api("GET", f"/projects/{p['id']}/digest", quiet=True) or ""
            try:
                with st.spinner("thinking..."):
                    r = httpx.post(f"{AGENT}/v1/chat", json={"messages": ss.chat, "context": ctx},
                                   timeout=LLM_TIMEOUT)
                r.raise_for_status()
                ss.chat = r.json()["messages"]
            except httpx.HTTPError as exc:
                ss.chat.append({"role": "assistant", "content": f"_agent unavailable: {exc}_"})
            st.rerun()
        c1, c2 = st.columns(2)
        c1.download_button("Save chat", json.dumps(ss.chat, indent=2), "loopback-chat.json",
                           "application/json", width="stretch")
        if c2.button("Clear", width="stretch"):
            ss.chat = []
            st.rerun()
        up = st.file_uploader("Restore chat", type="json", label_visibility="collapsed")
        if up is not None:
            try:
                ss.chat = [m for m in json.load(up) if m.get("role") in ("user", "assistant")]
            except (ValueError, AttributeError):
                st.warning("not a chat file")


# ================================================================== layout helpers

def header(p: dict) -> tuple[Any, Any]:
    i = ss.step
    st.progress(i / len(STEPS), text=f"Step {i} of {len(STEPS)} · **{STEPS[i - 1]}**")
    left, right = st.columns([3, 2], gap="large")
    return left, right


def agent_card(title: str, body_md: str) -> None:
    st.markdown(f'<div class="agent-card"><b>{title}</b><br>', unsafe_allow_html=True)
    st.markdown(body_md)
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================== 1. Data

def step_data(p: dict) -> None:
    left, right = header(p)
    with left:
        st.subheader("Data")
        tab_up, tab_open = st.tabs(["Upload", "Open datasets"])
        with tab_up:
            files = st.file_uploader("CSV / Parquet, or a .zip of a YOLO or COCO folder",
                                     type=["csv", "parquet", "zip"], accept_multiple_files=True)
            if files and st.button("Use uploaded files", type="primary"):
                payload = [("files", (f.name, f.getvalue())) for f in files]
                if api("POST", f"/projects/{p['id']}/upload", files=payload, timeout=600):
                    api("POST", f"/projects/{p['id']}/scan", json={}, timeout=600)
                    st.rerun()
        with tab_open:
            ds = api("GET", "/datasets", quiet=True) or []
            if not ds:
                st.info("No datasets found. Run `python scripts/fetch_datasets.py`.")
            else:
                names = [d["name"] for d in ds]
                kinds = {d["name"]: d["kind"] for d in ds}
                name = st.selectbox("Dataset", names, format_func=lambda n: f"{n}  ({kinds[n]})")
                if st.button("Use this dataset", type="primary"):
                    if api("POST", f"/projects/{p['id']}/dataset", json={"name": name}):
                        api("POST", f"/projects/{p['id']}/scan", json={}, timeout=600)
                        st.rerun()

        card = p.get("card")
        if not card:
            return
        st.divider()
        kind = card["kind"]
        st.markdown(f"**Detected:** `{kind.upper()}`  ·  `{p['data_path']}`")
        if kind == "tabular":
            c = st.columns(4)
            c[0].metric("Rows", card["n_rows"])
            c[1].metric("Columns", card["n_cols"])
            c[2].metric("Duplicates", card["duplicate_rows"])
            c[3].metric("Target", card.get("target") or "-")
            df = pd.DataFrame(card["columns"])[["name", "kind", "n_unique", "null_pct", "flags", "sample"]]
            df["flags"] = df["flags"].map(", ".join)
            df["sample"] = df["sample"].map(lambda xs: ", ".join(map(str, xs)))
            st.dataframe(df, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(df)))
            ts = card.get("target_stats") or {}
            if ts.get("kind") == "categorical":
                st.caption("Target balance")
                st.bar_chart(pd.Series(ts["counts"], name="rows"), height=180)
        else:
            splits = pd.DataFrame(card["splits"]).T
            c = st.columns(4)
            c[0].metric("Images", int(splits["images"].sum()))
            c[1].metric("Boxes", int(splits["boxes"].sum()))
            c[2].metric("Classes", len(card["classes"]))
            s = card.get("image_size") or {}
            c[3].metric("Median size", f"{s.get('median_w', '-')}x{s.get('median_h', '-')}")
            st.dataframe(splits, width="stretch")
            cls = pd.DataFrame(card["classes"]).set_index("name")["count"]
            st.caption("Class balance (boxes)")
            st.bar_chart(cls[cls > 0].sort_values(ascending=False).head(40), height=200)
            st.caption(f"Box sizes: {card['box_sizes']}  ·  split overlap: {card['overlap']}")
        for w in card.get("warnings", []):
            st.warning(w, icon="⚠️")

    with right:
        if not card:
            agent_card("Agent", "Add data on the left. I will scan schema, missing values, class balance, "
                                "image sizes and split leakage.")
            return
        d = p["defaults"]
        agent_card("Detected task", f"`{d['task']}` scored by `{d['target_metric']}`. Confirm or change it.")
        with st.form("confirm"):
            tasks = ["classify", "regress"] if kind == "tabular" else ["detect"]
            task = st.selectbox("Task", tasks, index=tasks.index(d["task"]) if d["task"] in tasks else 0)
            target = None
            if kind == "tabular":
                cands = d["target_candidates"] or [c["name"] for c in card["columns"]]
                cur = card.get("target")
                target = st.selectbox("Target column", cands, index=cands.index(cur) if cur in cands else 0)
            metrics = {"classify": ["f1_macro", "accuracy"], "regress": ["rmse", "mae", "r2"],
                       "detect": ["mAP50", "mAP50-95"]}[task]
            metric = st.selectbox("Target metric", metrics,
                                  index=metrics.index(d["target_metric"]) if d["target_metric"] in metrics else 0)
            tv = st.number_input("Target value (stop when reached)", value=float(d["target_value"] or 0.0),
                                 format="%.4f")
            if st.form_submit_button("✅ Approve & continue", type="primary", width="stretch"):
                if api("POST", f"/projects/{p['id']}/confirm",
                       json={"task": task, "target": target, "target_metric": metric,
                             "target_value": tv or None}, timeout=120):
                    goto(2)


# ================================================================== 2. Context

def step_context(p: dict) -> None:
    left, right = header(p)
    ctx = p["context"]
    card = p["card"]
    with left:
        st.subheader("Context")
        st.caption("What the agent sees about your data")
        st.code(p["card_digest"], language="text")
        cons = ctx.get("constraints") or {}
        with st.form("ctx"):
            notes = st.text_area("Domain notes", value=ctx.get("notes", ""), height=110,
                                 placeholder="e.g. 'false negatives are expensive', 'do not flip class X'")
            c1, c2 = st.columns(2)
            drop = []
            if card["kind"] == "tabular":
                cols = [c["name"] for c in card["columns"] if c["name"] != ctx.get("target")]
                drop = c1.multiselect("Never use these columns", cols, default=cons.get("drop_columns", []))
                no_flip = False
            else:
                no_flip = c1.checkbox("No horizontal flip (orientation matters)", value=cons.get("no_flip", False))
            max_ep = c2.number_input("Max epochs (0 = no cap)", min_value=0, value=int(cons.get("max_epochs") or 0))
            device = c2.selectbox("Device", ["auto", "cpu", "cuda"],
                                  index=["auto", "cpu", "cuda"].index(cons.get("device", "auto")))
            if st.form_submit_button("🤖 Ask agent to read the context", width="stretch"):
                if agent_call("Reading context", "POST", f"/projects/{p['id']}/context",
                              json={"notes": notes, "constraints": {"drop_columns": drop, "no_flip": no_flip,
                                                                    "max_epochs": max_ep or None,
                                                                    "device": device}}):
                    st.rerun()
    with right:
        if not ctx.get("source"):
            agent_card("Agent", "Add notes and constraints, then ask me to read the data in context.")
            return
        source_badge(ctx["source"], ctx.get("agent_note"))
        agent_card("My reading", ctx["summary"])
        if ctx.get("risks"):
            agent_card("Risks", "\n".join(f"- {r}" for r in ctx["risks"]))
        if ctx.get("questions"):
            agent_card("Questions for you", "\n".join(f"- {q}" for q in ctx["questions"]))
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve", type="primary", width="stretch"):
            if api("POST", f"/projects/{p['id']}/context/approve"):
                goto(3)
        if c2.button("↩️ Reject (edit notes & re-ask)", width="stretch"):
            st.info("Edit the notes / constraints on the left and ask again.")


# ================================================================== 3. Suggest

def step_suggest(p: dict) -> None:
    left, right = header(p)
    sug = p.get("suggestions")
    with left:
        st.subheader("Suggest")
        st.caption("Fixed model zoo - the agent can only pick from these")
        if sug:
            st.code("\n".join(sug["all_candidates"]), language="text")
        feedback = st.text_input("Feedback for the agent (optional)", placeholder="e.g. 'prefer a neural net'")
        if st.button("🤖 Suggest models" if not sug else "🔄 Reject all & re-suggest", width="stretch"):
            if agent_call("Choosing models", "POST", f"/projects/{p['id']}/suggest", json={"feedback": feedback}):
                st.rerun()
    with right:
        if not sug:
            agent_card("Agent", "I will propose 1-3 job specs from the zoo, each with a one-line reason.")
            return
        source_badge(sug["source"], sug.get("note"))
        for i, s in enumerate(sug["suggestions"]):
            with st.container(border=True):
                st.markdown(f"**{i + 1}. {s['spec']['model']}**" +
                            (f"  ·  _already tried in {s['already_tried']}_" if s.get("already_tried") else ""))
                st.markdown(s["reason"])
                edited = st.text_area("job_spec.yaml", s["yaml"], height=220, key=f"spec{i}")
                if st.button(f"✅ Approve {s['spec']['model']}", key=f"ap{i}", type="primary",
                             width="stretch"):
                    if agent_call("Planning adaptation", "POST", f"/projects/{p['id']}/suggest/approve",
                                  json={"index": i, "spec_yaml": edited}):
                        goto(4)


# ================================================================== 4. Adapt

def step_adapt(p: dict) -> None:
    left, right = header(p)
    plan = p.get("adapt_plan")
    review = p.get("adapt_review") or {}
    diff = p.get("adapt_diff")
    if not plan:
        with right:
            if st.button("🤖 Plan adaptation", type="primary"):
                if agent_call("Planning", "POST", f"/projects/{p['id']}/adapt/plan"):
                    st.rerun()
        return
    with left:
        st.subheader("Adapt input / output")
        new_plan = dict(plan)
        if plan["kind"] == "tabular":
            card = p["card"]
            cols = [c["name"] for c in card["columns"] if c["name"] != plan["target"]]
            reasons = {d["column"]: d["reason"] for d in plan["drop"]}
            extra = [c for c in review.get("extra_drop", []) if c not in reasons]
            default_drop = list(reasons) + extra
            drop = st.multiselect("Drop columns", cols, default=default_drop,
                                  help="Pre-filled by rules" + (" + agent" if extra else ""))
            for c in drop:
                st.caption(f"• {c}: {reasons.get(c, 'agent: leakage / ID' if c in extra else 'user choice')}")
            keep = [c for c in cols if c not in drop]
            by_kind = {c["name"]: c for c in card["columns"]}
            numeric = [c for c in keep if by_kind[c]["kind"] in ("numeric", "bool")]
            cat = [c for c in keep if c not in numeric]
            onehot = st.multiselect("One-hot encode", cat, default=[c for c in cat if c in plan["onehot"]])
            ordinal = [c for c in cat if c not in onehot]
            st.caption(f"Numeric (median impute + scale): {', '.join(numeric) or '-'}")
            st.caption(f"Ordinal (mode impute + scale): {', '.join(ordinal) or '-'}")
            c1, c2, c3 = st.columns(3)
            cw = c1.selectbox("Class weight", [None, "balanced"],
                              index=0 if plan.get("class_weight") is None else 1, disabled=plan["task"] != "classify")
            val = c2.slider("Val size", 0.05, 0.3, float(plan["val_size"]), 0.05)
            test = c3.slider("Test size", 0.05, 0.3, float(plan["test_size"]), 0.05)
            new_plan.update(drop=[{"column": c, "reason": reasons.get(c, "agent/user")} for c in drop],
                            numeric=numeric, onehot=onehot, ordinal=ordinal, class_weight=cw,
                            val_size=val, test_size=test)
        else:
            classes = p["card"]["classes"]
            st.caption(f"Source format: **{plan['source_format'].upper()}**"
                       + ("  →  converting to YOLO" if plan.get("convert") else ""))
            table = pd.DataFrame([{"class": c["name"], "boxes": c["count"],
                                   "rename to": plan["rename"].get(c["name"], c["name"]),
                                   "keep": c["name"] not in plan["drop_classes"]} for c in classes])
            edited = st.data_editor(table, hide_index=True, width="stretch",
                                    disabled=["class", "boxes"], height=min(360, 38 + 35 * len(table)))
            rename = {r["class"]: r["rename to"] for _, r in edited.iterrows()
                      if r["rename to"] and r["rename to"] != r["class"]}
            dropc = [r["class"] for _, r in edited.iterrows() if not r["keep"]]
            final = []
            for _, r in edited.iterrows():
                if r["keep"]:
                    n = r["rename to"] or r["class"]
                    if n not in final:
                        final.append(n)
            c1, c2, c3 = st.columns(3)
            sizes = [320, 416, 512, 640, 768, 960, 1280]
            imgsz = c1.selectbox("imgsz", sizes, index=sizes.index(plan["imgsz"]) if plan["imgsz"] in sizes else 3)
            fl = c2.slider("fliplr", 0.0, 0.5, float(plan["fliplr"]), 0.5)
            vf = c3.number_input("Carve val from train", 0.0, 0.5, float(plan.get("val_fraction") or 0.0), 0.05,
                                 help="0 = keep the existing val split")
            dd = st.checkbox("Remove val images identical to train images", value=plan["dedupe_val"])
            new_plan.update(rename=rename, drop_classes=dropc, class_map=final, imgsz=imgsz, fliplr=fl,
                            val_fraction=vf or None, dedupe_val=dd)
        if diff:
            st.divider()
            st.markdown("**Applied - diff**")
            st.json({k: v for k, v in diff.items() if k != "path"}, expanded=True)
    with right:
        source_badge(review.get("source"), review.get("note"))
        agent_card("Agent review", review.get("notes", ""))
        if review.get("extra_drop"):
            agent_card("Also drop", ", ".join(f"`{c}`" for c in review["extra_drop"]) + " (pre-selected on the left)")
        for w in review.get("warnings", []):
            st.warning(w, icon="⚠️")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve & apply", type="primary", width="stretch"):
            with st.spinner("Applying..."):
                if api("POST", f"/projects/{p['id']}/adapt/apply", json={"plan": new_plan}, timeout=900):
                    st.rerun()
        if c2.button("🔄 Reject & re-plan", width="stretch"):
            if agent_call("Re-planning", "POST", f"/projects/{p['id']}/adapt/plan"):
                st.rerun()
        if diff and st.button("Continue to Train →", width="stretch"):
            goto(5)


# ================================================================== 5. Train

def _chart(rows: list[dict], metric: str) -> None:
    if not rows:
        st.info("Waiting for the first epoch...")
        return
    df = pd.DataFrame(rows).set_index("epoch")
    num = lambda cols: [c for c in cols if c in df and pd.api.types.is_numeric_dtype(df[c])]  # noqa: E731
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Loss")
        st.line_chart(df[num(["train_loss", "val_loss"])], height=220)
    with c2:
        st.caption(f"val {metric}")
        st.line_chart(df[num(["val_metric"])], height=220)


def step_train(p: dict) -> None:
    left, right = header(p)
    rid = p.get("current_run")
    if not rid:
        with left:
            st.subheader("Train")
            spec_yaml = st.text_area("job_spec.yaml (final)", yaml.safe_dump(p["spec"], sort_keys=False), height=380)
        with right:
            agent_card("Ready", "Training runs in its own process. Logs are written every epoch to "
                                "`runs/<run_id>/train.log` and read by the agent.")
            if st.button("▶️ Start training", type="primary", width="stretch"):
                if api("POST", f"/projects/{p['id']}/train", json={"spec_yaml": spec_yaml}, timeout=60):
                    st.rerun()
        return
    live_train(rid, p)


@st.fragment(run_every=3)
def live_train(rid: str, p: dict) -> None:
    v = api("GET", f"/runs/{rid}", quiet=True)
    if not v:
        return
    stt = v["status"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader(f"Train · `{rid}`")
        _chart(v["rows"], v["spec"]["target_metric"])
        st.caption("Raw log tail (train.log)")
        st.code("\n".join(json.dumps(r) for r in v["rows"][-12:]) or "-", language="json")
    with right:
        state = stt.get("state")
        icon = {"running": "🟢", "paused": "⏸️", "completed": "✅", "early_stopped": "🛑", "stopped": "⏹️",
                "failed": "❌", "queued": "⏳"}.get(state, "•")
        agent_card(f"{icon} {state}", f"epoch {stt.get('epoch', '-')} / {v['spec']['train']['epochs']}"
                   + (f"\n\n**Reason:** {stt['reason']}" if stt.get("reason") else ""))
        c1, c2, c3 = st.columns(3)
        alive = stt.get("alive")
        if c1.button("⏸ Pause", disabled=not alive or state == "paused", width="stretch"):
            api("POST", f"/runs/{rid}/control", json={"action": "pause"})
        if c2.button("▶ Resume", disabled=state != "paused", width="stretch"):
            api("POST", f"/runs/{rid}/control", json={"action": "resume"})
        if c3.button("⏹ Stop", disabled=not alive, width="stretch"):
            api("POST", f"/runs/{rid}/control", json={"action": "stop"})
        vd = v.get("verdict")
        if vd and vd.get("status") == "pending":
            st.warning(f"Agent verdict waiting: **{vd['verdict']}** - {vd['evidence']}", icon="🤖")
            if st.button("Open Watch →", width="stretch"):
                ss.step = 6
                st.rerun(scope="app")
        if v.get("error"):
            st.error("Worker error")
            st.code(v["error"][-1500:])
        if not alive and state in ("completed", "early_stopped", "stopped"):
            if st.button("Continue to Eval →", type="primary", width="stretch"):
                ss.step = 7
                st.rerun(scope="app")
        if state == "failed" and st.button("Diagnose →", width="stretch"):
            ss.step = 7
            st.rerun(scope="app")
        st.caption("Other runs: " + ", ".join(p["run_ids"]))


# ================================================================== 6. Watch

def step_watch(p: dict) -> None:
    left, right = header(p)
    rid = p.get("current_run")
    v = api("GET", f"/runs/{rid}")
    if not v:
        return
    vd = v.get("verdict")
    with left:
        st.subheader(f"Watch · `{rid}`")
        _chart(v["rows"], v["spec"]["target_metric"])
        if vd:
            st.caption("Signals measured from the log")
            for s in vd.get("signals", []):
                st.markdown(f"- **{s['kind']}** @ epoch {s['epoch']}: {s['evidence']} → _{s['severity']}_")
        st.caption("Event trail (events.jsonl)")
        ev = [f"{e.get('ts', '')[11:19]}  {e.get('event')}  {e.get('detail') or ''}" for e in v["events"][-15:]]
        st.code("\n".join(ev) or "-", language="text")
    with right:
        if st.button("🤖 Ask agent to review the log now", width="stretch"):
            if agent_call("Reviewing log", "POST", f"/runs/{rid}/review"):
                st.rerun()
        if not vd:
            agent_card("Agent", "I review the log tail every few epochs automatically, or when you ask.")
            return
        source_badge(vd.get("source"), vd.get("note"))
        icon = {"continue": "🟢", "change": "🟠", "stop": "🔴"}[vd["verdict"]]
        body = f"**Evidence (epoch {vd['evidence_epoch']}):** {vd['evidence']}\n\n{vd['reason']}"
        if vd.get("change"):
            body += f"\n\n**Proposed change:** `{vd['change']['param']}` → `{vd['change']['value']}` (starts a new run)"
        agent_card(f"{icon} {vd['verdict'].upper()}  ·  at epoch {vd.get('at_epoch')}", body)
        if vd.get("status") == "pending":
            c1, c2 = st.columns(2)
            if c1.button("✅ Approve", type="primary", width="stretch"):
                r = api("POST", f"/runs/{rid}/verdict", json={"approve": True}, timeout=60)
                if r:
                    goto(5)
            if c2.button("❌ Reject", width="stretch"):
                if api("POST", f"/runs/{rid}/verdict", json={"approve": False}):
                    st.rerun()
        else:
            st.caption(f"Verdict status: {vd.get('status')}")


# ================================================================== 7. Eval

def step_eval(p: dict) -> None:
    left, right = header(p)
    rid = p.get("current_run")
    v = api("GET", f"/runs/{rid}")
    if not v:
        return
    ev = v.get("eval")
    with left:
        st.subheader(f"Eval · `{rid}`")
        state = v["status"]["state"]
        if state == "failed":
            st.error(f"Run failed: {v['status'].get('reason')}")
        elif state in ("running", "paused", "queued"):
            st.info("This run is still training - evaluate it once it finishes.")
            return
        elif not ev:
            if st.button("▶️ Evaluate on hold-out split", type="primary"):
                with st.spinner("Evaluating..."):
                    if api("POST", f"/runs/{rid}/evaluate", timeout=900):
                        st.rerun()
        else:
            m = ev["metrics"]
            cols = st.columns(len(m))
            for c, (k, val) in zip(cols, m.items()):
                c.metric(k, f"{val:.4f}" if isinstance(val, float) else val)
            st.caption(f"Split: {ev['split']} (n={ev['n']})" + (f" · {ev['note']}" if ev.get("note") else ""))
            if ev.get("per_class"):
                st.dataframe(pd.DataFrame(ev["per_class"]).T, width="stretch")
            for img in ev.get("images", []):
                run_image(rid, img)
            if ev.get("errors"):
                st.caption(f"{ev['errors']['legend']} · conf={ev['errors'].get('conf')}")
                st.caption(f"Missed by class: {ev['errors']['false_negatives']} · "
                           f"false positives: {ev['errors']['false_positives']}")
    with right:
        if ev:
            hit = ev.get("target_hit")
            agent_card("Target", f"`{ev['target_metric']}` = **{ev['value']}** vs target {ev['target_value']} → "
                       + ("✅ met" if hit else "❌ not met" if hit is False else "no target set"))
        if not ev and state != "failed":
            return
        if st.button("🤖 What next?", width="stretch"):
            ss.next = agent_call("Planning next step", "POST", f"/runs/{rid}/next")
        nx = ss.get("next")
        if nx:
            source_badge(nx.get("source"), nx.get("note"))
            body = f"**Evidence:** {nx['evidence']}\n\n{nx['reason']}"
            if nx.get("diagnosis"):
                d = nx["diagnosis"]
                body = f"**Failure mode:** `{d['failure_mode']}`\n\n**Evidence:** {d['evidence']}\n\n**Fix:** {d['fix']}"
            if nx.get("change"):
                body += f"\n\n**Change:** `{nx['change']['param']}` → `{nx['change']['value']}`"
            agent_card(f"{'🔁 RETRAIN' if nx['action'] == 'retrain' else '🏁 FINISH'}", body)
            c1, c2 = st.columns(2)
            if nx["action"] == "retrain" and nx.get("change"):
                if c1.button("✅ Approve new run", type="primary", width="stretch"):
                    if api("POST", f"/runs/{rid}/next/start", json=nx["change"]):
                        ss.next = None
                        goto(5)
            if c2.button("🏁 Finish → Report", width="stretch", disabled=not ev):
                ss.next = None
                goto(8)


# ================================================================== 8. Report

def step_report(p: dict) -> None:
    left, right = header(p)
    rid = p.get("current_run")
    with right:
        agent_card("Report", "Numbers are read from the run files; the agent writes the narrative.")
        if st.button("🤖 Generate report", type="primary", width="stretch"):
            md = agent_call("Writing report", "POST", f"/runs/{rid}/report")
            if md:
                ss[f"report_{rid}"] = md
        md = ss.get(f"report_{rid}")
        if md:
            st.download_button("⬇️ Download report.md", md, f"{rid}-report.md", "text/markdown",
                               width="stretch")
        exps = api("GET", f"/projects/{p['id']}/experiments", quiet=True) or []
        if exps:
            st.caption("Experiments (memory)")
            st.dataframe(pd.DataFrame(exps)[["run_id", "model", "status", "best_metric", "final_metric", "change"]],
                         hide_index=True, width="stretch")
    with left:
        md = ss.get(f"report_{rid}")
        if not md:
            st.info("Generate the report on the right.")
            return
        # render markdown, swapping local image refs for images fetched from the backend
        for chunk in re.split(r"(!\[[^\]]*\]\([^)]+\))", md):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", chunk)
            if m:
                run_image(rid, m.group(2), m.group(1))
            elif chunk.strip():
                st.markdown(chunk)


# ================================================================== main

def main() -> None:
    p = project()
    sidebar(p)
    if not p:
        st.title("🔁 Loopback agent")
        st.markdown("Look at data → pick a model → adapt → train → watch → evaluate → report.  \n"
                    "Create a project in the sidebar to start.")
        return
    if ss.step > p["unlocked"]:
        ss.step = p["unlocked"]
    [step_data, step_context, step_suggest, step_adapt, step_train, step_watch, step_eval, step_report][ss.step - 1](p)


main()
