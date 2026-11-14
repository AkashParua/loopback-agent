"""Pages beyond the 8-step flow: Dashboard, Experiments, System; and the richer
charts used inside the flow (data explorer, live training curves)."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import AGENT, COLORS, api, kpi, plot, style

ss = st.session_state
KIND_ICON = {"tabular": "📋", "yolo": "🎯", "coco": "🧩"}
DATASET_BLURB = {
    "titanic": "Survival prediction · nulls, IDs, free text",
    "breast_cancer": "Tumour diagnosis · clean binary classification",
    "diabetes": "Disease progression · regression (parquet)",
    "wine": "Grape cultivar from chemistry · 3 classes",
    "california_housing": "Median house value · 20k rows regression",
    "coco8": "80 COCO classes · tiny smoke test",
    "medical-pills": "Pill detection on blister packs · 115 images",
    "medical-pills-coco": "Same pills, COCO format · tests COCO→YOLO",
    "african-wildlife": "Buffalo, elephant, rhino, zebra · 1.5k images",
}


# ================================================================== dashboard

def start_with_dataset(name: str) -> None:
    new = api("POST", "/projects")
    if not new:
        return
    with st.spinner(f"Scanning {name}..."):
        if api("POST", f"/projects/{new['id']}/dataset", json={"name": name}):
            api("POST", f"/projects/{new['id']}/scan", json={}, timeout=600)
    ss.pid, ss.step, ss.page = new["id"], 1, "Workflow"
    st.rerun()


def dashboard() -> None:
    sysinfo = api("GET", "/system", quiet=True, timeout=8) or {}
    eng = sysinfo.get("engine") or {}
    avg = eng.get("avg_recent") or {}
    st.markdown(
        '<div class="hero"><h1>🔁 Loopback</h1>'
        '<p>An ML engineer that runs on your laptop: it reads the data, picks a model, adapts the inputs, '
        'trains in PyTorch, watches the logs and writes the report. You approve every step.</p>'
        '<span class="chip">🧠 Qwen 3.5 4B · MLX on Apple Silicon</span>'
        '<span class="chip">🔒 100% local · no data leaves the machine</span>'
        '<span class="chip">✅ schema-constrained agent answers</span>'
        '<span class="chip">⚡ PyTorch on Metal (MPS)</span></div>', unsafe_allow_html=True)

    ov = api("GET", "/overview", quiet=True) or {}
    by = ov.get("by_status", {})
    c = st.columns(5)
    kpi(c[0], "Projects", ov.get("n_projects", 0))
    kpi(c[1], "Training runs", ov.get("n_runs", 0),
        f"{by.get('completed', 0) + by.get('early_stopped', 0)} finished · {by.get('running', 0)} running")
    kpi(c[2], "Agent speed", f"{avg['gen_tps']:.0f} tok/s" if avg.get("gen_tps") else "-",
        f"first token {avg['ttft_s']:.2f} s" if avg.get("ttft_s") else "warming up")
    mem = sysinfo.get("memory") or {}
    kpi(c[3], "Free memory", f"{mem.get('available_gb', '-')} GB", f"of {mem.get('total_gb', '-')} GB unified")
    kpi(c[4], "LLM resident", f"{(eng.get('mlx_memory_gb') or {}).get('active', '-')} GB",
        "model loaded" if eng.get("loaded") else "unloaded (loads on demand)")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Start from an open dataset")
        st.caption("One click: new project → dataset attached → schema scanned. Or drag your own file in on the Data step.")
        ds = api("GET", "/datasets", quiet=True) or []
        cols = st.columns(3)
        for i, d in enumerate(ds):
            with cols[i % 3]:
                st.markdown(f'<div class="ds-card">{KIND_ICON.get(d["kind"], "📦")} <b>{d["name"]}</b><br>'
                            f'<span class="muted">{d["kind"].upper()} · {DATASET_BLURB.get(d["name"], "")}</span>'
                            f'</div>', unsafe_allow_html=True)
                if st.button("Start →", key=f"ds_{d['name']}", width="stretch"):
                    start_with_dataset(d["name"])
    with right:
        st.subheader("Leaderboard")
        lb = ov.get("leaderboard") or []
        if not lb:
            st.info("No finished runs yet. Start a project on the left.")
        else:
            df = pd.DataFrame(lb)
            df["label"] = df["data"].fillna(df["project"]) + " · " + df["model"]
            fig = px.bar(df.sort_values("metric"), x="metric", y="label", orientation="h", color="target_metric",
                         text=df.sort_values("metric")["metric"].round(3), color_discrete_sequence=COLORS)
            fig.update_layout(yaxis_title=None, xaxis_title=None, hovermode="closest")
            plot(style(fig, height=80 + 46 * len(df), legend=True), key="lb")
        if by:
            fig = go.Figure(go.Pie(labels=list(by), values=list(by.values()), hole=.6,
                                   marker=dict(colors=COLORS)))
            plot(style(fig, height=230, title="Runs by outcome"), key="donut")

    exps = ov.get("experiments") or []
    if exps:
        st.subheader("Recent experiments")
        df = pd.DataFrame(exps)[["run_id", "project_id", "model", "status", "target_metric", "best_metric",
                                 "final_metric", "change", "created_at"]].iloc[::-1]
        st.dataframe(df, hide_index=True, width="stretch", height=min(380, 38 + 35 * len(df)),
                     column_config={"best_metric": st.column_config.ProgressColumn(
                         "best", format="%.4f", min_value=0.0,
                         max_value=float(max(1.0, df["best_metric"].fillna(0).max())))})


# ================================================================== data explorer

def tabular_explorer(pid: str, card: dict) -> None:
    prof = api("GET", f"/projects/{pid}/profile", quiet=True, timeout=120)
    if not prof:
        return
    t1, t2, t3, t4, t5 = st.tabs(["📊 Distributions", "🔥 Correlations", "🎯 Target vs features",
                                  "🕳️ Missing values", "🔎 Rows"])
    with t1:
        hists = prof["histograms"]
        names = list(hists)
        for i in range(0, len(names), 3):
            cols = st.columns(3)
            for col, name in zip(cols, names[i:i + 3]):
                h = hists[name]
                mids = [(a + b) / 2 for a, b in zip(h["edges"][:-1], h["edges"][1:])]
                fig = go.Figure(go.Bar(x=mids, y=h["counts"], marker_color=COLORS[i // 3 % len(COLORS)]))
                fig.update_layout(bargap=0.05)
                with col:
                    plot(style(fig, height=190, title=name, legend=False), key=f"h_{name}")
        ts = card.get("target_stats") or {}
        if ts.get("kind") == "categorical":
            fig = px.pie(names=list(ts["counts"]), values=list(ts["counts"].values()), hole=.55,
                         color_discrete_sequence=COLORS)
            plot(style(fig, height=260, title=f"Target balance · {card.get('target')}"), key="tbal")
    with t2:
        cm = prof["correlation"]
        if cm["columns"]:
            fig = go.Figure(go.Heatmap(z=cm["values"], x=cm["columns"], y=cm["columns"], zmin=-1, zmax=1,
                                       colorscale="RdBu", reversescale=True,
                                       text=[[f"{v:.2f}" for v in r] for r in cm["values"]],
                                       texttemplate="%{text}", textfont={"size": 10}))
            fig.update_layout(hovermode="closest")
            plot(style(fig, height=140 + 28 * len(cm["columns"]), legend=False), key="corr")
            st.caption("Pearson correlation on numeric columns; the target is label-encoded when categorical.")
        else:
            st.info("Not enough numeric columns for a correlation matrix.")
    with t3:
        if prof["scatter"]:
            df = pd.DataFrame(prof["scatter"])
            tgt = prof["target"]
            cat = (card.get("target_stats") or {}).get("kind") == "categorical"
            if cat:
                df[tgt] = df[tgt].astype(str)
            fig = px.scatter_matrix(df, dimensions=prof["top_features"], color=tgt,
                                    color_discrete_sequence=["#4f46e5", "#f59e0b", "#10b981", "#ef4444"], color_continuous_scale="Viridis",
                                    opacity=.6)
            fig.update_traces(diagonal_visible=False, marker=dict(size=4))
            fig.update_layout(hovermode="closest")
            plot(style(fig, height=560), key="smat")
            st.caption(f"The {len(prof['top_features'])} numeric features most correlated with `{tgt}`.")
        else:
            st.info("Pick a target column to see feature/target relationships.")
    with t4:
        miss = prof["missing_pct"]
        if miss:
            s = pd.Series(miss).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                                   marker_color=["#ef4444" if v > 60 else "#f59e0b" if v > 10 else "#10b981"
                                                 for v in s.values],
                                   text=[f"{v:.1f}%" for v in s.values], textposition="outside"))
            fig.update_layout(xaxis_title="% missing", xaxis_range=[0, 105], hovermode="closest")
            plot(style(fig, height=80 + 32 * len(s), legend=False), key="miss")
        else:
            st.success("No missing values.")
    with t5:
        st.dataframe(pd.DataFrame(prof["head"]), width="stretch", hide_index=True)
        st.caption(f"First 25 of {card['n_rows']} rows")


def detect_explorer(pid: str, card: dict) -> None:
    prof = api("GET", f"/projects/{pid}/profile", quiet=True, timeout=300)
    if not prof:
        return
    t1, t2, t3 = st.tabs(["🖼️ Sample gallery", "📐 Box geometry", "🏷️ Classes"])
    with t1:
        c1, c2, c3 = st.columns([2, 2, 1])
        split = c1.selectbox("Split", ["all", *prof["splits"]], key="gal_split")
        n = c2.select_slider("Images", [4, 8, 12, 16], value=8, key="gal_n")
        if c3.button("🔀 Shuffle", width="stretch"):
            ss.gal_seed = ss.get("gal_seed", 0) + 1
        params = {"n": n, "seed": ss.get("gal_seed", 0)}
        if split != "all":
            params["split"] = split
        png = api("GET", f"/projects/{pid}/samples.png", params=params, quiet=True, timeout=120)
        if isinstance(png, (bytes, bytearray)):
            st.image(png, width="stretch", caption="Ground-truth boxes, one color per class")
    with t2:
        wh = pd.DataFrame(prof["box_wh"])
        cols = st.columns(2)
        if not wh.empty:
            fig = px.scatter(wh, x="w", y="h", color="class", opacity=.55, color_discrete_sequence=COLORS)
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(xaxis_title="box width (fraction of image)", yaxis_title="box height",
                              hovermode="closest")
            with cols[0]:
                plot(style(fig, height=340, title="Box sizes"), key="bwh")
        bpi = prof["boxes_per_image"]
        fig = go.Figure(go.Bar(x=[int(k) for k in bpi], y=list(bpi.values()), marker_color=COLORS[1]))
        fig.update_layout(xaxis_title="boxes in image", yaxis_title="images", hovermode="closest")
        with cols[1]:
            plot(style(fig, height=340, title="Boxes per image", legend=False), key="bpi")
    with t3:
        cls = pd.DataFrame(card["classes"])
        cls = cls[cls["count"] > 0].sort_values("count", ascending=True).tail(40)
        fig = go.Figure(go.Bar(x=cls["count"], y=cls["name"], orientation="h", marker_color=COLORS[0],
                               text=cls["count"], textposition="outside"))
        fig.update_layout(hovermode="closest")
        plot(style(fig, height=80 + 26 * len(cls), title="Boxes per class", legend=False), key="cls")
        bs = card.get("box_sizes") or {}
        if bs:
            fig = px.pie(names=list(bs), values=list(bs.values()), hole=.55, color_discrete_sequence=COLORS)
            plot(style(fig, height=240, title="Small / medium / large (COCO area buckets)"), key="bsz")


# ================================================================== training charts

def train_charts(rows: list[dict], metric: str, epochs: Optional[int] = None, key: str = "tc") -> None:
    if not rows:
        st.info("Waiting for the first epoch...")
        return
    df = pd.DataFrame(rows)
    num = lambda cols: [c for c in cols if c in df and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]  # noqa: E731
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for i, c in enumerate(num(["train_loss", "val_loss"])):
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[c], name=c, mode="lines+markers",
                                 line=dict(width=2.5, color=COLORS[i]), marker=dict(size=5)))
    with c1:
        plot(style(fig, height=280, title="Loss"), key=f"{key}_loss")
    fig = go.Figure()
    if "val_metric" in num(["val_metric"]):
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_metric"], name=f"val {metric}", mode="lines+markers",
                                 fill="tozeroy", line=dict(width=2.5, color=COLORS[2]), marker=dict(size=5)))
        best = df["val_metric"].idxmax() if metric not in ("rmse", "mae") else df["val_metric"].idxmin()
        fig.add_trace(go.Scatter(x=[df.loc[best, "epoch"]], y=[df.loc[best, "val_metric"]], name="best",
                                 mode="markers", marker=dict(size=13, symbol="star", color="#f59e0b")))
    for i, c in enumerate(num(["precision", "recall", "mAP50-95"])):
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[c], name=c, mode="lines",
                                 line=dict(width=1.5, dash="dot", color=COLORS[4 + i])))
    with c2:
        plot(style(fig, height=280, title=f"Validation · {metric}"), key=f"{key}_metric")
    extra = num(["lr", "grad_norm", "box_loss", "cls_loss"])
    if extra:
        cols = st.columns(len(extra))
        for i, (col, c) in enumerate(zip(cols, extra)):
            fig = go.Figure(go.Scatter(x=df["epoch"], y=df[c], mode="lines", line=dict(color=COLORS[(i + 5) % 10])))
            with col:
                plot(style(fig, height=170, title=c, legend=False), key=f"{key}_{c}")


# ================================================================== experiments

def experiments(pid: Optional[str]) -> None:
    st.title("📊 Experiments")
    if not pid:
        st.info("Select or create a project in the sidebar.")
        return
    curves = api("GET", f"/projects/{pid}/curves", quiet=True) or {}
    exps = api("GET", f"/projects/{pid}/experiments", quiet=True) or []
    if not exps:
        st.info("No runs in this project yet - train one in the Workflow.")
        return
    metric = exps[0]["target_metric"]
    lower = metric in ("rmse", "mae")
    df = pd.DataFrame(exps)
    df["score"] = df["best_metric"].fillna(df["final_metric"])
    best = df.loc[df["score"].idxmin() if lower else df["score"].idxmax()] if df["score"].notna().any() else None
    c = st.columns(4)
    kpi(c[0], "Runs", len(df))
    kpi(c[1], f"Best {metric}", f"{best['score']:.4f}" if best is not None else "-",
        best["run_id"][-15:] if best is not None else "")
    kpi(c[2], "Models tried", df["model"].nunique(), ", ".join(df["model"].unique()))
    kpi(c[3], "Agent changes", int(df["change"].notna().sum()), "evidence-based retrains")
    st.write("")

    fig = go.Figure()
    fig2 = go.Figure()
    for i, (rid, v) in enumerate(curves.items()):
        r = pd.DataFrame(v["rows"])
        if r.empty:
            continue
        name = f"{rid.split('-')[1]} · {v['model']}"
        if "val_metric" in r:
            fig.add_trace(go.Scatter(x=r["epoch"], y=r["val_metric"], name=name, mode="lines",
                                     line=dict(width=2.5, color=COLORS[i % 10])))
        if "train_loss" in r:
            fig2.add_trace(go.Scatter(x=r["epoch"], y=r["train_loss"], name=name, mode="lines",
                                      line=dict(width=2, color=COLORS[i % 10])))
    c1, c2 = st.columns(2)
    with c1:
        plot(style(fig, height=330, title=f"Validation {metric} - all runs"), key="ex_val")
    with c2:
        plot(style(fig2, height=330, title="Train loss - all runs"), key="ex_loss")

    sc = df.dropna(subset=["score"])
    if not sc.empty:
        sc = sc.assign(run=sc["run_id"].str.split("-").str[1] + " · " + sc["model"])
        fig = px.bar(sc, x="run", y="score", color="model", text=sc["score"].round(4),
                     color_discrete_sequence=COLORS)
        if best is not None:
            fig.add_hline(y=best["score"], line_dash="dot", line_color="#10b981")
        plot(style(fig, height=300, title=f"Best {metric} per run"), key="ex_bar")

    st.subheader("Lineage")
    st.caption("Each retrain records its parent run and the single change the agent made, with evidence.")
    for _, r in df.iterrows():
        arrow = f"↳ from `{r['parent_run_id']}` · **{r['change']}**" if pd.notna(r["parent_run_id"]) else "root run"
        score = f"{r['score']:.4f}" if pd.notna(r["score"]) else "-"
        st.markdown(f"- `{r['run_id']}` · {r['model']} · {r['status']} · {metric} **{score}** - {arrow}"
                    + (f" · failure: `{r['failure_mode']}`" if pd.notna(r["failure_mode"]) else ""))


# ================================================================== system

def system_page() -> None:
    st.title("🖥️ System")
    st.caption("Local inference engine (MLX) and training host. Everything runs on this machine.")
    info = api("GET", "/system", quiet=True, timeout=10)
    if not info:
        st.error("Backend unreachable")
        return
    eng = info.get("engine") or {}
    mem = info["memory"]
    mlx = eng.get("mlx_memory_gb") or {}
    avg = eng.get("avg_recent") or {}
    c = st.columns(5)
    kpi(c[0], "Engine", (eng.get("engine") or "-").upper(), (eng.get("model") or "").split("/")[-1])
    kpi(c[1], "Model", "🟢 loaded" if eng.get("loaded") else "⚪ unloaded",
        f"load took {eng['load_s']} s" if eng.get("load_s") else "")
    kpi(c[2], "Generation", f"{avg['gen_tps']:.1f} tok/s" if avg.get("gen_tps") else "-", "avg of last 20 calls")
    kpi(c[3], "Time to first token", f"{avg['ttft_s']:.2f} s" if avg.get("ttft_s") else "-",
        f"agent call avg {avg['total_s']:.1f} s" if avg.get("total_s") else "")
    pc = eng.get("prefix_cache") or {}
    tot = (pc.get("hits") or 0) + (pc.get("misses") or 0)
    kpi(c[4], "Prompt cache", f"{100 * pc.get('hits', 0) / tot:.0f}% hits" if tot else "-",
        f"{pc.get('slots', 0)} system prompts cached")
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        hist = pd.DataFrame(eng.get("history") or [])
        if not hist.empty:
            hist["time"] = pd.to_datetime(hist["ts"], unit="s").dt.strftime("%H:%M:%S")
            n = list(range(1, len(hist) + 1))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=n, y=hist["gen_tps"], name="generation tok/s", customdata=hist["time"],
                                 hovertemplate="#%{x} %{customdata}<br>%{y} tok/s<extra></extra>",
                                 marker_color=[COLORS[0] if k == "json" else COLORS[1] for k in hist["kind"]]))
            fig.add_trace(go.Scatter(x=n, y=hist["total_s"], name="latency (s)", yaxis="y2",
                                     mode="lines+markers", line=dict(color=COLORS[3], width=2)))
            fig.update_layout(yaxis=dict(title="tok/s", rangemode="tozero"),
                              yaxis2=dict(title="seconds", overlaying="y", side="right", rangemode="tozero"),
                              xaxis=dict(title="request #"), hovermode="closest")
            plot(style(fig, height=320, title="Per-request throughput (indigo = schema task, sky = chat)"),
                 key="sys_tps")
            hh = hist.tail(12).iloc[::-1][["time", "kind", "prompt_tokens", "cached_tokens", "gen_tokens",
                                           "prompt_tps", "gen_tps", "ttft_s", "total_s", "peak_gb"]]
            st.dataframe(hh, hide_index=True, width="stretch")
        else:
            st.info("No requests yet - run a speed test.")
    with right:
        total = mem["total_gb"]
        llm = float(mlx.get("active") or 0)
        used_other = max(0.0, total - mem["available_gb"] - llm)
        fig = go.Figure()
        for name, val, color in [("LLM (MLX)", llm, COLORS[0]), ("macOS + apps + training", used_other, "#94a3b8"),
                                 ("available", mem["available_gb"], "#10b981")]:
            fig.add_trace(go.Bar(y=["unified memory"], x=[val], name=f"{name} {val:.1f} GB", orientation="h",
                                 marker_color=color))
        fig.update_layout(barmode="stack", xaxis=dict(range=[0, total], title="GB"), hovermode="closest")
        plot(style(fig, height=190, title=f"Unified memory · {total:.0f} GB"), key="sys_mem")
        lim = eng.get("limits_gb") or {}
        st.caption(f"MLX caps: wired {lim.get('wired')} GB · memory {lim.get('memory')} GB · buffer cache "
                   f"{lim.get('cache')} GB · refuses to load below {lim.get('min_free')} GB free · unloads after "
                   f"{int(eng.get('keep_alive_s') or 0)} s idle. Training workers cap PyTorch MPS at 60% "
                   f"of the GPU working set.")
        c1, c2, c3 = st.columns(3)
        if c1.button("⚡ Speed test", type="primary", width="stretch"):
            speed_test()
        if c2.button("⏏ Unload", width="stretch", disabled=not eng.get("loaded")):
            api("POST", "/system/engine/unload", timeout=120)
            st.rerun()
        if c3.button("⤴ Load", width="stretch", disabled=bool(eng.get("loaded"))):
            with st.spinner("Loading model into unified memory..."):
                api("POST", "/system/engine/load", timeout=180)
            st.rerun()
        t = info["torch"]
        st.markdown(f"**Training device:** {'🟢 Apple MPS (Metal)' if t['mps'] else '🟢 CUDA' if t['cuda'] else 'CPU'}"
                    f" · torch {t['version']}  \n**CPU:** {info['cpu_pct']}% of {info['cpu_count']} cores  \n"
                    f"**Active training runs:** {', '.join(info['active_runs']) or 'none'}  \n"
                    f"**Swap in use:** {mem['swap_used_gb']} GB")


def speed_test() -> None:
    import httpx

    prompt = ("Explain in about 120 words why a validation metric can plateau while the training loss keeps "
              "falling, and what an ML engineer should try next.")
    box = st.empty()
    t0 = time.perf_counter()
    first = None
    text = ""
    try:
        with httpx.stream("POST", f"{AGENT}/v1/chat/stream", timeout=300,
                          json={"messages": [{"role": "user", "content": prompt}]}) as r:
            for chunk in r.iter_text():
                if first is None and chunk.strip():
                    first = time.perf_counter() - t0
                text += chunk
                box.markdown(f"> {text}▌")
    except httpx.HTTPError as exc:
        st.error(f"agent unavailable: {exc}")
        return
    total = time.perf_counter() - t0
    box.markdown(f"> {text}")
    eng = (api("GET", "/system", quiet=True) or {}).get("engine") or {}
    last = (eng.get("history") or [{}])[-1]
    st.success(f"⚡ {last.get('gen_tps', '?')} tok/s generation · {last.get('gen_tokens', '?')} tokens · "
               f"first token in {first or 0:.2f} s · {total:.1f} s end to end")


def fmt_ts(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "-"
