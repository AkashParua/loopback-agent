"""Drive the full 8-step loop through the backend API, exactly as a user clicking
"approve" would. Leaves finished, evaluated, reported projects for a demo and
doubles as an end-to-end check of backend + agent + training.

    scripts/dev.sh &                      # stack must be running
    .venv/bin/python scripts/seed_demo.py                       # titanic + medical-pills
    .venv/bin/python scripts/seed_demo.py titanic:mlp wine california_housing medical-pills:yolov8n:20

Each argument is dataset[:model[:epochs]]; model defaults to the agent's first pick.
"""

from __future__ import annotations

import os
import sys
import time

import httpx
import yaml

API = os.getenv("BACKEND_URL", "http://127.0.0.1:8100").rstrip("/")
LOOPS = int(os.getenv("SEED_LOOPS", "2"))  # agent-proposed retrains to approve per project
NOTES = {
    "titanic": "Passenger survival. Do not use the passenger name or ticket number.",
    "medical-pills": "Pills on blister packs; orientation does not matter.",
    "african-wildlife": "Safari camera images; animals can appear at any scale.",
}


def call(method: str, path: str, timeout: float = 600, **kw):
    r = httpx.request(method, f"{API}{path}", timeout=timeout, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text


def step(msg: str, t0: float) -> None:
    print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)


def run(dataset: str, model: str | None = None, epochs: int | None = None) -> str:
    t0 = time.time()
    print(f"\n=== {dataset} ===", flush=True)
    p = call("POST", "/projects")
    pid = p["id"]
    call("POST", f"/projects/{pid}/dataset", json={"name": dataset})
    p = call("POST", f"/projects/{pid}/scan", json={})
    d, card = p["defaults"], p["card"]
    step(f"scanned {card['kind']} · task={d['task']} metric={d['target_metric']} target={card.get('target')}", t0)
    call("POST", f"/projects/{pid}/confirm", json={"task": d["task"], "target": card.get("target"),
                                                   "target_metric": d["target_metric"],
                                                   "target_value": d.get("target_value")})
    p = call("POST", f"/projects/{pid}/context",
             json={"notes": NOTES.get(dataset, ""), "constraints": {"device": "auto"}})
    step(f"context [{p['context']['source']}] {p['context']['summary'][:110]}...", t0)
    call("POST", f"/projects/{pid}/context/approve")
    p = call("POST", f"/projects/{pid}/suggest", json={"feedback": ""})
    sug = p["suggestions"]["suggestions"]
    step(f"suggest [{p['suggestions']['source']}] " + ", ".join(s["spec"]["model"] for s in sug), t0)
    if model and model not in [s["spec"]["model"] for s in sug]:
        p = call("POST", f"/projects/{pid}/suggest", json={"feedback": f"I want a {model} model."})
        sug = p["suggestions"]["suggestions"]
        step(f"re-suggest with feedback [{p['suggestions']['source']}] "
             + ", ".join(s["spec"]["model"] for s in sug), t0)
    idx = next((i for i, s in enumerate(sug) if s["spec"]["model"] == model), 0) if model else 0
    spec = sug[idx]["spec"]
    if epochs:
        spec["train"]["epochs"] = epochs
    p = call("POST", f"/projects/{pid}/suggest/approve", json={"index": idx, "spec_yaml": yaml.safe_dump(spec)})
    step(f"adapt review [{p['adapt_review']['source']}] {p['adapt_review']['notes'][:100]}...", t0)
    p = call("POST", f"/projects/{pid}/adapt/apply", json={"plan": p["adapt_plan"]}, timeout=900)
    p = call("POST", f"/projects/{pid}/train", json={})
    rid = p["current_run"]
    for loop in range(LOOPS + 1):
        if not train_and_eval(rid, t0):
            return pid
        nx = call("POST", f"/runs/{rid}/next")
        step(f"next [{nx.get('source')}] {nx['action']} {nx.get('change') or ''} - {nx['reason'][:90]}", t0)
        if loop == LOOPS or nx["action"] != "retrain" or not nx.get("change"):
            break
        rid = call("POST", f"/runs/{rid}/next/start", json=nx["change"])["new_run_id"]
    md = call("POST", f"/runs/{rid}/report")
    step(f"report {len(md)} chars: {md.splitlines()[2][:100] if len(md.splitlines()) > 2 else ''}", t0)
    return pid


def train_and_eval(rid: str, t0: float) -> bool:
    spec = call("GET", f"/runs/{rid}")["spec"]
    step(f"training {rid} ({spec['model']}, {spec['train']['epochs']} epochs)", t0)
    seen = -1
    while True:
        v = call("GET", f"/runs/{rid}")
        st = v["status"]
        if v["rows"] and v["rows"][-1]["epoch"] != seen:
            seen = v["rows"][-1]["epoch"]
            last = v["rows"][-1]
            if seen % 5 == 4 or seen == 0:
                step(f"epoch {seen}: train_loss={last.get('train_loss')} val_metric={last.get('val_metric')}", t0)
        if st["state"] in ("completed", "early_stopped", "stopped", "failed") and not st.get("alive"):
            break
        time.sleep(3)
    step(f"run {st['state']} {st.get('reason') or ''}", t0)
    if st["state"] == "failed":
        print(v.get("error") or st.get("worker_tail"))
        return False
    ev = call("POST", f"/runs/{rid}/evaluate", timeout=900)
    step(f"eval {ev['metrics']} target_hit={ev.get('target_hit')}", t0)
    return True


def main() -> None:
    jobs = sys.argv[1:] or ["titanic:mlp", "wine", "medical-pills:yolov8n:20"]
    for job in jobs:
        name, *rest = job.split(":")
        run(name, rest[0] if rest and rest[0] else None, int(rest[1]) if len(rest) > 1 else None)


if __name__ == "__main__":
    main()
