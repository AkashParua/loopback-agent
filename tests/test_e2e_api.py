"""Whole flow through the HTTP API with the agent offline (rule-based fallbacks)."""

import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from loopback.api import app
from loopback.config import DATASETS_DIR


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def ok(r):
    assert r.status_code == 200, r.text
    return r


def wait_run(c, rid, timeout=300):
    t = time.time()
    while time.time() - t < timeout:
        v = c.get(f"/runs/{rid}").json()
        if not v["status"].get("alive") and v["status"]["state"] != "queued":
            return v
        time.sleep(1)
    raise TimeoutError(rid)


def flow(c, dataset, target, model, max_epochs):
    pid = ok(c.post("/projects")).json()["id"]
    ok(c.post(f"/projects/{pid}/dataset", json={"name": dataset}))
    p = ok(c.post(f"/projects/{pid}/scan", json={"target": target})).json()
    d = p["defaults"]
    ok(c.post(f"/projects/{pid}/confirm", json={k: d[k] for k in ("task", "target", "target_metric", "target_value")}))
    p = ok(c.post(f"/projects/{pid}/context", json={"notes": "", "constraints": {"max_epochs": max_epochs}})).json()
    assert p["context"]["source"] == "rules"
    ok(c.post(f"/projects/{pid}/context/approve"))
    p = ok(c.post(f"/projects/{pid}/suggest", json={})).json()
    sug = p["suggestions"]["suggestions"]
    assert 1 <= len(sug) <= 3
    idx = next(i for i, s in enumerate(sug) if s["spec"]["model"] == model)
    p = ok(c.post(f"/projects/{pid}/suggest/approve", json={"index": idx})).json()
    p = ok(c.post(f"/projects/{pid}/adapt/apply", json={"plan": p["adapt_plan"]})).json()
    assert p["unlocked"] == 5
    rid = ok(c.post(f"/projects/{pid}/train", json={})).json()["current_run"]
    v = wait_run(c, rid)
    assert v["status"]["state"] in ("completed", "early_stopped"), v
    assert v["rows"] and set(v["rows"][0]) >= {"epoch", "train_loss", "val_metric"}
    assert ok(c.get(f"/projects/{pid}")).json()["unlocked"] >= 7
    vd = ok(c.post(f"/runs/{rid}/review")).json()
    assert vd["verdict"] in ("continue", "change", "stop") and vd["source"] == "rules"
    ev = ok(c.post(f"/runs/{rid}/evaluate")).json()
    assert ev["value"] is not None and ev["images"]
    for img in ev["images"]:
        ok(c.get(f"/runs/{rid}/files/{img}"))
    nx = ok(c.post(f"/runs/{rid}/next")).json()
    assert nx["action"] in ("retrain", "finish")
    md = ok(c.post(f"/runs/{rid}/report")).text
    assert "## Final metrics" in md and "## Experiment history" in md
    return pid, rid, nx


def test_tabular_flow_and_loop(client, tabular_csv):
    dst = DATASETS_DIR / "tiny_tab"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(tabular_csv, dst / "data.csv")
    pid, rid, nx = flow(client, "tiny_tab", "label", "mlp", max_epochs=4)
    # the loop: approve the agent's next change -> child run recorded in memory with its parent
    if nx["action"] == "retrain":
        new = ok(client.post(f"/runs/{rid}/next/start", json=nx["change"])).json()["new_run_id"]
        wait_run(client, new)
        exps = ok(client.get(f"/projects/{pid}/experiments")).json()
        child = next(e for e in exps if e["run_id"] == new)
        assert child["parent_run_id"] == rid and child["change"]
        # the same idea cannot be started twice
        r = client.post(f"/runs/{rid}/next/start", json=nx["change"])
        assert r.status_code == 400 and "already tried" in r.text


def test_control_stop(client, tabular_csv):
    dst = DATASETS_DIR / "tiny_tab"
    pid = ok(client.post("/projects")).json()["id"]
    ok(client.post(f"/projects/{pid}/dataset", json={"name": "tiny_tab"}))
    p = ok(client.post(f"/projects/{pid}/scan", json={"target": "label"})).json()
    d = p["defaults"]
    ok(client.post(f"/projects/{pid}/confirm", json={k: d[k] for k in ("task", "target", "target_metric", "target_value")}))
    ok(client.post(f"/projects/{pid}/context", json={}))
    ok(client.post(f"/projects/{pid}/context/approve"))
    p = ok(client.post(f"/projects/{pid}/suggest", json={})).json()
    idx = next(i for i, s in enumerate(p["suggestions"]["suggestions"]) if s["spec"]["model"] == "mlp")
    p = ok(client.post(f"/projects/{pid}/suggest/approve", json={"index": idx})).json()
    spec = p["spec"]  # pre-adapt spec: data.path is still the raw file; the backend must override it
    ok(client.post(f"/projects/{pid}/adapt/apply", json={"plan": p["adapt_plan"]}))
    spec["train"].update(epochs=500, patience=500)
    import yaml
    rid = ok(client.post(f"/projects/{pid}/train", json={"spec_yaml": yaml.safe_dump(spec)})).json()["current_run"]
    t = time.time()
    while time.time() - t < 120:
        v = client.get(f"/runs/{rid}").json()
        assert v["status"]["state"] != "failed", v
        if v["rows"]:
            break
        time.sleep(0.5)
    ok(client.post(f"/runs/{rid}/control", json={"action": "stop"}))
    v = wait_run(client, rid)
    assert v["status"]["state"] == "stopped" and "user" in v["status"]["reason"]


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="set RUN_SLOW=1 (downloads yolov8n weights)")
def test_detect_flow(client, yolo_dir):
    dst = DATASETS_DIR / "tiny_yolo"
    if not dst.exists():
        shutil.copytree(yolo_dir, dst)
    flow(client, "tiny_yolo", None, "yolov8n", max_epochs=2)
