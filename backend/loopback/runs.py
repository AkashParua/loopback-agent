"""Run lifecycle: create run dir, launch worker process, control, status."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import memory
from .config import RUNS_DIR, ROOT
from .jobspec import JobSpec
from .logs import event_logger, read_events, read_train_log
from .train.runctx import read_json, write_json

_procs: dict[str, subprocess.Popen] = {}
TERMINAL = {"completed", "early_stopped", "stopped", "failed"}


def run_dir(run_id: str) -> Path:
    d = (RUNS_DIR / run_id).resolve()
    if d.parent != RUNS_DIR.resolve() or not d.exists():
        raise FileNotFoundError(f"no run {run_id}")
    return d


def new_run_id(project_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    n = len([p for p in RUNS_DIR.glob(f"{project_id}-*")])
    return f"{project_id}-{n + 1:02d}-{stamp}"


def start(project_id: str, spec: JobSpec, *, parent_run_id: Optional[str] = None,
          change: Optional[str] = None) -> str:
    run_id = new_run_id(project_id)
    d = RUNS_DIR / run_id
    d.mkdir(parents=True)
    spec.save(d / "job_spec.yaml")
    write_json(d / "status.json", {"state": "queued"})
    ev = event_logger(d, run_id=run_id)
    if parent_run_id:
        ev.info("change", parent=parent_run_id, detail=change)
    memory.record_start(run_id, project_id, spec, parent_run_id=parent_run_id, change=change)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "backend") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    log = open(d / "worker.out", "w")
    _procs[run_id] = subprocess.Popen([sys.executable, "-m", "loopback.train.worker", str(d)],
                                      stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(d))
    return run_id


def control(run_id: str, action: str) -> dict:
    if action not in ("pause", "resume", "stop"):
        raise ValueError(action)
    d = run_dir(run_id)
    write_json(d / "control.json", {"action": action})
    event_logger(d, run_id=run_id).info("user_" + action)
    return status(run_id)


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie still answers kill(0); check its state
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def status(run_id: str) -> dict:
    d = run_dir(run_id)
    st = read_json(d / "status.json", {}) or {}
    proc = _procs.get(run_id)
    # after a backend restart _procs is empty: fall back to the worker pid in status.json
    alive = proc.poll() is None if proc is not None else _pid_alive(st.get("pid"))
    if not alive and st.get("state") in ("running", "paused", "queued"):
        if proc is not None or st.get("state") != "queued":
            # process vanished without a terminal status: treat as crash
            tail = (d / "worker.out").read_text()[-1500:] if (d / "worker.out").exists() else ""
            st = {**st, "state": "failed", "reason": "worker exited unexpectedly", "worker_tail": tail}
            write_json(d / "status.json", st)
            memory.update(run_id, status="failed", failure_mode="crash")
    st["alive"] = alive
    return st


def snapshot(run_id: str) -> dict:
    d = run_dir(run_id)
    return {
        "run_id": run_id,
        "status": status(run_id),
        "spec": JobSpec.load(d / "job_spec.yaml").model_dump(),
        "rows": read_train_log(d),
        "events": read_events(d)[-50:],
        "metrics": read_json(d / "metrics.json"),
        "eval": read_json(d / "eval.json"),
        "error": (d / "error.txt").read_text()[-3000:] if (d / "error.txt").exists() else None,
        "has_report": (d / "report.md").exists(),
    }


def wait(run_id: str, timeout: float = 3600) -> dict:
    proc = _procs.get(run_id)
    if proc:
        proc.wait(timeout=timeout)
    return status(run_id)
