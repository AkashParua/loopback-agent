"""SQLite `experiments` table: every job spec + result, so failed ideas are not retried."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from .config import DB_PATH
from .jobspec import JobSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id        TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    parent_run_id TEXT,
    created_at    TEXT NOT NULL,
    model         TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    change        TEXT,
    spec_yaml     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running',
    target_metric TEXT NOT NULL,
    final_metric  REAL,
    best_metric   REAL,
    failure_mode  TEXT,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS ix_exp_project ON experiments(project_id);
"""


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def record_start(run_id: str, project_id: str, spec: JobSpec, *,
                 parent_run_id: Optional[str] = None, change: Optional[str] = None) -> None:
    with _db() as con:
        con.execute(
            "INSERT OR REPLACE INTO experiments (run_id, project_id, parent_run_id, created_at, model,"
            " fingerprint, change, spec_yaml, target_metric) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, project_id, parent_run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"),
             spec.model, spec.fingerprint(), change, spec.to_yaml(), spec.target_metric),
        )


def update(run_id: str, **fields: Any) -> None:
    allowed = {"status", "final_metric", "best_metric", "failure_mode", "notes"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _db() as con:
        con.execute(f"UPDATE experiments SET {sets} WHERE run_id = ?", (*fields.values(), run_id))


def history(project_id: str) -> list[dict]:
    with _db() as con:
        rows = con.execute(
            "SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get(run_id: str) -> Optional[dict]:
    with _db() as con:
        row = con.execute("SELECT * FROM experiments WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def tried_fingerprints(project_id: str) -> dict[str, dict]:
    return {r["fingerprint"]: r for r in history(project_id)}


def already_tried(project_id: str, spec: JobSpec) -> Optional[dict]:
    return tried_fingerprints(project_id).get(spec.fingerprint())


def history_lines(project_id: str) -> list[str]:
    """One line per experiment, for LLM prompts."""
    out = []
    for r in history(project_id):
        metric = r["best_metric"] if r["best_metric"] is not None else r["final_metric"]
        line = (f"{r['run_id']}: model={r['model']} status={r['status']} "
                f"{r['target_metric']}={'-' if metric is None else round(metric, 4)}")
        if r["change"]:
            line += f" change=({r['change']})"
        if r["failure_mode"]:
            line += f" failure={r['failure_mode']}"
        out.append(line)
    return out


def to_json(project_id: str) -> str:
    return json.dumps(history(project_id), indent=2)
