"""Run logs that both humans and a 4B model can read.

Files (README 'Training logs'):
  train.log     one JSON object per epoch
  events.jsonl  agent decisions (start / verdict / change / early_stop / ...)

`digest()` turns them into a compact text block: a fixed-width table of the
last N epochs, computed trends and the event trail. Small models read
aligned tables far better than raw JSON, and the trend lines save them from
doing arithmetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)


def event_logger(run_dir: Path, **bind: Any) -> structlog.BoundLogger:
    """structlog logger writing JSON lines to <run_dir>/events.jsonl."""
    fh = open(run_dir / "events.jsonl", "a", buffering=1)
    return structlog.wrap_logger(
        structlog.WriteLogger(fh),
        processors=[
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            structlog.processors.JSONRenderer(),
        ],
    ).bind(**bind)


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)  # keep NaN visible, keep JSON valid
    try:  # numpy / torch scalars
        return _clean(float(v)) if hasattr(v, "item") else v
    except (TypeError, ValueError):
        return v


class TrainLog:
    """Append-only per-epoch log: {"epoch": 12, "train_loss": 0.41, "val_metric": 0.71, "lr": 0.001}"""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "train.log"

    def write(self, **row: Any) -> dict:
        row = {k: _clean(v) for k, v in row.items()}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def read_train_log(run_dir: Path) -> list[dict]:
    rows = _read_jsonl(run_dir / "train.log")
    for r in rows:  # restore NaN strings to floats for analysis
        for k, v in r.items():
            if v in ("nan", "inf", "-inf"):
                r[k] = float(v)
    return rows


def read_events(run_dir: Path) -> list[dict]:
    return _read_jsonl(run_dir / "events.jsonl")


# ------------------------------------------------------------------ digest

def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.2e}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def table(rows: list[dict], cols: Iterable[str]) -> str:
    cols = [c for c in cols if any(c in r for r in rows)]
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = lambda xs: "  ".join(x.rjust(w) for x, w in zip(xs, widths))  # noqa: E731
    return "\n".join([line(cols), *(line(r) for r in cells)])


def curve_facts(rows: list[dict], higher_is_better: bool) -> dict:
    """Numbers the LLM should not have to compute itself."""
    vals = [(r["epoch"], r["val_metric"]) for r in rows
            if isinstance(r.get("val_metric"), (int, float)) and math.isfinite(r["val_metric"])]
    losses = [r.get("train_loss") for r in rows if isinstance(r.get("train_loss"), (int, float))]
    facts: dict[str, Any] = {"epochs_logged": len(rows)}
    if vals:
        best = (max if higher_is_better else min)(vals, key=lambda t: t[1])
        facts |= {"best_val_metric": best[1], "best_epoch": best[0],
                  "last_val_metric": vals[-1][1], "epochs_since_best": vals[-1][0] - best[0]}
        if len(vals) >= 4:
            tail = [v for _, v in vals[-4:]]
            facts["val_slope_last4"] = (tail[-1] - tail[0]) / 3
    if losses:
        finite = [x for x in losses if math.isfinite(x)]
        facts |= {"first_train_loss": losses[0], "last_train_loss": losses[-1],
                  "min_train_loss": min(finite) if finite else None,
                  "nonfinite_losses": len(losses) - len(finite)}
    return facts


LOG_COLS = ["epoch", "train_loss", "val_loss", "val_metric", "lr", "grad_norm",
            "mAP50", "mAP50-95", "precision", "recall"]


def digest(run_dir: Path, *, higher_is_better: bool, tail: int = 12,
           header: str = "", signals: list[str] | None = None,
           tried: list[str] | None = None) -> str:
    rows = read_train_log(run_dir)
    events = [e for e in read_events(run_dir) if e.get("event") != "epoch"]
    parts = [header.strip()] if header else []
    if rows:
        shown = rows[-tail:]
        parts.append(f"LOG (last {len(shown)} of {len(rows)} epochs; val_metric is "
                     f"{'higher' if higher_is_better else 'lower'}-is-better)")
        parts.append(table(shown, LOG_COLS))
        facts = curve_facts(rows, higher_is_better)
        parts.append("TRENDS " + ", ".join(f"{k}={_fmt(v)}" for k, v in facts.items()))
    else:
        parts.append("LOG empty (no epoch finished yet)")
    parts.append("SIGNALS " + ("; ".join(signals) if signals else "none"))
    if events:
        trail = [f"{e.get('event')}" + (f"({e['detail']})" if e.get("detail") else "")
                 for e in events[-6:]]
        parts.append("EVENTS " + " -> ".join(trail))
    if tried:
        parts.append("ALREADY TRIED " + "; ".join(tried))
    return "\n".join(parts)
