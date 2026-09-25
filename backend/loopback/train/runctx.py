"""Shared plumbing for every trainer: logs, control file, status, guards.

The backend talks to a running worker only through files in the run dir:
  control.json  {"action": "pause" | "resume" | "stop"}   (written by backend)
  status.json   {"state": ..., "reason": ..., "epoch": ...} (written by worker)
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..jobspec import JobSpec
from ..logs import TrainLog, event_logger

DEAD_GRAD = 1e-7
DEAD_GRAD_EPOCHS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class RunContext:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.spec = JobSpec.load(run_dir / "job_spec.yaml")
        self.log = TrainLog(run_dir)
        self.events = event_logger(run_dir, run_id=run_dir.name)
        self.stop_reason: Optional[str] = None
        self.stop_kind: Optional[str] = None  # early_stopped | stopped
        self._dead = 0
        self.epoch = -1
        self.status("running")

    # ------------------------------------------------------------ status
    def status(self, state: str, **extra: Any) -> None:
        cur = read_json(self.dir / "status.json", {}) or {}
        cur.update({"state": state, "epoch": self.epoch, "updated_at": _now(), "pid": os.getpid(), **extra})
        cur.setdefault("started_at", _now())
        write_json(self.dir / "status.json", cur)

    # ------------------------------------------------------------ control
    def _action(self) -> Optional[str]:
        return (read_json(self.dir / "control.json", {}) or {}).get("action")

    def check_control(self) -> bool:
        """Blocks while paused. Returns True if the run must stop."""
        action = self._action()
        if action == "pause":
            self.events.info("paused", epoch=self.epoch)
            self.status("paused")
            while (action := self._action()) == "pause":
                time.sleep(1)
            self.events.info("resumed", epoch=self.epoch)
            self.status("running")
        if action == "stop":
            self.request_stop("stopped", "stopped by user")
            return True
        return self.stop_reason is not None

    def request_stop(self, kind: str, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_kind, self.stop_reason = kind, reason
            self.events.info("early_stop" if kind == "early_stopped" else "stop",
                             epoch=self.epoch, detail=reason)

    # ------------------------------------------------------------ per-epoch
    def epoch_end(self, epoch: int, **row: Any) -> bool:
        """Write the log line, run guards, poll control. Returns True -> stop."""
        self.epoch = epoch
        row = self.log.write(epoch=epoch, **row)
        for key in ("train_loss", "val_loss"):
            v = row.get(key)
            if isinstance(v, str) or (isinstance(v, float) and not math.isfinite(v)):
                self.request_stop("early_stopped", f"{key} is {v} at epoch {epoch}")
        g = row.get("grad_norm")
        if isinstance(g, (int, float)):
            self._dead = self._dead + 1 if g < DEAD_GRAD else 0
            if self._dead >= DEAD_GRAD_EPOCHS:
                self.request_stop("early_stopped",
                                  f"gradients dead: grad_norm < {DEAD_GRAD} for {self._dead} epochs (epoch {epoch})")
        self.status("running")
        return self.check_control()

    # ------------------------------------------------------------ end
    def finish(self, metrics: dict, *, patience_stop: bool = False) -> dict:
        if patience_stop and self.stop_reason is None:
            self.request_stop("early_stopped",
                              f"val_metric flat for patience={self.spec.train.patience} epochs")
        state = self.stop_kind or "completed"
        out = {"state": state, "reason": self.stop_reason, "epochs_run": self.epoch + 1,
               "target_metric": self.spec.target_metric, **metrics}
        write_json(self.dir / "metrics.json", out)
        self.events.info("finished", state=state, detail=self.stop_reason,
                         **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        self.status(state, reason=self.stop_reason, ended_at=_now())
        return out


def resolve_device(device: str) -> str:
    """A job spec's "auto" defers to LOOPBACK_DEVICE, then to CUDA if present."""
    import torch

    from ..config import DEVICE

    if device in ("auto", "", None):
        device = DEVICE
    if device in ("auto", "", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
