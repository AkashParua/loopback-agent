"""Training worker: `python -m loopback.train.worker <run_dir>`.

One process per run so stop/pause/crash never takes the API down.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from .. import memory
from .runctx import RunContext


def main(run_dir: Path) -> int:
    ctx = RunContext(run_dir)
    run_id = run_dir.name
    ctx.events.info("start", model=ctx.spec.model, task=ctx.spec.task,
                    detail=f"{ctx.spec.model} epochs={ctx.spec.train.epochs}")
    try:
        if ctx.spec.task == "detect":
            from .detect import train_detect
            out = train_detect(ctx)
        else:
            from .tabular import train_tabular
            out = train_tabular(ctx)
    except Exception as exc:
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb)
        mode = "out_of_memory" if "out of memory" in str(exc).lower() else "crash"
        ctx.events.error("crash", detail=f"{type(exc).__name__}: {str(exc)[:300]}")
        ctx.status("failed", reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        memory.update(run_id, status="failed", failure_mode=mode, notes=str(exc)[:500])
        return 1
    memory.update(run_id, status=out["state"], best_metric=out.get("best_val_metric"),
                  failure_mode="nan_loss" if out.get("reason") and "nan" in out["reason"] else None,
                  notes=out.get("reason"))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
