"""Step 6 - Monitor: the LLM reads the log digest and says continue / change / stop.

Rules (README 'The loop'):
  * every proposed change points at a measured log line (evidence_epoch must exist),
  * memory blocks re-trying a spec that was already run,
  * NaN loss / dead gradients / flat val metric stop the run no matter what the LLM says.

The same validation is reused after evaluation (`propose_next`) and for failed
runs (`diagnose`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .. import memory, runs
from ..agent_client import agent
from ..config import LOG_TAIL
from ..jobspec import JobSpec
from ..logs import curve_facts, digest, event_logger, read_train_log
from ..train.runctx import read_json, write_json
from ..zoo import ZOO, models_for


@dataclass
class Signal:
    kind: str       # nan_loss | dead_gradients | flat | plateau | diverging | overfitting | underfitting | target_hit
    epoch: int
    severity: str   # stop | change | info
    evidence: str

    def line(self) -> str:
        return f"{self.kind.upper()} at epoch {self.epoch}: {self.evidence} -> {self.severity}"


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def signals(rows: list[dict], spec: JobSpec) -> list[Signal]:
    out: list[Signal] = []
    if not rows:
        return out
    higher = spec.higher_is_better
    patience = spec.train.patience
    for r in rows:
        for k in ("train_loss", "val_loss"):
            v = r.get(k)
            if (isinstance(v, float) and not math.isfinite(v)) or v in ("nan", "inf"):
                out.append(Signal("nan_loss", r["epoch"], "stop", f"{k}={v}"))
                return out
    g = [(r["epoch"], _num(r.get("grad_norm"))) for r in rows[-3:]]
    if len(g) == 3 and all(v is not None and v < 1e-7 for _, v in g):
        out.append(Signal("dead_gradients", g[-1][0], "stop", f"grad_norm={g[-1][1]:.1e} for 3 epochs"))

    facts = curve_facts(rows, higher)
    since = facts.get("epochs_since_best", 0)
    if len(rows) > 1 and since >= patience:
        out.append(Signal("flat", rows[-1]["epoch"], "stop",
                          f"val_metric best {facts['best_val_metric']:.4g} at epoch {facts['best_epoch']}, "
                          f"no gain for {since} epochs (patience {patience})"))
    elif len(rows) > 1 and since >= max(3, patience // 2):
        out.append(Signal("plateau", rows[-1]["epoch"], "change",
                          f"val_metric best {facts['best_val_metric']:.4g} at epoch {facts['best_epoch']}, "
                          f"flat for {since} epochs"))

    tl = [(r["epoch"], _num(r.get("train_loss"))) for r in rows if _num(r.get("train_loss")) is not None]
    if len(tl) >= 4:
        last = [v for _, v in tl[-3:]]
        lo = min(v for _, v in tl)
        if last[0] < last[1] < last[2] and last[2] > 1.5 * lo:
            out.append(Signal("diverging", tl[-1][0], "change",
                              f"train_loss rose {last[0]:.4g}->{last[2]:.4g} (min {lo:.4g})"))
        if len(tl) >= 5 and tl[-1][1] > 0.9 * tl[0][1]:
            out.append(Signal("underfitting", tl[-1][0], "change",
                              f"train_loss {tl[0][1]:.4g}->{tl[-1][1]:.4g} after {len(tl)} epochs"))

    vl = [(r["epoch"], _num(r.get("val_loss")), _num(r.get("train_loss"))) for r in rows]
    vl = [x for x in vl if x[1] is not None and x[2] is not None]
    if len(vl) >= 4:
        a, b, c, d = vl[-4:]
        if a[1] < b[1] < c[1] < d[1] and d[2] < a[2]:
            out.append(Signal("overfitting", d[0], "change",
                              f"val_loss {a[1]:.4g}->{d[1]:.4g} while train_loss {a[2]:.4g}->{d[2]:.4g}"))

    if spec.target_value is not None and "best_val_metric" in facts:
        best = facts["best_val_metric"]
        if (best >= spec.target_value) if higher else (best <= spec.target_value):
            out.append(Signal("target_hit", facts["best_epoch"], "info",
                              f"{spec.target_metric}={best:.4g} meets target {spec.target_value}"))
    return out


# ================================================================== changes

APPLICABLE = {
    "lr": {"mlp", "tabnet", "yolov8n", "yolov8s", "yolo11n"},
    "batch": {"mlp", "tabnet", "yolov8n", "yolov8s", "yolo11n"},
    "epochs": {"mlp", "tabnet", "yolov8n", "yolov8s", "yolo11n"},
    "patience": {"mlp", "tabnet", "yolov8n", "yolov8s", "yolo11n"},
    "dropout": {"mlp"},
    "weight_decay": {"mlp"},
    "imgsz": {"yolov8n", "yolov8s", "yolo11n"},
    "model": set(ZOO),
}


def apply_change(spec: JobSpec, param: str, value: Any) -> tuple[JobSpec, str]:
    """Returns (new spec, human-readable change). Raises ValueError if invalid."""
    if spec.model not in APPLICABLE.get(param, set()):
        raise ValueError(f"'{param}' does not apply to model {spec.model}")
    new = spec.model_copy(deep=True)
    new.reason = None
    if param == "model":
        value = str(value).strip()
        if value not in {e.name for e in models_for(spec.task)}:
            raise ValueError(f"model {value!r} not in zoo for task {spec.task}")
        old, new.model = spec.model, value
        new.framework = ZOO[value].framework if ZOO[value].framework == "sklearn" else "pytorch"
        if new.framework == "sklearn":
            new.train.epochs, new.train.patience, new.train.device = 1, 1, "cpu"
        return new, f"model {old} -> {value}"
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"value {value!r} for {param} is not a number")
    if not math.isfinite(num) or num <= 0 and param != "dropout":
        raise ValueError(f"invalid {param}={value}")
    if param == "imgsz":
        old = spec.adapt.imgsz
        new.adapt.imgsz = int(max(160, min(1536, round(num / 32) * 32)))
        return new, f"imgsz {old} -> {new.adapt.imgsz}"
    bounds = {"lr": (1e-6, 1.0), "batch": (2, 2048), "epochs": (1, 500), "patience": (1, 100),
              "dropout": (0.0, 0.8), "weight_decay": (0.0, 0.5)}[param]
    num = max(bounds[0], min(bounds[1], num))
    if param in ("batch", "epochs", "patience"):
        num = int(num)
    old = getattr(spec.train, param)
    if old == num:
        raise ValueError(f"{param} is already {old}")
    setattr(new.train, param, num)
    return new, f"{param} {old} -> {num}"


def _validate_change(spec: JobSpec, project_id: str, change: Optional[dict]) -> Optional[str]:
    if not change:
        return "verdict 'change' without a change"
    try:
        new, _ = apply_change(spec, change["param"], change["value"])
    except ValueError as exc:
        return str(exc)
    if prev := memory.already_tried(project_id, new):
        return f"already tried in {prev['run_id']}"
    return None


def _rule_change(spec: JobSpec, project_id: str, sigs: list[Signal]) -> Optional[dict]:
    """First valid, untried change suggested by the strongest signal."""
    kinds = {s.kind for s in sigs}
    lr = spec.train.lr or (0.01 if spec.task == "detect" else 1e-3)
    ideas: list[tuple[str, Any]] = []
    if kinds & {"diverging", "nan_loss"}:
        ideas += [("lr", lr / 3), ("batch", spec.train.batch * 2)]
    if "overfitting" in kinds:
        ideas += [("dropout", min(0.6, (spec.train.dropout or 0) + 0.2)),
                  ("weight_decay", (spec.train.weight_decay or 1e-4) * 10)]
    if kinds & {"plateau", "flat"}:
        ideas += [("lr", lr / 3)]
    if "underfitting" in kinds:
        ideas += [("lr", lr * 3)]
        bigger = {"yolov8n": "yolov8s", "yolo11n": "yolov8s", "logreg": "random_forest",
                  "ridge": "random_forest", "random_forest": "mlp", "mlp": "tabnet"}.get(spec.model)
        if bigger:
            ideas.append(("model", bigger))
    for param, value in ideas:
        change = {"param": param, "value": str(value)}
        if _validate_change(spec, project_id, change) is None:
            return change
    return None


def _rules_verdict(spec: JobSpec, project_id: str, sigs: list[Signal], rows: list[dict]) -> dict:
    last_epoch = rows[-1]["epoch"] if rows else 0
    stop = next((s for s in sigs if s.severity == "stop"), None)
    if stop:
        return {"verdict": "stop", "evidence_epoch": stop.epoch, "evidence": stop.evidence,
                "change": None, "reason": f"{stop.kind}: the run cannot recover"}
    for s in sigs:
        if s.severity == "change" and (ch := _rule_change(spec, project_id, [s])):
            return {"verdict": "change", "evidence_epoch": s.epoch, "evidence": s.evidence,
                    "change": ch, "reason": f"{s.kind} detected"}
    facts = curve_facts(rows, spec.higher_is_better)
    ev = (f"val_metric {facts['last_val_metric']:.4g} at epoch {last_epoch}"
          if "last_val_metric" in facts else "no epoch finished yet")
    return {"verdict": "continue", "evidence_epoch": last_epoch, "evidence": ev, "change": None,
            "reason": "no problem signal in the log"}


def _header(run_id: str, spec: JobSpec) -> str:
    target = f"{spec.target_metric} target={spec.target_value}" if spec.target_value is not None \
        else spec.target_metric
    t = spec.train
    return (f"RUN {run_id} model={spec.model} task={spec.task} {target}\n"
            f"SPEC epochs={t.epochs} batch={t.batch} lr={t.lr} patience={t.patience}"
            + (f" dropout={t.dropout} weight_decay={t.weight_decay}" if spec.model == "mlp" else "")
            + (f" imgsz={spec.adapt.imgsz}" if spec.task == "detect" else ""))


def review(run_id: str, project_id: str, *, use_llm: bool = True) -> dict:
    d = runs.run_dir(run_id)
    spec = JobSpec.load(d / "job_spec.yaml")
    rows = read_train_log(d)
    sigs = signals(rows, spec)
    epochs = {r["epoch"] for r in rows}
    text = digest(d, higher_is_better=spec.higher_is_better, tail=LOG_TAIL, header=_header(run_id, spec),
                  signals=[s.line() for s in sigs], tried=memory.history_lines(project_id))
    hard = next((s for s in sigs if s.kind in ("nan_loss", "dead_gradients", "flat")), None)

    def validate(out: dict) -> Optional[str]:
        if rows and out.get("evidence_epoch") not in epochs:
            return f"evidence_epoch {out.get('evidence_epoch')} is not in the log"
        if hard and out.get("verdict") != "stop":
            return f"hard signal {hard.kind} requires stop"
        if out.get("verdict") == "change":
            return _validate_change(spec, project_id, out.get("change"))
        return None

    fallback = lambda: _rules_verdict(spec, project_id, sigs, rows)  # noqa: E731
    if use_llm and rows:
        out, source, note = agent.task("monitor", text, fallback=fallback, validate=validate)
    else:
        out, source, note = fallback(), "rules", None
    verdict = {**out, "source": source, "note": note, "run_id": run_id,
               "at_epoch": rows[-1]["epoch"] if rows else None, "signals": [asdict(s) for s in sigs],
               "status": "pending" if out["verdict"] != "continue" else "info", "digest": text}
    write_json(d / "verdict.json", verdict)
    event_logger(d, run_id=run_id).info("agent_verdict", verdict=out["verdict"], source=source,
                                        detail=f"{out['verdict']}: {out.get('evidence')}",
                                        change=out.get("change"))
    return verdict


def resolve_verdict(run_id: str, project_id: str, approve: bool) -> dict:
    """Approve -> act on the pending verdict (stop, or stop + child run). Reject -> log it."""
    d = runs.run_dir(run_id)
    v = read_json(d / "verdict.json")
    if not v or v.get("status") != "pending":
        raise ValueError("no pending verdict")
    ev = event_logger(d, run_id=run_id)
    result: dict[str, Any] = {"verdict": v["verdict"], "approved": approve}
    if not approve:
        ev.info("verdict_rejected", detail=v["verdict"])
    elif v["verdict"] == "stop":
        if runs.status(run_id).get("alive"):
            runs.control(run_id, "stop")
        memory.update(run_id, failure_mode=_mode_from_signals(v.get("signals", [])))
        ev.info("verdict_approved", detail="stop")
    elif v["verdict"] == "change":
        spec = JobSpec.load(d / "job_spec.yaml")
        new_spec, desc = apply_change(spec, v["change"]["param"], v["change"]["value"])
        if runs.status(run_id).get("alive"):
            runs.control(run_id, "stop")
        desc = f"{desc} (evidence: epoch {v['evidence_epoch']} {v['evidence']})"
        new_spec.reason = desc
        result["new_run_id"] = runs.start(project_id, new_spec, parent_run_id=run_id, change=desc)
        ev.info("verdict_approved", detail=desc, new_run=result["new_run_id"])
    v["status"] = "approved" if approve else "rejected"
    write_json(d / "verdict.json", v)
    return result


def _mode_from_signals(sigs: list[dict]) -> Optional[str]:
    for s in sigs:
        if s["kind"] in ("nan_loss", "dead_gradients", "diverging", "overfitting", "underfitting"):
            return s["kind"]
        if s["kind"] in ("flat", "plateau"):
            return "plateau"
    return None


# ================================================================== after a run

def diagnose(run_id: str, project_id: str) -> dict:
    d = runs.run_dir(run_id)
    spec = JobSpec.load(d / "job_spec.yaml")
    rows = read_train_log(d)
    sigs = signals(rows, spec)
    st = runs.status(run_id)
    err = (d / "error.txt").read_text()[-800:] if (d / "error.txt").exists() else ""
    text = digest(d, higher_is_better=spec.higher_is_better, tail=LOG_TAIL, header=_header(run_id, spec),
                  signals=[s.line() for s in sigs]) + f"\nSTATUS {st.get('state')} reason={st.get('reason')}"
    if err:
        text += f"\nERROR (tail)\n{err}"

    def fallback() -> dict:
        if "out of memory" in err.lower():
            return {"failure_mode": "out_of_memory", "evidence": "CUDA out of memory in error.txt",
                    "fix": f"batch {spec.train.batch} -> {max(2, spec.train.batch // 2)}"}
        if err:
            return {"failure_mode": "crash", "evidence": err.strip().splitlines()[-1][:200],
                    "fix": "fix the data/config error above and retry"}
        mode = _mode_from_signals([asdict(s) for s in sigs]) or "none"
        s = next((s for s in sigs if s.kind != "target_hit"), None)
        return {"failure_mode": mode, "evidence": s.evidence if s else "no problem signal",
                "fix": "see next-step proposal"}

    out, source, note = agent.task("diagnose", text, fallback=fallback)
    memory.update(run_id, failure_mode=out["failure_mode"] if out["failure_mode"] != "none" else None)
    return {**out, "source": source, "note": note}


def propose_next(run_id: str, project_id: str) -> dict:
    """After eval: finish if target met or nothing untried is left, else one evidence-based change."""
    d = runs.run_dir(run_id)
    spec = JobSpec.load(d / "job_spec.yaml")
    rows = read_train_log(d)
    ev = read_json(d / "eval.json") or {}
    score = ev.get("metrics", {}).get(spec.target_metric)
    hit = ev.get("target_hit")
    if hit:
        return {"action": "finish", "change": None, "source": "rules",
                "evidence": f"{spec.target_metric}={score} meets target {spec.target_value}",
                "reason": "target reached"}
    sigs = signals(rows, spec)
    if spec.target_value is not None and score is not None:
        sigs = [s for s in sigs if s.kind != "target_hit"]
    text = digest(d, higher_is_better=spec.higher_is_better, tail=LOG_TAIL, header=_header(run_id, spec),
                  signals=[s.line() for s in sigs], tried=memory.history_lines(project_id))
    text += f"\nEVAL (held-out {ev.get('split')}) {', '.join(f'{k}={v}' for k, v in ev.get('metrics', {}).items())}"
    if spec.target_value is not None and score is not None:
        text += (f"\nHELD-OUT TARGET MET: no ({spec.target_metric}={score} vs target {spec.target_value}; "
                 "val numbers above are optimistic)")

    def fallback() -> dict:
        ch = _rule_change(spec, project_id, sigs)
        if not ch and spec.task != "detect" and spec.model in ("logreg", "ridge", "random_forest"):
            ch = next(({"param": "model", "value": m} for m in ("random_forest", "mlp")
                       if _validate_change(spec, project_id, {"param": "model", "value": m}) is None), None)
        if not ch and spec.task == "detect":
            ch = next(({"param": p, "value": v} for p, v in (("epochs", spec.train.epochs * 2),
                                                             ("model", "yolov8s"), ("imgsz", 960))
                       if _validate_change(spec, project_id, {"param": p, "value": str(v)}) is None), None)
        if ch:
            return {"action": "retrain", "change": ch, "evidence": f"held-out {spec.target_metric}={score}",
                    "reason": "target not met; next untried idea"}
        return {"action": "finish", "change": None, "evidence": f"held-out {spec.target_metric}={score}",
                "reason": "no new evidence-based change left"}

    rules = fallback()

    def validate(out: dict) -> Optional[str]:
        if out.get("action") == "retrain":
            return _validate_change(spec, project_id, out.get("change"))
        if rules["action"] == "retrain":
            return "target not met on held-out data and an untried change exists"
        return None

    out, source, note = agent.task("next_step", text, fallback=lambda: rules, validate=validate)
    return {**out, "source": source, "note": note}


def start_next(run_id: str, project_id: str, change: dict) -> str:
    d = runs.run_dir(run_id)
    spec = JobSpec.load(d / "job_spec.yaml")
    if problem := _validate_change(spec, project_id, change):
        raise ValueError(problem)
    new_spec, desc = apply_change(spec, change["param"], change["value"])
    new_spec.reason = desc
    return runs.start(project_id, new_spec, parent_run_id=run_id, change=desc)
