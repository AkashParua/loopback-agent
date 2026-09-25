"""Project state + one function per UI step. The API is a thin wrapper over this."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml
from pydantic import BaseModel, Field

from . import memory, runs
from .agent_client import agent
from .config import DATASETS_DIR, MONITOR_EVERY, PROJECTS_DIR
from .jobspec import JobSpec
from .logs import read_train_log
from .steps import adapt as adapt_step
from .steps import evaluate as eval_step
from .steps import monitor as monitor_step
from .steps import report as report_step
from .steps.context import Constraints, Context, context_digest, defaults, understand
from .steps.inspect_data import card_digest, detect_kind, inspect
from .steps.suggest import suggest as suggest_step
from .train.runctx import read_json, write_json

log = structlog.get_logger("pipeline")

STEPS = ["data", "context", "suggest", "adapt", "train", "watch", "eval", "report"]


class Project(BaseModel):
    id: str
    created_at: str
    unlocked: int = 1  # highest step (1-based) the user may open
    data_path: Optional[str] = None
    card: Optional[dict] = None
    card_digest: str = ""
    defaults: Optional[dict] = None
    context: Optional[dict] = None
    suggestions: Optional[dict] = None
    spec: Optional[dict] = None
    adapt_plan: Optional[dict] = None
    adapt_review: Optional[dict] = None
    adapt_diff: Optional[dict] = None
    run_ids: list[str] = Field(default_factory=list)
    current_run: Optional[str] = None


_lock = threading.RLock()


def _pdir(pid: str) -> Path:
    d = (PROJECTS_DIR / pid).resolve()
    if d.parent != PROJECTS_DIR.resolve():
        raise FileNotFoundError(pid)
    return d


def load(pid: str) -> Project:
    f = _pdir(pid) / "project.json"
    if not f.exists():
        raise FileNotFoundError(f"no project {pid}")
    return Project.model_validate_json(f.read_text())


def save(p: Project) -> Project:
    with _lock:
        d = _pdir(p.id)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "project.json", p.model_dump())
    return p


def unlock(p: Project, step: int) -> None:
    p.unlocked = max(p.unlocked, step)


def create() -> Project:
    pid = "p" + uuid.uuid4().hex[:6]
    return save(Project(id=pid, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))


def list_projects() -> list[dict]:
    out = []
    for f in sorted(PROJECTS_DIR.glob("*/project.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        p = Project.model_validate_json(f.read_text())
        out.append({"id": p.id, "created_at": p.created_at, "unlocked": p.unlocked,
                    "data": Path(p.data_path).name if p.data_path else None, "runs": len(p.run_ids)})
    return out


def _reset_from(p: Project, step: int) -> None:
    """Changing an earlier step invalidates everything after it."""
    if step <= 1:
        p.card = p.defaults = None
        p.card_digest = ""
    if step <= 2:
        p.context = None
    if step <= 3:
        p.suggestions = p.spec = None
    if step <= 4:
        p.adapt_plan = p.adapt_review = p.adapt_diff = None
    p.unlocked = min(p.unlocked, step)


# ------------------------------------------------------------------ step 1: data

def list_datasets() -> list[dict]:
    out = []
    if DATASETS_DIR.exists():
        for d in sorted(DATASETS_DIR.iterdir()):
            if d.is_dir():
                try:
                    kind, _ = detect_kind(d)
                except (ValueError, FileNotFoundError):
                    continue
                out.append({"name": d.name, "kind": kind})
    return out


def use_dataset(pid: str, name: str) -> Project:
    src = (DATASETS_DIR / name).resolve()
    if src.parent != DATASETS_DIR.resolve() or not src.exists():
        raise FileNotFoundError(name)
    p = load(pid)
    _reset_from(p, 1)
    p.data_path = str(src)
    return save(p)


def upload(pid: str, files: list[tuple[str, bytes]]) -> Project:
    p = load(pid)
    d = _pdir(pid) / "data"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for name, content in files:
        target = (d / Path(name).name)
        target.write_bytes(content)
        if target.suffix.lower() == ".zip":
            with zipfile.ZipFile(target) as zf:
                zf.extractall(d / target.stem)
            target.unlink()
    kids = [k for k in d.iterdir() if not k.name.startswith((".", "__MACOSX"))]
    _reset_from(p, 1)
    p.data_path = str(kids[0] if len(kids) == 1 else d)
    return save(p)


def scan(pid: str, target: Optional[str] = None) -> Project:
    p = load(pid)
    if not p.data_path:
        raise ValueError("add data first")
    card = inspect(p.data_path, target=target)
    _reset_from(p, 1)
    p.card, p.card_digest, p.defaults = card, card_digest(card), defaults(card)
    return save(p)


def confirm_data(pid: str, task: str, target: Optional[str], target_metric: str,
                 target_value: Optional[float]) -> Project:
    p = load(pid)
    if not p.card:
        raise ValueError("scan the data first")
    if p.card["kind"] == "tabular" and target != p.card.get("target"):
        p = scan(pid, target)  # target stats / leakage depend on the target
    ctx = Context(task=task, target=target if p.card["kind"] == "tabular" else None,
                  target_metric=target_metric, target_value=target_value)
    _reset_from(p, 2)
    p.context = ctx.model_dump()
    unlock(p, 2)
    return save(p)


# ------------------------------------------------------------------ step 2: context

def run_context(pid: str, notes: str, constraints: dict) -> Project:
    p = load(pid)
    ctx = Context.model_validate(p.context)
    ctx.notes = notes
    ctx.constraints = Constraints.model_validate(constraints)
    ctx = understand(p.card, ctx)
    _reset_from(p, 2)
    p.context = ctx.model_dump()
    unlock(p, 2)
    return save(p)


def approve_context(pid: str) -> Project:
    p = load(pid)
    if not p.context or not p.context.get("source"):
        raise ValueError("ask the agent to read the context first")
    unlock(p, 3)
    return save(p)


# ------------------------------------------------------------------ step 3: suggest

def run_suggest(pid: str, feedback: str = "") -> Project:
    p = load(pid)
    ctx = Context.model_validate(p.context)
    _reset_from(p, 3)
    p.suggestions = suggest_step(p.card, ctx, p.data_path, pid, feedback=feedback)
    unlock(p, 3)
    return save(p)


def approve_suggestion(pid: str, index: int, spec_yaml: Optional[str] = None) -> Project:
    p = load(pid)
    sug = p.suggestions["suggestions"][index]
    spec = JobSpec.from_yaml(spec_yaml) if spec_yaml else JobSpec.model_validate(sug["spec"])
    spec.reason = spec.reason or sug["reason"]
    _reset_from(p, 4)
    p.spec = spec.model_dump()
    unlock(p, 4)
    save(p)
    return plan_adapt(pid)


# ------------------------------------------------------------------ step 4: adapt

def plan_adapt(pid: str) -> Project:
    p = load(pid)
    ctx = Context.model_validate(p.context)
    spec = JobSpec.model_validate(p.spec)
    if p.card["kind"] == "tabular":
        plan = adapt_step.plan_tabular(p.card, ctx.target, ctx.task, ctx.constraints.drop_columns)
        kept = [c for c in plan.numeric + plan.onehot + plan.ordinal]
        review_ctx = "\n".join([p.card_digest, context_digest(ctx), "PLAN",
                                *(f"drop {d.column}: {d.reason}" for d in plan.drop),
                                f"numeric: {', '.join(plan.numeric) or '-'}",
                                f"one-hot: {', '.join(plan.onehot) or '-'}",
                                f"ordinal: {', '.join(plan.ordinal) or '-'}",
                                f"class_weight: {plan.class_weight}"])
        choices = {"columns": kept} if kept else {}
    else:
        plan = adapt_step.plan_detect(p.card, no_flip=ctx.constraints.no_flip)
        if spec.adapt.imgsz:
            plan.imgsz = spec.adapt.imgsz
        review_ctx = "\n".join([p.card_digest, context_digest(ctx), "PLAN " + plan.model_dump_json()])
        choices = {}
    review, source, note = agent.task(
        "adapt_review", review_ctx, choices=choices,
        fallback=lambda: {"notes": "Rule-based plan: IDs, constant, free-text and >60%-missing columns dropped; "
                                   "numeric median-imputed and scaled; categoricals one-hot (<=20 levels) "
                                   "or ordinal." if p.card["kind"] == "tabular" else
                                   "Rule-based plan: convert to YOLO layout, keep classes with boxes, "
                                   "create/clean the val split.",
                          "warnings": [], "extra_drop": []})
    p.adapt_plan = plan.model_dump()
    p.adapt_review = {**review, "source": source, "note": note}
    p.adapt_diff = None
    return save(p)


def apply_adapt(pid: str, plan: dict) -> Project:
    p = load(pid)
    spec = JobSpec.model_validate(p.spec)
    out = _pdir(pid) / "adapted"
    if plan.get("kind") == "tabular":
        tp = adapt_step.TabularPlan.model_validate(plan)
        diff = adapt_step.apply_tabular(tp, Path(p.card["path"]), out)
    else:
        dp = adapt_step.DetectPlan.model_validate(plan)
        diff = adapt_step.apply_detect(dp, Path(p.card["path"]), out)
        spec.adapt.imgsz, spec.adapt.class_map, spec.adapt.fliplr = dp.imgsz, diff["classes_after"], dp.fliplr
        spec.data.format = "yolo"
    spec.data.path = str(out)
    p.adapt_plan, p.adapt_diff, p.spec = plan, diff, spec.model_dump()
    unlock(p, 5)
    return save(p)


# ------------------------------------------------------------------ step 5-6: train / watch

def start_training(pid: str, spec_yaml: Optional[str] = None) -> Project:
    p = load(pid)
    if not p.adapt_diff:
        raise ValueError("apply the adaptation first")
    spec = JobSpec.from_yaml(spec_yaml) if spec_yaml else JobSpec.model_validate(p.spec)
    approved = JobSpec.model_validate(p.spec)
    spec.data = approved.data  # training always reads the adapted data, whatever the edited yaml says
    if spec.task != approved.task:
        raise ValueError(f"task cannot change at train time ({approved.task} -> {spec.task})")
    if cur := p.current_run:
        if runs.status(cur).get("alive"):
            raise ValueError(f"run {cur} is still active")
    rid = runs.start(pid, spec)
    p.spec = spec.model_dump()
    p.run_ids.append(rid)
    p.current_run = rid
    unlock(p, 6)
    return save(p)


def project_of(run_id: str) -> str:
    rec = memory.get(run_id)
    if not rec:
        raise FileNotFoundError(run_id)
    return rec["project_id"]


def adopt_run(pid: str, run_id: str) -> None:
    p = load(pid)
    if run_id not in p.run_ids:
        p.run_ids.append(run_id)
    p.current_run = run_id
    save(p)


def run_view(run_id: str) -> dict:
    snap = runs.snapshot(run_id)
    snap["verdict"] = read_json(runs.run_dir(run_id) / "verdict.json")
    if snap["verdict"]:
        snap["verdict"].pop("digest", None)
    return snap


def review(run_id: str) -> dict:
    return monitor_step.review(run_id, project_of(run_id))


def resolve_verdict(run_id: str, approve: bool) -> dict:
    pid = project_of(run_id)
    out = monitor_step.resolve_verdict(run_id, pid, approve)
    if new := out.get("new_run_id"):
        adopt_run(pid, new)
    return out


# ------------------------------------------------------------------ step 7-8: eval / report / loop

def evaluate(run_id: str) -> dict:
    out = eval_step.evaluate(run_id)
    p = load(project_of(run_id))
    unlock(p, 8)
    save(p)
    return out


def next_step(run_id: str) -> dict:
    pid = project_of(run_id)
    st = runs.status(run_id)
    diag = monitor_step.diagnose(run_id, pid) if st.get("state") == "failed" else None
    if diag:
        return {"action": "finish" if diag["failure_mode"] == "crash" else "retrain", "diagnosis": diag,
                "change": None, "evidence": diag["evidence"], "reason": diag["fix"], "source": diag["source"]}
    return monitor_step.propose_next(run_id, pid)


def start_next(run_id: str, change: dict) -> dict:
    pid = project_of(run_id)
    new = monitor_step.start_next(run_id, pid, change)
    adopt_run(pid, new)
    p = load(pid)
    p.unlocked = 6  # eval/report re-open once the new run is evaluated
    save(p)
    return {"new_run_id": new}


def build_report(run_id: str) -> str:
    return report_step.build(run_id, load(project_of(run_id)).model_dump())


def digest_for_chat(pid: str) -> str:
    """Everything the chat assistant should know about the project, compact."""
    p = load(pid)
    parts = [p.card_digest] if p.card_digest else ["no data scanned yet"]
    if p.context:
        ctx = Context.model_validate(p.context)
        parts.append(context_digest(ctx))
        if ctx.summary:
            parts.append("AGENT READING " + ctx.summary)
    if p.spec:
        parts.append("APPROVED SPEC\n" + yaml.safe_dump(p.spec, sort_keys=False)[:800])
    if p.adapt_diff:
        parts.append("ADAPT " + str({k: v for k, v in p.adapt_diff.items() if k != "path"})[:600])
    if p.current_run:
        d = runs.run_dir(p.current_run)
        rows = read_train_log(d)
        st = runs.status(p.current_run)
        parts.append(f"CURRENT RUN {p.current_run} state={st.get('state')} reason={st.get('reason')} "
                     f"epochs={len(rows)} last={rows[-1] if rows else '-'}")
        ev = read_json(d / "eval.json")
        if ev:
            parts.append(f"EVAL {ev.get('metrics')} target_hit={ev.get('target_hit')}")
    hist = memory.history_lines(pid)
    if hist:
        parts.append("HISTORY " + "; ".join(hist))
    return "\n".join(parts)


# ------------------------------------------------------------------ background watcher

class Watcher(threading.Thread):
    """Every few seconds: for active runs, ask the agent to review every MONITOR_EVERY epochs."""

    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True, name="loopback-watcher")
        self.interval = interval
        self.reviewed: dict[str, int] = {}
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                self.tick()
            except Exception as exc:  # never let the watcher die
                log.warning("watcher_error", error=repr(exc))

    def tick(self) -> None:
        for f in PROJECTS_DIR.glob("*/project.json"):
            p = Project.model_validate_json(f.read_text())
            rid = p.current_run
            if not rid:
                continue
            try:
                st = runs.status(rid)
            except FileNotFoundError:
                continue
            if not st.get("alive"):
                continue
            rows = read_train_log(runs.run_dir(rid))
            if not rows:
                continue
            epoch = rows[-1]["epoch"]
            v = read_json(runs.run_dir(rid) / "verdict.json") or {}
            if v.get("status") == "pending":
                continue
            if (epoch + 1) % MONITOR_EVERY == 0 and self.reviewed.get(rid, -1) < epoch:
                self.reviewed[rid] = epoch
                t0 = time.time()
                verdict = monitor_step.review(rid, p.id)
                log.info("auto_review", run=rid, epoch=epoch, verdict=verdict["verdict"],
                         source=verdict["source"], secs=round(time.time() - t0, 1))


def export_yaml(spec: dict) -> str:
    return JobSpec.model_validate(spec).to_yaml()


def refresh(p: Project) -> Project:
    """Eval unlocks once the current run has reached a terminal state."""
    if p.current_run and p.unlocked < 7:
        try:
            if runs.status(p.current_run).get("state") in runs.TERMINAL:
                unlock(p, 7)
                save(p)
        except FileNotFoundError:
            pass
    return p


def as_dict(p: Project) -> dict[str, Any]:
    d = refresh(p).model_dump()
    d["steps"] = STEPS
    return d
