"""Step 2 - Understand context: task type, target metric, domain notes, constraints."""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..agent_client import agent
from ..jobspec import Metric, Task
from .inspect_data import card_digest

DEFAULT_METRIC: dict[str, str] = {"classify": "f1_macro", "regress": "rmse", "detect": "mAP50"}
METRICS_FOR: dict[str, list[str]] = {
    "classify": ["f1_macro", "accuracy"],
    "regress": ["rmse", "mae", "r2"],
    "detect": ["mAP50", "mAP50-95"],
}


class Constraints(BaseModel):
    drop_columns: list[str] = Field(default_factory=list)
    max_epochs: Optional[int] = None
    no_flip: bool = False
    device: str = "auto"


class Context(BaseModel):
    task: Task
    target: Optional[str] = None
    target_metric: Metric
    target_value: Optional[float] = None
    notes: str = ""
    constraints: Constraints = Field(default_factory=Constraints)
    # filled by the agent
    summary: str = ""
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    source: Literal["llm", "rules", ""] = ""
    agent_note: Optional[str] = None


def infer_task(card: dict) -> str:
    if card["kind"] in ("yolo", "coco"):
        return "detect"
    ts = card.get("target_stats") or {}
    return "classify" if ts.get("kind") == "categorical" else "regress"


def default_target_value(card: dict, task: str, metric: str) -> Optional[float]:
    if metric in ("f1_macro", "accuracy"):
        return 0.85
    if metric == "r2":
        return 0.6
    if metric in ("mAP50",):
        return 0.8
    if metric == "mAP50-95":
        return 0.5
    ts = card.get("target_stats") or {}
    if metric in ("rmse", "mae") and ts.get("std"):
        # 25% better than always predicting the mean
        return round(0.75 * ts["std"] * (0.8 if metric == "mae" else 1.0), 4)
    return None


def defaults(card: dict) -> dict:
    task = infer_task(card)
    metric = DEFAULT_METRIC[task]
    return {"task": task, "target": card.get("target"), "target_metric": metric,
            "target_value": default_target_value(card, task, metric),
            "metrics": METRICS_FOR[task], "target_candidates": card.get("target_candidates", [])}


def context_digest(ctx: Context) -> str:
    c = ctx.constraints
    lines = [f"GOAL task={ctx.task} target={ctx.target or '-'} metric={ctx.target_metric} "
             f"target_value={ctx.target_value if ctx.target_value is not None else 'none'}"]
    cons = []
    if c.drop_columns:
        cons.append("do not use columns " + ", ".join(c.drop_columns))
    if c.max_epochs:
        cons.append(f"max {c.max_epochs} epochs")
    if c.no_flip:
        cons.append("no horizontal flip augmentation")
    if cons:
        lines.append("CONSTRAINTS " + "; ".join(cons))
    if ctx.notes.strip():
        lines.append("USER NOTES " + ctx.notes.strip()[:800])
    return "\n".join(lines)


def _rules_summary(card: dict, ctx: Context) -> dict:
    if card["kind"] == "tabular":
        s = (f"{card['n_rows']} rows x {card['n_cols']} columns; {ctx.task} on '{ctx.target}' "
             f"scored by {ctx.target_metric}.")
    else:
        n = sum(v["images"] for v in card["splits"].values())
        s = f"{n} images, {len(card['classes'])} classes ({card['kind'].upper()}); detection scored by {ctx.target_metric}."
    return {"summary": s, "task": ctx.task, "target_metric": ctx.target_metric,
            "risks": card.get("warnings", [])[:5], "questions": []}


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    out = set()
    for n in _NUM.findall(text):
        out.add(n)
        if "." in n:
            out.add(n.rstrip("0").rstrip("."))
    return out


def grounded(claim: str, source: str) -> bool:
    """True if every number quoted in `claim` appears in `source` (small models invent stats)."""
    return _numbers(claim) <= _numbers(source)


def understand(card: dict, ctx: Context) -> Context:
    prompt = card_digest(card) + "\n" + context_digest(ctx)

    def validate(out: dict) -> Optional[str]:
        if out.get("task") != ctx.task:
            return f"agent said task={out.get('task')} but data/user say {ctx.task}"
        return None

    out, source, note = agent.task("data_summary", prompt, fallback=lambda: _rules_summary(card, ctx),
                                   validate=validate)
    sentences = re.split(r"(?<=[.!?])\s+", out["summary"].strip())
    kept = [x for x in sentences if grounded(x, prompt)]
    risks = [r for r in out.get("risks", []) if grounded(r, prompt)]
    dropped = len(sentences) - len(kept) + len(out.get("risks", [])) - len(risks)
    if dropped and source == "llm":
        note = f"dropped {dropped} statement(s) quoting numbers not in the data card"
    ctx.summary = " ".join(kept) or _rules_summary(card, ctx)["summary"]
    ctx.risks = risks
    ctx.questions = out.get("questions", [])
    if out.get("target_metric") in METRICS_FOR[ctx.task] and out["target_metric"] != ctx.target_metric:
        ctx.risks.append(f"agent would score with {out['target_metric']} instead of {ctx.target_metric}")
    ctx.source = source
    ctx.agent_note = note
    return ctx
