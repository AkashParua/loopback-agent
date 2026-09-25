"""Step 8 - Report. Numbers come from files (deterministic tables); the LLM writes
only the narrative (headline, rationale, curve paragraph, what to try next)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import memory, runs  # noqa: E402
from ..agent_client import agent  # noqa: E402
from ..jobspec import JobSpec  # noqa: E402
from ..logs import curve_facts, read_events, read_train_log  # noqa: E402
from ..train.runctx import read_json  # noqa: E402


def _f(v, nd=4) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def curve_plot(d: Path, rows: list[dict], metric: str) -> Optional[str]:
    if not rows:
        return None
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=110)
    tl = [r.get("train_loss") if isinstance(r.get("train_loss"), (int, float)) else None for r in rows]
    ax.plot(ep, tl, label="train_loss", color="#2a6fdb")
    if any(isinstance(r.get("val_loss"), (int, float)) for r in rows):
        ax.plot(ep, [r.get("val_loss") for r in rows], label="val_loss", color="#2a6fdb", ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(ep, [r.get("val_metric") for r in rows], label=f"val {metric}", color="#d9480f")
    ax2.set_ylabel(metric)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "curve.png")
    plt.close(fig)
    return "curve.png"


def build(run_id: str, project: dict) -> str:
    d = runs.run_dir(run_id)
    spec = JobSpec.load(d / "job_spec.yaml")
    rows = read_train_log(d)
    facts = curve_facts(rows, spec.higher_is_better)
    st = read_json(d / "metrics.json") or runs.status(run_id)
    ev = read_json(d / "eval.json") or {}
    hist = memory.history(project["id"])
    card = project.get("card") or {}
    adapt = project.get("adapt_diff") or {}
    ctx = project.get("context") or {}
    early = st.get("reason")
    changes = [e for e in read_events(d) if e.get("event") in ("change", "agent_verdict", "verdict_approved")]

    facts_text = "\n".join([
        f"DATA {project.get('card_digest', '')[:1500]}",
        f"GOAL {spec.target_metric} target={spec.target_value}",
        f"MODEL {spec.model} reason: {spec.reason or '-'}",
        f"ADAPT dropped={[x['column'] for x in adapt.get('dropped', [])]} "
        f"format={adapt.get('format', 'tabular')} split={adapt.get('split') or adapt.get('images')}",
        "CURVE " + ", ".join(f"{k}={_f(v)}" for k, v in facts.items()),
        f"STOP state={st.get('state')} reason={early or 'ran all epochs'}",
        f"EVAL {ev.get('split', '-')}: " + ", ".join(f"{k}={v}" for k, v in ev.get("metrics", {}).items()),
        f"TARGET HIT {ev.get('target_hit')}",
        "HISTORY " + "; ".join(memory.history_lines(project["id"])),
    ])

    def fallback() -> dict:
        v = ev.get("value")
        hit = ev.get("target_hit")
        head = (f"{spec.model} reached {spec.target_metric}={_f(v)} on the {ev.get('split', 'hold-out')} split"
                + ("" if hit is None else (" - target met." if hit else f" - below target {spec.target_value}.")))
        curve = (f"Trained {facts.get('epochs_logged', 0)} epochs; best val {spec.target_metric} "
                 f"{_f(facts.get('best_val_metric'))} at epoch {facts.get('best_epoch')}, last "
                 f"{_f(facts.get('last_val_metric'))}. Train loss {_f(facts.get('first_train_loss'))} -> "
                 f"{_f(facts.get('last_train_loss'))}." + (f" Stopped: {early}." if early else ""))
        return {"headline": head, "model_rationale": spec.reason or "chosen from the fixed zoo",
                "curve_paragraph": curve, "what_to_try_next": ["see the next-step proposal in the UI"]}

    nar, source, _ = agent.task("report", facts_text, fallback=fallback)
    plot = curve_plot(d, rows, spec.target_metric)

    md = [f"# Loopback report - `{run_id}`", "",
          f"**{nar['headline']}**", "",
          f"_Narrative by {'Qwen 3.5 4B (Ollama)' if source == 'llm' else 'rule-based fallback'}; "
          f"all numbers below are read from run files._", "",
          "## Data", ""]
    if card.get("kind") == "tabular":
        md += [f"- `{card.get('file')}`: {card.get('n_rows')} rows x {card.get('n_cols')} columns, "
               f"target `{ctx.get('target')}` ({ctx.get('task')})"]
    elif card:
        md += [f"- {card['kind'].upper()} detection: " + ", ".join(
            f"{k} {v['images']} images / {v['boxes']} boxes" for k, v in card.get("splits", {}).items()),
            f"- classes: {', '.join(c['name'] for c in card.get('classes', []) if c['count'])}"]
    md += [f"- warning: {w}" for w in card.get("warnings", [])[:6]]
    if ctx.get("notes"):
        md += [f"- user notes: {ctx['notes']}"]
    md += ["", "## Model", "", f"- **{spec.model}** ({spec.framework})", f"- {nar['model_rationale']}", "",
           "```yaml", spec.to_yaml().strip(), "```", "", "## Adaptations", ""]
    if adapt.get("dropped"):
        md += [f"- dropped `{x['column']}`: {x['reason']}" for x in adapt["dropped"]]
    if adapt.get("encoding"):
        md += [f"- {k}: {', '.join(v) if v else '-'}" for k, v in adapt["encoding"].items()]
    if adapt.get("format"):
        md += [f"- format: {adapt['format']}; classes {adapt.get('classes_before')} -> {adapt.get('classes_after')}",
               f"- images: {adapt.get('images')}; imgsz {adapt.get('imgsz')}; "
               f"removed val duplicates {adapt.get('removed_val_duplicates', 0)}"]
    if adapt.get("split"):
        md += [f"- split: {adapt['split']}"]
    md += ["", "## Training", "", nar["curve_paragraph"], "",
           "| epochs run | best epoch | best val | last val | stop reason |",
           "|---|---|---|---|---|",
           f"| {facts.get('epochs_logged', 0)} | {facts.get('best_epoch', '-')} | {_f(facts.get('best_val_metric'))} "
           f"| {_f(facts.get('last_val_metric'))} | {early or 'ran all epochs'} |", ""]
    if plot:
        md += [f"![training curve]({plot})", ""]
    if changes:
        md += ["Agent decisions during this run:", ""]
        md += [f"- {e.get('event')}: {e.get('detail')}" for e in changes]
        md += [""]
    md += ["## Final metrics", "", f"Split: **{ev.get('split', '-')}** (n={ev.get('n', '-')})", "",
           "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in ev.get("metrics", {}).items()]
    if ev.get("note"):
        md += ["", f"> {ev['note']}"]
    if ev.get("per_class"):
        keys = list(next(iter(ev["per_class"].values())).keys())
        md += ["", "| class | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        md += [f"| {c} | " + " | ".join(_f(v.get(k)) for k in keys) + " |" for c, v in ev["per_class"].items()]
    for img in ev.get("images", []):
        md += ["", f"![{img}]({img})"]
    if ev.get("errors"):
        e = ev["errors"]
        md += ["", f"Missed boxes by class: {e['false_negatives'] or 'none'}; "
                   f"false positives: {e['false_positives'] or 'none'}. {e['legend']}."]
    md += ["", "## Experiment history", "", "| run | model | change | status | best val | test |",
           "|---|---|---|---|---|---|"]
    md += [f"| {h['run_id']} | {h['model']} | {h['change'] or '-'} | {h['status']} | {_f(h['best_metric'])} "
           f"| {_f(h['final_metric'])} |" for h in hist]
    md += ["", "## What to try next", ""] + [f"- {x}" for x in nar["what_to_try_next"]]
    text = "\n".join(md) + "\n"
    (d / "report.md").write_text(text)
    return text
