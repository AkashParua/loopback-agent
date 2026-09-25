"""Step 3 - Suggest a model from the fixed zoo, as 1-3 job specs with a one-line reason.

Rules generate every valid candidate (with sizing heuristics); the LLM only
ranks them by id and may nudge a few bounded knobs. The schema's candidate_id
enum makes an out-of-zoo pick impossible.
"""

from __future__ import annotations

from typing import Optional

from .. import memory
from ..agent_client import agent
from ..jobspec import AdaptCfg, DataRef, JobSpec, TrainCfg
from ..zoo import models_for
from .context import Context, context_digest
from .inspect_data import card_digest


def _n_images(card: dict) -> int:
    return sum(v["images"] for v in card.get("splits", {}).values())


def candidates(card: dict, ctx: Context, data_path: str) -> list[JobSpec]:
    specs = []
    for entry in models_for(ctx.task):
        if ctx.task == "detect":
            n = _n_images(card)
            s = card.get("image_size") or {}
            side = max(s.get("median_w", 640), s.get("median_h", 640))
            imgsz = 320 if side <= 320 else 640
            train = TrainCfg(epochs=50 if n < 1000 else 100, batch=8 if n < 50 else 16,
                             patience=10 if n < 1000 else 20, device=ctx.constraints.device)
            adapt = AdaptCfg(imgsz=imgsz, class_map=[c["name"] for c in card["classes"] if c["count"]] or None,
                             fliplr=0.0 if ctx.constraints.no_flip else None)
            fmt = card["kind"]
        else:
            n = card["n_rows"]
            fmt = "tabular"
            adapt = AdaptCfg()
            if entry.framework == "sklearn":
                train = TrainCfg(epochs=1, batch=n, patience=1, device="cpu")
            elif entry.name == "tabnet":
                train = TrainCfg(epochs=100, batch=min(1024, max(64, n // 8)), patience=15, lr=2e-2,
                                 device=ctx.constraints.device)
            else:
                train = TrainCfg(epochs=100 if n < 2000 else 50,
                                 batch=32 if n < 1000 else 64 if n < 10_000 else 256,
                                 patience=10, lr=1e-3, dropout=0.2, weight_decay=1e-4,
                                 device=ctx.constraints.device)
        if ctx.constraints.max_epochs:
            train.epochs = min(train.epochs, ctx.constraints.max_epochs)
        specs.append(JobSpec(task=ctx.task, framework="sklearn" if entry.framework == "sklearn" else "pytorch",
                             model=entry.name, data=DataRef(format=fmt, path=data_path, target=ctx.target),
                             adapt=adapt, train=train, target_metric=ctx.target_metric,
                             target_value=ctx.target_value))
    return specs


def _rule_picks(card: dict, ctx: Context, cands: list[JobSpec], skip: set[int]) -> list[dict]:
    order: list[str]
    if ctx.task == "detect":
        n = _n_images(card)
        bs = card.get("box_sizes", {})
        small = bs.get("small", 0) / (sum(bs.values()) or 1)
        order = ["yolo11n", "yolov8n", "yolov8s"] if small > 0.3 else \
            ["yolov8n", "yolov8s", "yolo11n"] if n < 2000 else ["yolov8s", "yolo11n", "yolov8n"]
    else:
        n = card["n_rows"]
        base = "logreg" if ctx.task == "classify" else "ridge"
        order = [base, "random_forest", "mlp"] if n < 5000 else ["mlp", "random_forest", "tabnet", base]
    why = {
        "logreg": f"{card.get('n_rows')} rows: a linear baseline sets the bar in seconds",
        "ridge": f"{card.get('n_rows')} rows: a linear baseline sets the bar in seconds",
        "random_forest": "captures non-linear effects with no tuning",
        "mlp": "PyTorch MLP with early stopping; room to beat the baselines",
        "tabnet": "attention-based tabular net for larger data",
        "yolov8n": f"{_n_images(card)} images: the nano model trains fast and resists overfitting",
        "yolov8s": "more capacity if the nano model underfits",
        "yolo11n": "newer nano model, better on small objects",
    }
    picks = []
    for name in order:
        for i, s in enumerate(cands):
            if s.model == name and i not in skip:
                picks.append({"candidate_id": i, "reason": why.get(name, ""), "overrides": {}})
    return picks[:3]


def _cand_line(i: int, s: JobSpec) -> str:
    t = s.train
    knobs = f"epochs={t.epochs} batch={t.batch}" + (f" lr={t.lr}" if t.lr else "")
    if s.adapt.imgsz:
        knobs += f" imgsz={s.adapt.imgsz}"
    return f"{i}: {s.model} ({s.framework}) {knobs}"


def suggest(card: dict, ctx: Context, data_path: str, project_id: str, feedback: str = "") -> dict:
    cands = candidates(card, ctx, data_path)
    tried = memory.tried_fingerprints(project_id)
    failed = {i for i, s in enumerate(cands)
              if (r := tried.get(s.fingerprint())) and r["status"] in ("failed", "early_stopped")}
    allowed = [i for i in range(len(cands)) if i not in failed] or list(range(len(cands)))

    prompt = "\n".join([
        card_digest(card), context_digest(ctx),
        "CANDIDATES", *(_cand_line(i, cands[i]) for i in allowed),
        "HISTORY " + ("; ".join(memory.history_lines(project_id)) or "none"),
        *([f"USER FEEDBACK {feedback}"] if feedback else []),
    ])
    out, source, note = agent.task(
        "suggest", prompt, choices={"candidate_id": allowed},
        fallback=lambda: {"picks": _rule_picks(card, ctx, cands, failed)},
    )
    specs, seen = [], set()
    for p in out["picks"]:
        i = p["candidate_id"]
        if i in seen or i not in allowed:
            continue
        seen.add(i)
        spec = cands[i].model_copy(deep=True)
        ov = {k: v for k, v in (p.get("overrides") or {}).items() if v is not None}
        if spec.framework == "sklearn":
            ov = {}
        for k in ("epochs", "batch", "lr"):
            if k in ov:
                setattr(spec.train, k, ov[k])
        if "imgsz" in ov and spec.task == "detect":
            spec.adapt.imgsz = int(round(ov["imgsz"] / 32) * 32)
        if ctx.constraints.max_epochs:
            spec.train.epochs = min(spec.train.epochs, ctx.constraints.max_epochs)
        spec.reason = p["reason"]
        prev: Optional[dict] = tried.get(spec.fingerprint())
        specs.append({"candidate_id": i, "spec": spec.model_dump(), "yaml": spec.to_yaml(),
                      "reason": p["reason"], "overrides": ov,
                      "already_tried": prev["run_id"] if prev else None})
    return {"suggestions": specs, "source": source, "note": note,
            "all_candidates": [_cand_line(i, s) for i, s in enumerate(cands)]}
