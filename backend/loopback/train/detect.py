"""Detection training through Ultralytics YOLO (PyTorch under the hood). No custom heads."""

from __future__ import annotations

import math
from pathlib import Path

from ..config import WEIGHTS_DIR
from .runctx import RunContext, resolve_device

METRIC_KEYS = {"mAP50": "metrics/mAP50(B)", "mAP50-95": "metrics/mAP50-95(B)"}


def weights_path(model: str) -> str:
    """Pretrained weights cached in the workspace volume (downloaded once)."""
    return str(WEIGHTS_DIR / f"{model}.pt")


def _train_losses(tloss) -> dict[str, float]:
    """Ultralytics >= 8.4 keeps a {name: value} dict; older versions a [box, cls, dfl] tensor."""
    if tloss is None:
        return {}
    if isinstance(tloss, dict):
        return {str(k): float(v) for k, v in tloss.items()}
    return {k: float(v) for k, v in zip(("box_loss", "cls_loss", "dfl_loss"), tloss)}


def train_detect(ctx: RunContext) -> dict:
    from ultralytics import YOLO

    spec, cfg = ctx.spec, ctx.spec.train
    data_yaml = Path(spec.data.path) / "data.yaml"
    model = YOLO(weights_path(spec.model))
    best = {"epoch": None, "value": None}
    state = {"final": False, "final_metrics": {}}
    key = METRIC_KEYS[spec.target_metric]

    def on_train_start(trainer):
        # final_eval() re-validates best.pt and re-fires on_fit_epoch_end with epoch+1;
        # flag it so it is recorded as the best-weights score, not a fake epoch row.
        orig = trainer.final_eval

        def final_eval():
            state["final"] = True
            return orig()
        trainer.final_eval = final_eval

    def on_fit_epoch_end(trainer):
        m = trainer.metrics or {}
        if state["final"]:
            state["final_metrics"] = {k.split("/")[-1].replace("(B)", ""): float(v) for k, v in m.items()
                                      if k.startswith("metrics/")}
            return
        tl = _train_losses(trainer.tloss)
        vm = m.get(key)
        val_loss = sum(float(m.get(k, 0.0)) for k in ("val/box_loss", "val/cls_loss", "val/dfl_loss"))
        if vm is not None and math.isfinite(vm) and (best["value"] is None or vm > best["value"]):
            best.update(epoch=trainer.epoch, value=float(vm))
        stop = ctx.epoch_end(
            trainer.epoch, train_loss=sum(tl.values()) if tl else None, val_loss=val_loss or None,
            val_metric=vm, lr=next(iter(trainer.lr.values()), None) if trainer.lr else None,
            box_loss=tl.get("box_loss"), cls_loss=tl.get("cls_loss"),
            **{"mAP50": m.get("metrics/mAP50(B)"), "mAP50-95": m.get("metrics/mAP50-95(B)"),
               "precision": m.get("metrics/precision(B)"), "recall": m.get("metrics/recall(B)")})
        if stop:
            trainer.stop = True

    def on_train_batch_end(trainer):  # lets "stop" interrupt a long epoch
        if ctx._action() == "stop":
            ctx.request_stop("stopped", "stopped by user")
            trainer.stop = True

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    device = resolve_device(cfg.device)
    args = dict(data=str(data_yaml), epochs=cfg.epochs, batch=cfg.batch, imgsz=spec.adapt.imgsz or 640,
                patience=cfg.patience, device=0 if device == "cuda" else device,
                project=str(ctx.dir), name="yolo", exist_ok=True, workers=2, seed=0,
                verbose=False, plots=True, amp=device != "cpu")
    if cfg.lr:
        args["lr0"] = cfg.lr
    if spec.adapt.fliplr is not None:
        args["fliplr"] = spec.adapt.fliplr
    ctx.events.info("fit_start", model=spec.model, device=str(args["device"]), imgsz=args["imgsz"])
    model.train(**args)
    ran = ctx.epoch + 1
    weights = ctx.dir / "yolo" / "weights" / "best.pt"
    if not weights.exists():
        weights = ctx.dir / "yolo" / "weights" / "last.pt"
    return ctx.finish({"best_val_metric": best["value"], "best_epoch": best["epoch"],
                       "best_weights_val": state["final_metrics"],
                       "artifact": str(weights.relative_to(ctx.dir))},
                      patience_stop=ran < cfg.epochs and ctx.stop_reason is None)
