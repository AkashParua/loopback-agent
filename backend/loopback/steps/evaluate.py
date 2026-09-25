"""Step 7 - Evaluate on the hold-out split.

Tabular: accuracy/F1 or RMSE/MAE/R2 on test; confusion matrix or residual plot.
Detection: Ultralytics val -> mAP50, mAP50-95, per-class P/R; a grid of the
images with the most missed (FN) and spurious (FP) boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .. import memory, runs  # noqa: E402
from ..jobspec import JobSpec  # noqa: E402
from ..train.metrics import all_metrics  # noqa: E402
from ..train.runctx import resolve_device, write_json  # noqa: E402


def _hit(spec: JobSpec, value) -> bool | None:
    if spec.target_value is None or value is None:
        return None
    return value >= spec.target_value if spec.higher_is_better else value <= spec.target_value


# ================================================================== tabular

def _predict_tabular(spec: JobSpec, d: Path, X: np.ndarray, data) -> np.ndarray:
    if spec.model in ("logreg", "ridge", "random_forest"):
        import joblib
        return joblib.load(d / "model.joblib").predict(X)
    if spec.model == "mlp":
        import torch

        from ..train.tabular import build_mlp
        meta = json.loads((d / "model_meta.json").read_text())
        net = build_mlp(meta["d_in"], meta["d_out"], meta["dropout"])
        net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        net.eval()
        with torch.no_grad():
            out = net(torch.tensor(X))
        return out.argmax(1).numpy() if data.task == "classify" else \
            out.squeeze(1).numpy() * meta["y_std"] + meta["y_mean"]
    if spec.model == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        m = TabNetClassifier() if data.task == "classify" else TabNetRegressor()
        m.load_model(str(d / "tabnet.zip"))
        p = m.predict(X)
        return p if data.task == "classify" else p.reshape(-1)
    raise ValueError(spec.model)


def evaluate_tabular(spec: JobSpec, d: Path) -> dict:
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report

    from ..train.tabular import TabularData, _pair
    data = TabularData(Path(spec.data.path))
    X, y = data.splits["test"]
    pred = _predict_tabular(spec, d, X, data)
    y_true, y_pred = _pair(data, y, pred)
    metrics = all_metrics(data.task, y_true, y_pred)
    out: dict = {"split": "test", "n": int(len(y)), "metrics": metrics, "images": []}
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=110)
    if data.task == "classify":
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        out["per_class"] = {k: {m: round(v, 4) for m, v in rep[k].items()} for k in data.classes if k in rep}
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion matrix (test)")
        name = "confusion_matrix.png"
    else:
        res = y_true - y_pred
        ax.scatter(y_pred, res, s=12, alpha=0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("residual (true - pred)")
        ax.set_title("Residuals (test)")
        out["residuals"] = {"mean": round(float(res.mean()), 4), "std": round(float(res.std()), 4),
                            "worst": [round(float(x), 3) for x in res[np.argsort(-np.abs(res))[:5]]]}
        name = "residuals.png"
    fig.tight_layout()
    fig.savefig(d / name)
    plt.close(fig)
    out["images"].append(name)
    return out


# ================================================================== detection

def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a[N,4] and b[M,4] in xyxy."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _gt(label: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [l.split() for l in label.read_text().splitlines() if l.strip()] if label.exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    a = np.array([[float(x) for x in r[:5]] for r in rows])
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), a[:, 0].astype(int)


def best_f1_conf(m) -> float:
    """Confidence threshold that maximises mean F1 on the val curve (a fixed 0.25 hides
    everything on an under-trained model)."""
    try:
        f1 = np.asarray(m.box.f1_curve).mean(0)
        return float(np.clip(np.asarray(m.box.px)[int(f1.argmax())], 0.01, 0.9))
    except (AttributeError, ValueError, IndexError):
        return 0.25


def error_samples(model, images: list[Path], names: dict, imgsz: int, device, out: Path,
                  conf: float = 0.25, max_images: int = 200, grid: int = 9) -> dict:
    per_image = []
    fn_by_class: dict[str, int] = {}
    fp_by_class: dict[str, int] = {}
    for res in model.predict(source=[str(p) for p in images[:max_images]], imgsz=imgsz, conf=conf,
                             device=device, verbose=False, stream=True):
        h, w = res.orig_shape
        gt, gcls = _gt(Path(res.path.replace("/images/", "/labels/")).with_suffix(".txt"), w, h)
        pb = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        pc = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros(0, dtype=int)
        iou = _iou(gt, pb)
        matched_g, matched_p = set(), set()
        for gi, pi in sorted(((i, j) for i in range(len(gt)) for j in range(len(pb))),
                             key=lambda t: -iou[t]):
            if iou[gi, pi] < 0.5 or gi in matched_g or pi in matched_p or gcls[gi] != pc[pi]:
                continue
            matched_g.add(gi)
            matched_p.add(pi)
        fn = [i for i in range(len(gt)) if i not in matched_g]
        fp = [j for j in range(len(pb)) if j not in matched_p]
        for i in fn:
            fn_by_class[names[gcls[i]]] = fn_by_class.get(names[gcls[i]], 0) + 1
        for j in fp:
            fp_by_class[names[pc[j]]] = fp_by_class.get(names[pc[j]], 0) + 1
        per_image.append((len(fn) + len(fp), res.path, gt[fn], pb[fp], gt[list(matched_g)]))

    worst = sorted([p for p in per_image if p[0] > 0], key=lambda t: -t[0])[:grid]
    tiles = []
    for _, path, fn_boxes, fp_boxes, tp_boxes in worst:
        im = Image.open(path).convert("RGB")
        dr = ImageDraw.Draw(im)
        lw = max(2, im.width // 250)
        for b in tp_boxes:
            dr.rectangle(b.tolist(), outline=(40, 170, 80), width=max(1, lw // 2))
        for b in fn_boxes:
            dr.rectangle(b.tolist(), outline=(230, 40, 40), width=lw)
        for b in fp_boxes:
            dr.rectangle(b.tolist(), outline=(255, 150, 0), width=lw)
        im.thumbnail((360, 360))
        tiles.append(im)
    name = None
    if tiles:
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        th = max(t.height for t in tiles)
        sheet = Image.new("RGB", (cols * 360, rows * th), (245, 245, 245))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 360 + (360 - t.width) // 2, (i // cols) * th + (th - t.height) // 2))
        name = "errors.png"
        sheet.save(out / name)
    return {"images_checked": len(per_image), "conf": round(conf, 3),
            "false_negatives": fn_by_class, "false_positives": fp_by_class,
            "grid": name, "legend": "red = missed ground truth (FN), orange = false positive, green = correct"}


def evaluate_detect(spec: JobSpec, d: Path) -> dict:
    from ultralytics import YOLO

    data_yaml = Path(spec.data.path) / "data.yaml"
    cfg = yaml.safe_load(data_yaml.read_text())
    split = "test" if cfg.get("test") else "val"
    metrics_json = json.loads((d / "metrics.json").read_text())
    model = YOLO(str(d / metrics_json["artifact"]))
    device = resolve_device(spec.train.device)
    device = 0 if device == "cuda" else device
    imgsz = spec.adapt.imgsz or 640
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=spec.train.batch, device=device,
                  project=str(d), name="eval", exist_ok=True, plots=False, verbose=False)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    base = Path(cfg["path"])
    img_dir = base / cfg[split]
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    errors = error_samples(model, images, names, imgsz, device, d, conf=best_f1_conf(m))
    out = {"split": split, "n": len(images),
           "metrics": {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                       "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)},
           "per_class": per_class, "errors": errors, "images": [errors["grid"]] if errors["grid"] else []}
    if split == "val":
        out["note"] = "no test split: evaluated on val, which also drove early stopping (optimistic)"
    return out


def evaluate(run_id: str) -> dict:
    d = runs.run_dir(run_id)
    spec = JobSpec.load(d / "job_spec.yaml")
    st = runs.status(run_id)
    if st.get("state") not in ("completed", "early_stopped", "stopped"):
        raise ValueError(f"run is {st.get('state')}; evaluate needs a finished run with weights")
    out = evaluate_detect(spec, d) if spec.task == "detect" else evaluate_tabular(spec, d)
    value = out["metrics"].get(spec.target_metric)
    out |= {"target_metric": spec.target_metric, "target_value": spec.target_value,
            "value": value, "target_hit": _hit(spec, value)}
    write_json(d / "eval.json", out)
    memory.update(run_id, final_metric=value)
    return out
