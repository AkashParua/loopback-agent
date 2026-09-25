"""Step 4 - Adapt input / output.

plan_*()  -> an editable plan (pydantic) the user approves.
apply_*() -> materialise it under the project's `adapted/` folder and return a diff.

Tabular: drop columns, impute -> encode -> scale (sklearn ColumnTransformer), split.
Detection: COCO->YOLO conversion, class mapping (rename / merge / drop), val split,
de-duplicate val images that also appear in train, imgsz.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .inspect_data import (_file_md5, coco_layout, detect_kind, load_table, read_yolo_labels,
                           yolo_label_path, yolo_layout)

ONEHOT_MAX = 20


# ============================================================== tabular

class DropItem(BaseModel):
    column: str
    reason: str


class TabularPlan(BaseModel):
    kind: Literal["tabular"] = "tabular"
    target: str
    task: Literal["classify", "regress"]
    drop: list[DropItem] = Field(default_factory=list)
    numeric: list[str] = Field(default_factory=list)
    onehot: list[str] = Field(default_factory=list)
    ordinal: list[str] = Field(default_factory=list)
    drop_duplicate_rows: bool = True
    class_weight: Optional[Literal["balanced"]] = None
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42


def plan_tabular(card: dict, target: str, task: str, user_drop: list[str] | None = None) -> TabularPlan:
    drops: dict[str, str] = {}
    for c in card["columns"]:
        name = c["name"]
        if name == target:
            continue
        f = set(c["flags"])
        if "id_like" in f:
            drops[name] = "ID-like: every value unique"
        elif "constant" in f:
            drops[name] = "constant"
        elif "free_text" in f:
            drops[name] = "free text (out of scope)"
        elif c["null_pct"] > 60:
            drops[name] = f"{c['null_pct']}% missing"
        elif c["kind"] == "datetime":
            drops[name] = "datetime (not encoded in v1)"
    for s in card.get("leakage_suspects", []):
        drops.setdefault(s["column"], f"possible leakage: {s['why']}")
    for name in user_drop or []:
        drops.setdefault(name, "user constraint")

    numeric, onehot, ordinal = [], [], []
    for c in card["columns"]:
        name = c["name"]
        if name == target or name in drops:
            continue
        if c["kind"] in ("numeric", "bool"):
            numeric.append(name)
        elif c["n_unique"] <= ONEHOT_MAX:
            onehot.append(name)
        else:
            ordinal.append(name)
    ts = card.get("target_stats", {})
    return TabularPlan(
        target=target, task=task,
        drop=[DropItem(column=k, reason=v) for k, v in drops.items()],
        numeric=numeric, onehot=onehot, ordinal=ordinal,
        drop_duplicate_rows=card.get("duplicate_rows", 0) > 0,
        class_weight="balanced" if task == "classify" and ts.get("imbalance_ratio", 1) >= 3 else None,
    )


def prepare_frame(df: pd.DataFrame, plan: TabularPlan) -> pd.DataFrame:
    """Column typing used identically at fit, train and eval time."""
    df = df.copy()
    for c in plan.numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in plan.onehot + plan.ordinal:
        df[c] = df[c].where(df[c].isna(), df[c].astype(str)).astype(object)
        df[c] = df[c].where(df[c].notna(), np.nan)
    return df


def build_preprocessor(plan: TabularPlan) -> ColumnTransformer:
    parts = []
    if plan.numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), plan.numeric))
    if plan.onehot:
        parts.append(("onehot", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                          ("encode", OneHotEncoder(handle_unknown="ignore",
                                                                   sparse_output=False))]), plan.onehot))
    if plan.ordinal:
        parts.append(("ordinal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scale", StandardScaler())]), plan.ordinal))
    if not parts:
        raise ValueError("no feature columns left after drops")
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def apply_tabular(plan: TabularPlan, src: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = load_table(src)
    rows_before, cols_before = df.shape
    df = df[df[plan.target].notna()]
    dropped_target_na = rows_before - len(df)
    dups = 0
    if plan.drop_duplicate_rows:
        n = len(df)
        df = df.drop_duplicates()
        dups = n - len(df)
    features = plan.numeric + plan.onehot + plan.ordinal
    df = prepare_frame(df[features + [plan.target]], plan)

    y = df[plan.target]
    classes: Optional[list[str]] = None
    if plan.task == "classify":
        y = y.astype(str)
        classes = sorted(y.unique().tolist())
        rare = y.value_counts()
        strat = y if rare.min() >= 3 else None
    else:
        strat = None
    idx = np.arange(len(df))
    holdout = plan.val_size + plan.test_size
    tr, rest = train_test_split(idx, test_size=holdout, random_state=plan.seed,
                                stratify=strat.iloc[idx] if strat is not None else None)
    rel_test = plan.test_size / holdout
    strat_rest = strat.iloc[rest] if strat is not None and strat.iloc[rest].value_counts().min() >= 2 else None
    va, te = train_test_split(rest, test_size=rel_test, random_state=plan.seed, stratify=strat_rest)
    splits = {"train": df.iloc[tr], "val": df.iloc[va], "test": df.iloc[te]}
    for name, part in splits.items():
        part.to_parquet(out / f"{name}.parquet", index=False)

    pre = build_preprocessor(plan)
    pre.fit(splits["train"][features])
    joblib.dump(pre, out / "preprocessor.joblib")
    feat_names = [str(f) for f in pre.get_feature_names_out()]
    meta = {"target": plan.target, "task": plan.task, "features_in": features,
            "features_out": feat_names, "classes": classes, "class_weight": plan.class_weight}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    return {
        "rows": {"before": int(rows_before), "after": int(len(df)),
                 "dropped_target_missing": int(dropped_target_na), "dropped_duplicates": int(dups)},
        "columns": {"before": int(cols_before), "features_in": len(features), "features_out": len(feat_names)},
        "dropped": [d.model_dump() for d in plan.drop],
        "encoding": {"numeric (median impute + scale)": plan.numeric,
                     "one-hot (mode impute)": plan.onehot, "ordinal (mode impute + scale)": plan.ordinal},
        "classes": classes, "class_weight": plan.class_weight,
        "split": {k: int(len(v)) for k, v in splits.items()},
        "path": str(out),
    }


# ============================================================== detection: common representation

Box = tuple[str, float, float, float, float]  # class name, cx, cy, w, h (normalised)


class Sample(BaseModel):
    split: str
    image: str
    boxes: list[Box]


def load_yolo_samples(root: Path) -> tuple[list[str], list[Sample]]:
    names, splits, _ = yolo_layout(root)
    samples = []
    for split, imgs in splits.items():
        for img in imgs:
            raw, _ = read_yolo_labels(yolo_label_path(img))
            boxes = [(names[c] if c < len(names) else str(c), x, y, w, h) for c, x, y, w, h in raw]
            samples.append(Sample(split=split, image=str(img), boxes=boxes))
    return names, samples


def load_coco_samples(root: Path) -> tuple[list[str], list[Sample]]:
    names: dict[int, str] = {}
    samples = []
    for split, (js, img_dir) in coco_layout(root).items():
        data = json.loads(js.read_text())
        for c in data.get("categories", []):
            names.setdefault(int(c["id"]), str(c["name"]))
        by_img: dict[int, list[dict]] = {}
        for a in data.get("annotations", []):
            by_img.setdefault(a["image_id"], []).append(a)
        for im in data.get("images", []):
            p = img_dir / im["file_name"]
            if not p.exists():
                p = img_dir / Path(im["file_name"]).name
            if not p.exists():
                continue
            W, H = im.get("width"), im.get("height")
            if not W or not H:
                with Image.open(p) as pil:
                    W, H = pil.size
            boxes = []
            for a in by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                boxes.append((names.get(int(a["category_id"]), str(a["category_id"])),
                              (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
            samples.append(Sample(split="train" if split == "all" else split, image=str(p), boxes=boxes))
    return [names[k] for k in sorted(names)], samples


def load_samples(root: Path, fmt: str) -> tuple[list[str], list[Sample]]:
    return load_yolo_samples(root) if fmt == "yolo" else load_coco_samples(root)


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


# ============================================================== detection plan / apply

class DetectPlan(BaseModel):
    kind: Literal["detect"] = "detect"
    source_format: Literal["yolo", "coco"]
    convert: Optional[Literal["coco->yolo"]] = None
    rename: dict[str, str] = Field(default_factory=dict, description="old -> new; same new name merges")
    drop_classes: list[str] = Field(default_factory=list)
    class_map: list[str] = Field(default_factory=list, description="final class order (index = YOLO id)")
    imgsz: int = 640
    val_fraction: Optional[float] = Field(None, description="carve val from train if no val split")
    dedupe_val: bool = False
    fliplr: float = 0.5
    seed: int = 42


def suggest_imgsz(card: dict) -> int:
    s = card.get("image_size") or {}
    side = max(s.get("median_w", 640), s.get("median_h", 640))
    bs = card.get("box_sizes", {})
    small = bs.get("small", 0) / (sum(bs.values()) or 1)
    if small > 0.4 and side >= 960:
        return 960
    if side <= 320:
        return 320
    return 640


def plan_detect(card: dict, *, no_flip: bool = False) -> DetectPlan:
    names = [c["name"] for c in card["classes"]]
    empty = [c["name"] for c in card["classes"] if c["count"] == 0]
    return DetectPlan(
        source_format=card["kind"],
        convert="coco->yolo" if card["kind"] == "coco" else None,
        drop_classes=empty,
        class_map=[n for n in names if n not in empty],
        imgsz=suggest_imgsz(card),
        val_fraction=None if "val" in card["splits"] else 0.2,
        dedupe_val=card["overlap"]["by_hash"] > 0,
        fliplr=0.0 if no_flip else 0.5,
    )


def apply_detect(plan: DetectPlan, src: Path, out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names_before, samples = load_samples(src, plan.source_format)
    drop = set(plan.drop_classes)
    final = plan.class_map or sorted({plan.rename.get(n, n) for n in names_before} - drop)
    index = {n: i for i, n in enumerate(final)}

    rng = random.Random(plan.seed)
    splits_present = {s.split for s in samples}
    if plan.val_fraction and "val" not in splits_present:
        train = [s for s in samples if s.split == "train"]
        rng.shuffle(train)
        for s in train[: max(1, int(len(train) * plan.val_fraction))]:
            s.split = "val"

    removed_dupes = 0
    if plan.dedupe_val:
        train_hashes = {_file_md5(Path(s.image)) for s in samples if s.split == "train"}
        keep = []
        for s in samples:
            if s.split != "train" and _file_md5(Path(s.image)) in train_hashes:
                removed_dupes += 1
                continue
            keep.append(s)
        samples = keep

    counts: dict[str, Counter] = {}
    n_images: Counter = Counter()
    dropped_boxes = 0
    for i, s in enumerate(samples):
        img = Path(s.image)
        stem = f"{i:06d}_{img.stem}"
        _link(img, out / "images" / s.split / f"{stem}{img.suffix.lower()}")
        lines = []
        for name, x, y, w, h in s.boxes:
            name = plan.rename.get(name, name)
            if name in drop or name not in index:
                dropped_boxes += 1
                continue
            lines.append(f"{index[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            counts.setdefault(s.split, Counter())[name] += 1
        lbl = out / "labels" / s.split / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_images[s.split] += 1

    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "names": {i: n for i, n in enumerate(final)}}
    if n_images.get("test"):
        cfg["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    return {
        "format": f"{plan.source_format} -> yolo" if plan.convert else "yolo (normalised layout)",
        "classes_before": names_before, "classes_after": final,
        "renamed": plan.rename, "dropped_classes": sorted(drop), "dropped_boxes": dropped_boxes,
        "images": dict(n_images), "boxes": {k: dict(v) for k, v in counts.items()},
        "val_created_from_train": bool(plan.val_fraction and "val" not in splits_present),
        "removed_val_duplicates": removed_dupes, "imgsz": plan.imgsz, "fliplr": plan.fliplr,
        "data_yaml": str(out / "data.yaml"), "path": str(out),
    }


# ============================================================== YOLO -> COCO (export / test data)

def yolo_to_coco(src: Path, dst: Path) -> dict:
    """Write dst/images/<split>/* and dst/annotations/instances_<split>.json."""
    names, samples = load_yolo_samples(src)
    kind, _ = detect_kind(src)
    assert kind == "yolo", f"{src} is not a YOLO dataset"
    cats = [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(names)]
    cid = {n: i + 1 for i, n in enumerate(names)}
    out: dict[str, dict] = {}
    ann_id = 1
    for img_id, s in enumerate(samples, start=1):
        p = Path(s.image)
        split_dir = dst / "images" / s.split
        split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, split_dir / p.name)
        with Image.open(p) as pil:
            W, H = pil.size
        d = out.setdefault(s.split, {"images": [], "annotations": [], "categories": cats})
        d["images"].append({"id": img_id, "file_name": p.name, "width": W, "height": H})
        for name, x, y, w, h in s.boxes:
            bw, bh = w * W, h * H
            d["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cid[name],
                                     "bbox": [round((x - w / 2) * W, 2), round((y - h / 2) * H, 2),
                                              round(bw, 2), round(bh, 2)],
                                     "area": round(bw * bh, 2), "iscrowd": 0})
            ann_id += 1
    (dst / "annotations").mkdir(parents=True, exist_ok=True)
    for split, d in out.items():
        (dst / "annotations" / f"instances_{split}.json").write_text(json.dumps(d))
    return {split: len(d["images"]) for split, d in out.items()}
