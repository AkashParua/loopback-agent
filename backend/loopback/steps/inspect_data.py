"""Step 1 - Look at data: schema, missing values, class balance, image sizes, split leakage.

Everything here is deterministic. The output is a `card` dict (facts +
warnings) and `card_digest(card)` renders it as short text for the LLM.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from PIL import Image

TABULAR_EXT = {".csv", ".parquet", ".pq"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TARGET_HINTS = ("target", "label", "class", "y", "outcome", "survived", "price", "churn", "default")
MAX_IMAGES_SCANNED = 5000
MAX_ISSUES = 10


# ============================================================== detection of kind

def extract_zip(path: Path) -> Path:
    out = path.with_suffix("")
    if not out.exists():
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
    # unwrap a single top-level folder (dataset.zip -> dataset/dataset/...)
    kids = [p for p in out.iterdir() if not p.name.startswith((".", "__MACOSX"))]
    return kids[0] if len(kids) == 1 and kids[0].is_dir() else out


def _is_coco_json(p: Path) -> bool:
    try:
        if p.stat().st_size < 2_000_000:
            data = json.loads(p.read_text())
            return isinstance(data, dict) and "images" in data and ("annotations" in data or "categories" in data)
        with open(p) as fh:  # large file: COCO puts "images" near the top
            return '"images"' in fh.read(1 << 16)
    except (OSError, ValueError):
        return False


def detect_kind(path: Path) -> tuple[str, Path]:
    """Returns (kind, resolved_path). kind in tabular | yolo | coco."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        path = extract_zip(path)
    if path.is_file():
        if path.suffix.lower() in TABULAR_EXT:
            return "tabular", path
        raise ValueError(f"unsupported file type {path.suffix}; expected .csv/.parquet or a YOLO/COCO folder")
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if any(p.name in ("data.yaml", "data.yml") or (p.suffix in (".yaml", ".yml") and _yaml_has_names(p))
           for p in files if p.suffix in (".yaml", ".yml")):
        return "yolo", path
    if any(p.suffix == ".json" and _is_coco_json(p) for p in files):
        return "coco", path
    if any(p.parent.name == "labels" or "labels" in p.parts for p in files if p.suffix == ".txt"):
        return "yolo", path
    tabs = [p for p in files if p.suffix.lower() in TABULAR_EXT]
    if len(tabs) >= 1:
        return "tabular", sorted(tabs, key=lambda p: -p.stat().st_size)[0]
    raise ValueError("could not detect data type: need .csv/.parquet, a YOLO folder (data.yaml + labels/), "
                     "or a COCO folder (images/ + annotations json)")


def _yaml_has_names(p: Path) -> bool:
    try:
        return "names" in (yaml.safe_load(p.read_text()) or {})
    except Exception:
        return False


def inspect(path: str | Path, target: Optional[str] = None) -> dict:
    kind, resolved = detect_kind(Path(path))
    if kind == "tabular":
        card = inspect_tabular(resolved, target)
    elif kind == "yolo":
        card = inspect_yolo(resolved)
    else:
        card = inspect_coco(resolved)
    card["path"] = str(resolved)
    return card


# ============================================================== tabular

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _col_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nonnull = s.dropna().astype(str)
    if len(nonnull) and nonnull.str.len().mean() > 30 and nonnull.nunique() > 0.5 * len(nonnull):
        return "text"
    return "categorical"


def is_categorical_target(s: pd.Series) -> bool:
    s = s.dropna()
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    n_unique = s.nunique()
    integral = bool(np.all(np.equal(np.mod(s.to_numpy(dtype=float), 1), 0)))
    return integral and n_unique <= max(2, min(20, int(0.05 * len(s))))


def guess_target(df: pd.DataFrame, columns: list[dict]) -> Optional[str]:
    usable = {c["name"] for c in columns if not c["flags"] or c["flags"] == ["high_null"]}
    for hint in TARGET_HINTS:
        for c in df.columns:
            if c.lower() == hint and c in usable:
                return c
    last = df.columns[-1]
    return last if last in usable else None


def target_stats(s: pd.Series) -> dict:
    s = s.dropna()
    if is_categorical_target(s):
        counts = s.astype(str).value_counts()
        top = counts.head(20)
        ratio = float(counts.iloc[0] / counts.iloc[-1]) if len(counts) > 1 else float("inf")
        return {"kind": "categorical", "n_classes": int(len(counts)),
                "counts": {k: int(v) for k, v in top.items()},
                "imbalance_ratio": round(ratio, 2)}
    d = s.describe()
    return {"kind": "numeric", "mean": float(d["mean"]), "std": float(d["std"]),
            "min": float(d["min"]), "max": float(d["max"]), "skew": round(float(s.skew()), 3)}


def leakage_suspects(df: pd.DataFrame, target: str, columns: list[dict]) -> list[dict]:
    """Columns that (almost) determine the target on their own."""
    y = df[target]
    out = []
    cat_target = is_categorical_target(y)
    for c in columns:
        name = c["name"]
        if name == target or "id_like" in c["flags"] or "constant" in c["flags"]:
            continue
        x = df[name]
        mask = x.notna() & y.notna()
        if mask.sum() < 20:
            continue
        if c["kind"] == "numeric" and not cat_target:
            corr = abs(np.corrcoef(x[mask].astype(float), y[mask].astype(float))[0, 1])
            if np.isfinite(corr) and corr > 0.98:
                out.append({"column": name, "why": f"|corr| with target = {corr:.3f}"})
        elif cat_target and x[mask].nunique() < 0.5 * mask.sum():
            purity = df[mask].groupby(name, observed=True)[target].agg(
                lambda v: v.value_counts(normalize=True).iloc[0])
            sizes = x[mask].value_counts()
            weighted = float((purity * sizes.reindex(purity.index)).sum() / sizes.sum())
            if weighted > 0.99 and x[mask].nunique() > 1:
                out.append({"column": name, "why": f"predicts target with {weighted:.1%} purity"})
    return out


def inspect_tabular(path: Path, target: Optional[str] = None) -> dict:
    df = load_table(path)
    n = len(df)
    columns = []
    for name in df.columns:
        s = df[name]
        kind = _col_kind(s)
        n_unique = int(s.nunique(dropna=True))
        null_pct = round(100 * float(s.isna().mean()), 1)
        flags = []
        if n_unique <= 1:
            flags.append("constant")
        if n_unique == n and kind in ("numeric", "categorical", "text") and (
                kind != "numeric" or pd.api.types.is_integer_dtype(s)):
            flags.append("id_like")
        if null_pct > 40:
            flags.append("high_null")
        if kind == "categorical" and n_unique > 50 and "id_like" not in flags:
            flags.append("high_cardinality")
        if kind == "text":
            flags.append("free_text")
        info = {"name": str(name), "dtype": str(s.dtype), "kind": kind, "n_unique": n_unique,
                "null_pct": null_pct, "flags": flags,
                "sample": [str(v)[:24] for v in s.dropna().unique()[:3]]}
        if kind == "numeric":
            info |= {"min": _num(s.min()), "max": _num(s.max()), "mean": _num(s.mean())}
        columns.append(info)

    target = target if target in df.columns else guess_target(df, columns)
    card: dict[str, Any] = {
        "kind": "tabular", "file": path.name, "n_rows": n, "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()), "columns": columns,
        "target": target, "target_candidates": _target_candidates(df, columns),
    }
    warnings = []
    for c in columns:
        for f in c["flags"]:
            msg = {"constant": "is constant", "id_like": "looks like an ID (all values unique)",
                   "high_null": f"has {c['null_pct']}% missing values",
                   "high_cardinality": f"has {c['n_unique']} categories",
                   "free_text": "is free text"}[f]
            warnings.append(f"{c['name']} {msg}")
    if card["duplicate_rows"]:
        warnings.append(f"{card['duplicate_rows']} duplicate rows ({100 * card['duplicate_rows'] / n:.1f}%)")
    if target:
        card["target_stats"] = target_stats(df[target])
        ts = card["target_stats"]
        if df[target].isna().any():
            warnings.append(f"target {target} has {int(df[target].isna().sum())} missing values (rows will be dropped)")
        if ts["kind"] == "categorical" and ts["imbalance_ratio"] >= 3:
            warnings.append(f"target is imbalanced: majority/minority = {ts['imbalance_ratio']}")
        card["leakage_suspects"] = leakage_suspects(df, target, columns)
        warnings += [f"possible leakage: {s['column']} {s['why']}" for s in card["leakage_suspects"]]
    if n < 200:
        warnings.append(f"only {n} rows: expect high variance in metrics")
    card["warnings"] = warnings
    return card


def _target_candidates(df: pd.DataFrame, columns: list[dict]) -> list[str]:
    ok = [c["name"] for c in columns if not ({"id_like", "constant", "free_text"} & set(c["flags"]))]
    hinted = [c for c in ok if c.lower() in TARGET_HINTS]
    return hinted + [c for c in reversed(ok) if c not in hinted]


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ============================================================== detection shared

def _file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _size_bucket(area_px: float) -> str:
    # COCO convention: small < 32^2, medium < 96^2, large otherwise
    return "small" if area_px < 32 ** 2 else "medium" if area_px < 96 ** 2 else "large"


def _overlap(split_files: dict[str, list[Path]]) -> dict:
    names = list(split_files)
    by_name, by_hash = 0, 0
    examples = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = {p.stem for p in split_files[a]}
            common = sa & {p.stem for p in split_files[b]}
            by_name += len(common)
            ha = {_file_md5(p): p.name for p in split_files[a][:MAX_IMAGES_SCANNED]}
            for p in split_files[b][:MAX_IMAGES_SCANNED]:
                if (h := _file_md5(p)) in ha:
                    by_hash += 1
                    if len(examples) < 5:
                        examples.append(f"{a}/{ha[h]} == {b}/{p.name}")
    return {"by_name": by_name, "by_hash": by_hash, "examples": examples}


def _image_size_stats(sizes: list[tuple[int, int]]) -> dict:
    if not sizes:
        return {}
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])
    return {"median_w": int(np.median(w)), "median_h": int(np.median(h)),
            "min": [int(w.min()), int(h.min())], "max": [int(w.max()), int(h.max())],
            "n_distinct": len(set(sizes))}


def _detect_warnings(card: dict) -> list[str]:
    w = []
    counts = {c["name"]: c["count"] for c in card["classes"]}
    if counts:
        hi, lo = max(counts.values()), min(counts.values())
        if lo == 0:
            zero = [k for k, v in counts.items() if v == 0]
            more = f" (+{len(zero) - 8} more)" if len(zero) > 8 else ""
            w.append(f"{len(zero)} classes with zero boxes: " + ", ".join(zero[:8]) + more)
        elif hi / lo >= 5:
            w.append(f"class imbalance: most/least boxes = {hi}/{lo} = {hi / lo:.1f}")
    bs = card["box_sizes"]
    tot = sum(bs.values()) or 1
    if bs.get("small", 0) / tot > 0.4:
        w.append(f"{100 * bs['small'] / tot:.0f}% of boxes are small (<32px): consider larger imgsz")
    for split, s in card["splits"].items():
        if s["images"] and s["empty_images"] / s["images"] > 0.3:
            w.append(f"{split}: {s['empty_images']}/{s['images']} images have no boxes")
    if "val" not in card["splits"]:
        w.append("no validation split: adapt step will create one")
    ov = card["overlap"]
    if ov["by_hash"]:
        w.append(f"split leakage: {ov['by_hash']} identical images across splits")
    elif ov["by_name"]:
        w.append(f"{ov['by_name']} file names repeat across splits (content differs)")
    if card["issues"]:
        w.append(f"{len(card['issues'])}+ annotation problems, e.g. {card['issues'][0]}")
    total = sum(s["images"] for s in card["splits"].values())
    if total < 100:
        w.append(f"only {total} images: expect noisy mAP")
    return w


# ============================================================== YOLO

def _find_yaml(root: Path) -> Optional[Path]:
    for name in ("data.yaml", "data.yml"):
        hits = sorted(root.rglob(name), key=lambda p: len(p.parts))
        if hits:
            return hits[0]
    for p in sorted(root.rglob("*.y*ml"), key=lambda p: len(p.parts)):
        if _yaml_has_names(p):
            return p
    return None


def _yaml_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names or []]


def _list_images(spec: Any, base: Path) -> list[Path]:
    """A YOLO split entry may be a dir, a txt list file, or a list of those."""
    if spec is None:
        return []
    if isinstance(spec, list):
        return [p for s in spec for p in _list_images(s, base)]
    p = Path(spec)
    p = p if p.is_absolute() else (base / p)
    if p.is_dir():
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXT)
    if p.suffix == ".txt" and p.exists():
        out = []
        for line in p.read_text().split():
            q = Path(line)
            out.append(q if q.is_absolute() else (p.parent / q))
        return out
    return []


def yolo_label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):  # replace the last 'images' dir with 'labels'
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def yolo_layout(root: Path) -> tuple[list[str], dict[str, list[Path]], Optional[Path]]:
    """Returns (class names, {split: [image paths]}, yaml path)."""
    ypath = _find_yaml(root)
    names: list[str] = []
    splits: dict[str, list[Path]] = {}
    if ypath:
        cfg = yaml.safe_load(ypath.read_text()) or {}
        names = _yaml_names(cfg.get("names"))
        base = Path(cfg["path"]) if cfg.get("path") else ypath.parent
        if not base.is_absolute():
            base = (ypath.parent / base).resolve()
        if not base.exists():  # yaml written for another machine: fall back to its folder
            base = ypath.parent
        for split in ("train", "val", "test"):
            imgs = _list_images(cfg.get(split), base)
            if imgs:
                splits[split] = imgs
    if not splits:  # conventional layout without a usable yaml
        img_root = next((p for p in [root / "images", *root.rglob("images")] if p.is_dir()), None)
        if img_root:
            subs = [d for d in img_root.iterdir() if d.is_dir()]
            if subs:
                for d in subs:
                    key = {"valid": "val", "validation": "val"}.get(d.name, d.name)
                    splits[key] = sorted(q for q in d.rglob("*") if q.suffix.lower() in IMAGE_EXT)
            else:
                splits["train"] = sorted(q for q in img_root.iterdir() if q.suffix.lower() in IMAGE_EXT)
    if not names:
        cls_txt = next(root.rglob("classes.txt"), None)
        if cls_txt:
            names = [l.strip() for l in cls_txt.read_text().splitlines() if l.strip()]
    return names, splits, ypath


def read_yolo_labels(lbl: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes, problems = [], []
    if not lbl.exists():
        return boxes, problems
    for i, line in enumerate(lbl.read_text().splitlines()):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            problems.append(f"{lbl.name}:{i + 1} has {len(parts)} fields")
            continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5]) if len(parts) == 5 else _poly_to_box(parts[1:])
        except ValueError:
            problems.append(f"{lbl.name}:{i + 1} not numeric")
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1.001 and 0 < h <= 1.001):
            problems.append(f"{lbl.name}:{i + 1} box out of [0,1]")
            continue
        boxes.append((c, x, y, w, h))
    return boxes, problems


def _poly_to_box(vals: list[str]) -> tuple[float, float, float, float]:
    xs = [float(v) for v in vals[0::2]]
    ys = [float(v) for v in vals[1::2]]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)


def inspect_yolo(root: Path) -> dict:
    names, splits, ypath = yolo_layout(root)
    class_counts: Counter[int] = Counter()
    box_sizes: Counter[str] = Counter()
    sizes: list[tuple[int, int]] = []
    issues: list[str] = []
    split_info = {}
    for split, imgs in splits.items():
        empty = missing = 0
        n_boxes = 0
        for k, img in enumerate(imgs):
            lbl = yolo_label_path(img)
            if not lbl.exists():
                missing += 1
            boxes, probs = read_yolo_labels(lbl)
            issues += probs[: MAX_ISSUES - len(issues)] if len(issues) < MAX_ISSUES else []
            if not boxes:
                empty += 1
            n_boxes += len(boxes)
            wh = None
            if k < MAX_IMAGES_SCANNED:
                try:
                    with Image.open(img) as im:
                        wh = im.size
                    sizes.append(wh)
                except OSError:
                    issues.append(f"unreadable image {img.name}") if len(issues) < MAX_ISSUES else None
            for c, _, _, w, h in boxes:
                class_counts[c] += 1
                if wh:
                    box_sizes[_size_bucket(w * wh[0] * h * wh[1])] += 1
                if names and c >= len(names) and len(issues) < MAX_ISSUES:
                    issues.append(f"{lbl.name}: class id {c} >= {len(names)} names")
        split_info[split] = {"images": len(imgs), "boxes": n_boxes, "empty_images": empty,
                             "missing_label_files": missing}
    if not names:
        names = [str(i) for i in range(max(class_counts, default=-1) + 1)]
    card = {
        "kind": "yolo", "yaml": str(ypath) if ypath else None,
        "classes": [{"id": i, "name": n, "count": int(class_counts.get(i, 0))} for i, n in enumerate(names)],
        "splits": split_info, "box_sizes": dict(box_sizes), "image_size": _image_size_stats(sizes),
        "overlap": _overlap(splits), "issues": issues,
    }
    card["warnings"] = _detect_warnings(card)
    return card


# ============================================================== COCO

def _split_of(name: str) -> str:
    n = name.lower()
    for key, split in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test")):
        if key in n:
            return split
    return "all"


def coco_layout(root: Path) -> dict[str, tuple[Path, Path]]:
    """{split: (annotation json, image dir)}"""
    out = {}
    for js in sorted(root.rglob("*.json")):
        if not _is_coco_json(js):
            continue
        split = _split_of(js.stem)
        if split in out:
            continue
        # image dir: images/<split>, <split>/, images/, or the json's folder
        cands = [root / "images" / split, root / split, js.parent / "images" / split,
                 js.parent / "images", root / "images", js.parent.parent / "images" / split,
                 js.parent.parent / split, js.parent]
        img_dir = next((c for c in cands if c.is_dir()), js.parent)
        out[split] = (js, img_dir)
    return out


def inspect_coco(root: Path) -> dict:
    layout = coco_layout(root)
    cats: dict[int, str] = {}
    class_counts: Counter[int] = Counter()
    box_sizes: Counter[str] = Counter()
    sizes: list[tuple[int, int]] = []
    issues: list[str] = []
    split_info = {}
    split_files: dict[str, list[Path]] = {}
    for split, (js, img_dir) in layout.items():
        data = json.loads(js.read_text())
        for c in data.get("categories", []):
            cats.setdefault(int(c["id"]), str(c["name"]))
        per_img: defaultdict[int, int] = defaultdict(int)
        crowd = 0
        for a in data.get("annotations", []):
            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                if len(issues) < MAX_ISSUES:
                    issues.append(f"{js.name}: annotation {a.get('id')} has empty bbox")
                continue
            crowd += int(a.get("iscrowd", 0))
            per_img[a["image_id"]] += 1
            class_counts[int(a["category_id"])] += 1
            box_sizes[_size_bucket(a.get("area") or w * h)] += 1
        images = data.get("images", [])
        missing = 0
        files = []
        for im in images:
            sizes.append((int(im.get("width", 0)), int(im.get("height", 0))))
            p = img_dir / im["file_name"]
            if not p.exists():
                p = img_dir / Path(im["file_name"]).name
            if p.exists():
                files.append(p)
            else:
                missing += 1
        if missing and len(issues) < MAX_ISSUES:
            issues.append(f"{split}: {missing} image files referenced in {js.name} not found in {img_dir.name}/")
        split_files[split] = files
        split_info[split] = {"images": len(images), "boxes": int(sum(per_img.values())),
                             "empty_images": sum(1 for im in images if per_img.get(im["id"], 0) == 0),
                             "missing_image_files": missing, "iscrowd": crowd,
                             "annotation_file": str(js.relative_to(root))}
    card = {
        "kind": "coco",
        "classes": [{"id": cid, "name": n, "count": int(class_counts.get(cid, 0))} for cid, n in sorted(cats.items())],
        "splits": split_info, "box_sizes": dict(box_sizes),
        "image_size": _image_size_stats([s for s in sizes if s[0] and s[1]]),
        "overlap": _overlap(split_files), "issues": issues,
    }
    card["warnings"] = _detect_warnings(card)
    return card


# ============================================================== digest for the LLM

def card_digest(card: dict, max_cols: int = 30) -> str:
    if card["kind"] == "tabular":
        lines = [f"DATA tabular file={card['file']} rows={card['n_rows']} cols={card['n_cols']} "
                 f"duplicate_rows={card['duplicate_rows']}"]
        if card.get("target"):
            ts = card.get("target_stats", {})
            if ts.get("kind") == "categorical":
                total = sum(ts["counts"].values())
                dist = ", ".join(f"{k}={v} ({100 * v / total:.1f}%)" for k, v in list(ts["counts"].items())[:8])
                lines.append(f"TARGET {card['target']} categorical n_classes={ts['n_classes']}: {dist}")
            elif ts:
                lines.append(f"TARGET {card['target']} numeric mean={ts['mean']:.4g} std={ts['std']:.4g} "
                             f"range=[{ts['min']:.4g}, {ts['max']:.4g}] skew={ts['skew']}")
        lines.append("COLUMNS")
        for c in card["columns"][:max_cols]:
            lines.append(f"- {c['name']}: {c['kind']}, unique={c['n_unique']}, missing={c['null_pct']}%"
                         + (f", flags={'/'.join(c['flags'])}" if c["flags"] else ""))
        if len(card["columns"]) > max_cols:
            lines.append(f"- ... {len(card['columns']) - max_cols} more columns")
    else:
        splits = ", ".join(f"{k}: {v['images']} imgs/{v['boxes']} boxes/{v['empty_images']} empty"
                           for k, v in card["splits"].items())
        lines = [f"DATA detection format={card['kind']} classes={len(card['classes'])} splits: {splits}"]
        present = sorted((c for c in card["classes"] if c["count"]), key=lambda c: -c["count"])
        zero = len(card["classes"]) - len(present)
        lines.append("CLASSES (boxes) " + ", ".join(f"{c['name']}={c['count']}" for c in present[:40])
                     + (f" (+{zero} classes with 0 boxes)" if zero else ""))
        bs = card["box_sizes"]
        tot = sum(bs.values()) or 1
        lines.append("BOX SIZES " + ", ".join(f"{k}={v} ({100 * v / tot:.0f}%)" for k, v in bs.items()))
        if card["image_size"]:
            s = card["image_size"]
            lines.append(f"IMAGE SIZE median={s['median_w']}x{s['median_h']} min={s['min']} max={s['max']}")
    if card.get("warnings"):
        lines.append("WARNINGS " + "; ".join(card["warnings"]))
    return "\n".join(lines)
