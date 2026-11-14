"""Read-only views of a project's data for the UI: distributions, correlations, sample
images with their boxes. Nothing here feeds the agent; it is for the human."""

from __future__ import annotations

import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .steps.inspect_data import coco_layout, load_table, read_yolo_labels, yolo_label_path, yolo_layout

MAX_ROWS = 5000
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d",
           "#ea580c", "#4f46e5", "#059669", "#be123c"]


# ================================================================== tabular

def tabular_profile(card: dict) -> dict:
    df = load_table(Path(card["path"]))
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0)
    target = card.get("target")
    kinds = {c["name"]: c["kind"] for c in card["columns"]}
    numeric = [c for c in df.columns if kinds.get(c) in ("numeric", "bool") and c != target
               and pd.api.types.is_numeric_dtype(df[c])]
    hists = {}
    for c in numeric[:16]:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        counts, edges = np.histogram(s, bins=min(24, max(5, s.nunique())))
        hists[c] = {"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]}
    corr_df = df[numeric[:18]].astype(float)
    tcode = None
    if target and target in df:
        t = df[target]
        tcode = t.astype(float) if pd.api.types.is_numeric_dtype(t) else pd.Series(pd.factorize(t)[0], index=t.index)
        corr_df = corr_df.assign(**{f"{target} (target)": tcode})
    corr = corr_df.corr().round(3) if corr_df.shape[1] > 1 else pd.DataFrame()
    top = []
    if tcode is not None and numeric:
        tc = df[numeric].astype(float).corrwith(tcode).abs().sort_values(ascending=False)
        top = [c for c in tc.index if pd.notna(tc[c])][:4]
    sample = df[[*top, target]].dropna().head(1500) if target and top else pd.DataFrame()
    missing = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    return {
        "n_rows_profiled": int(len(df)),
        "histograms": hists,
        "correlation": {"columns": list(corr.columns), "values": corr.fillna(0).values.tolist()},
        "top_features": top,
        "target": target,
        "scatter": json.loads(sample.to_json(orient="records")) if not sample.empty else [],
        "missing_pct": {k: float(v) for k, v in missing.items() if v > 0},
        "head": json.loads(df.head(25).to_json(orient="records")),
    }


# ================================================================== detection

def _yolo_items(root: Path, names: list[str]) -> list[dict]:
    _, splits, _ = yolo_layout(root)
    out = []
    for split, imgs in splits.items():
        for img in imgs:
            boxes, _ = read_yolo_labels(yolo_label_path(img))
            out.append({"split": split, "image": img,
                        "boxes": [(c, x - w / 2, y - h / 2, x + w / 2, y + h / 2) for c, x, y, w, h in boxes],
                        "normalized": True})
    return out


def _coco_items(root: Path) -> tuple[list[dict], list[str]]:
    out, names = [], {}
    for split, (js, img_dir) in coco_layout(root).items():
        d = json.loads(js.read_text())
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        names.update(cats)
        order = {cid: i for i, cid in enumerate(sorted(cats))}
        by_img: dict[int, list] = {}
        for a in d.get("annotations", []):
            x, y, w, h = a["bbox"]
            by_img.setdefault(a["image_id"], []).append((order.get(a["category_id"], 0), x, y, x + w, y + h))
        for im in d.get("images", []):
            p = img_dir / im["file_name"]
            if not p.exists():
                p = img_dir / Path(im["file_name"]).name
            out.append({"split": split, "image": p, "boxes": by_img.get(im["id"], []), "normalized": False,
                        "wh": (im.get("width"), im.get("height"))})
    return out, [names[k] for k in sorted(names)]


def _items(card: dict) -> tuple[list[dict], list[str]]:
    root = Path(card["path"])
    names = [c["name"] for c in card["classes"]]
    if card["kind"] == "coco":
        items, coco_names = _coco_items(root)
        return items, coco_names or names
    return _yolo_items(root, names), names


def detect_profile(card: dict) -> dict:
    items, names = _items(card)
    per_image = Counter()
    pts = []
    rng = random.Random(0)
    for it in items:
        per_image[len(it["boxes"])] += 1
        wh = it.get("wh")
        for c, x1, y1, x2, y2 in it["boxes"]:
            if it["normalized"]:
                w, h = x2 - x1, y2 - y1
            elif wh and wh[0] and wh[1]:
                w, h = (x2 - x1) / wh[0], (y2 - y1) / wh[1]
            else:
                continue
            pts.append((names[c] if c < len(names) else str(c), round(w, 4), round(h, 4)))
    if len(pts) > 3000:
        pts = rng.sample(pts, 3000)
    return {"boxes_per_image": {str(k): v for k, v in sorted(per_image.items())},
            "box_wh": [{"class": c, "w": w, "h": h} for c, w, h in pts],
            "splits": sorted({it["split"] for it in items}), "n_images": len(items)}


def _font(size: int):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sample_grid(card: dict, *, n: int = 12, split: Optional[str] = None, seed: int = 0, cols: int = 4,
                tile: int = 300) -> bytes:
    """PNG grid of random images with their ground-truth boxes drawn, one color per class."""
    items, names = _items(card)
    pool = [it for it in items if (split is None or it["split"] == split) and it["image"].exists()]
    with_boxes = [it for it in pool if it["boxes"]]
    pool = with_boxes if len(with_boxes) >= n else pool
    picks = random.Random(seed).sample(pool, min(n, len(pool)))
    rows = max(1, (len(picks) + cols - 1) // cols)
    grid = Image.new("RGB", (cols * tile, rows * tile), (245, 245, 245))
    font = _font(14)
    for k, it in enumerate(picks):
        with Image.open(it["image"]) as im:
            im = im.convert("RGB")
            W, H = im.size
            s = tile / max(W, H)
            im = im.resize((max(1, int(W * s)), max(1, int(H * s))))
        draw = ImageDraw.Draw(im)
        for c, x1, y1, x2, y2 in it["boxes"]:
            if it["normalized"]:
                x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
            color = PALETTE[c % len(PALETTE)]
            box = [x1 * s, y1 * s, x2 * s, y2 * s]
            draw.rectangle(box, outline=color, width=2)
            label = names[c] if c < len(names) else str(c)
            tw = draw.textlength(label, font=font)
            draw.rectangle([box[0], max(0, box[1] - 17), box[0] + tw + 6, max(17, box[1])], fill=color)
            draw.text((box[0] + 3, max(0, box[1] - 17) + 1), label, fill="white", font=font)
        ox, oy = (k % cols) * tile + (tile - im.width) // 2, (k // cols) * tile + (tile - im.height) // 2
        grid.paste(im, (ox, oy))
    buf = io.BytesIO()
    grid.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
