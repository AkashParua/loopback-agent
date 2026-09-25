"""Pull small open datasets to exercise every step.

    python scripts/fetch_datasets.py            # tabular + coco8 + medical-pills (+ COCO copy)
    python scripts/fetch_datasets.py --large    # also african-wildlife (105 MB)

Writes to ./datasets (or $LOOPBACK_DATASETS).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

OUT = Path(os.getenv("LOOPBACK_DATASETS", ROOT / "datasets"))
ULTRA = "https://github.com/ultralytics/assets/releases/download/v0.0.0/{}.zip"
TITANIC = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def _get(url: str) -> bytes:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def tabular() -> None:
    from sklearn import datasets

    d = OUT / "breast_cancer"
    if not d.exists():
        d.mkdir(parents=True)
        bc = datasets.load_breast_cancer(as_frame=True)
        df = bc.frame.rename(columns={"target": "diagnosis"})
        df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
        df.to_csv(d / "breast_cancer.csv", index=False)
        print("  breast_cancer: 569 rows, binary classification")

    d = OUT / "diabetes"
    if not d.exists():
        d.mkdir(parents=True)
        dia = datasets.load_diabetes(as_frame=True, scaled=False)
        dia.frame.rename(columns={"target": "progression"}).to_parquet(d / "diabetes.parquet")
        print("  diabetes: 442 rows, regression (parquet)")

    d = OUT / "titanic"
    if not d.exists():
        d.mkdir(parents=True)
        (d / "titanic.csv").write_bytes(_get(TITANIC))
        print("  titanic: 891 rows, nulls + IDs + free text")


def ultralytics_zip(name: str) -> Path:
    d = OUT / name
    if not d.exists():
        with zipfile.ZipFile(io.BytesIO(_get(ULTRA.format(name)))) as zf:
            tops = {n.split("/")[0] for n in zf.namelist()}
            zf.extractall(OUT if tops == {name} else d)  # some zips have no top folder
        if not list(d.glob("*.yaml")):  # e.g. coco8: its yaml ships inside the ultralytics package
            import ultralytics
            import yaml

            cfg = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/datasets" / f"{name}.yaml").read_text())
            cfg = {"path": ".", **{k: cfg[k] for k in ("train", "val", "test", "names") if cfg.get(k)}}
            (d / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"  {name}: YOLO format")
    return d


def coco_copy(src: Path, dst: Path) -> None:
    """Real YOLO data re-exported as COCO so the COCO path is tested on real boxes."""
    from loopback.steps.adapt import yolo_to_coco

    if dst.exists():
        return
    yolo_to_coco(src, dst)
    print(f"  {dst.name}: COCO format (converted from {src.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="also fetch african-wildlife (105 MB)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"datasets -> {OUT}")
    tabular()
    ultralytics_zip("coco8")
    pills = ultralytics_zip("medical-pills")
    coco_copy(pills, OUT / "medical-pills-coco")
    if args.large:
        ultralytics_zip("african-wildlife")
    print("done")


if __name__ == "__main__":
    main()
