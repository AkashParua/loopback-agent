"""Tests run fully offline: the agent URL points at a closed port so every LLM
step exercises its rule-based fallback. `live` tests talk to a real agent."""

import os
import tempfile
from pathlib import Path

_WS = tempfile.mkdtemp(prefix="loopback-test-")
os.environ["LOOPBACK_WORKSPACE"] = _WS
os.environ.setdefault("LOOPBACK_DATASETS", str(Path(_WS) / "datasets"))
os.environ["AGENT_URL"] = os.environ.get("LIVE_AGENT_URL", "http://127.0.0.1:9")
os.environ["AGENT_TIMEOUT"] = "5" if "LIVE_AGENT_URL" not in os.environ else "400"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture(scope="session")
def workspace() -> Path:
    return Path(_WS)


@pytest.fixture(scope="session")
def tabular_csv(workspace) -> Path:
    rng = np.random.default_rng(0)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame({
        "id": np.arange(n),                                   # id-like -> dropped
        "x1": x1, "x2": x2,
        "color": rng.choice(["red", "green", "blue"], size=n),
        "mostly_null": np.where(rng.random(n) < 0.8, np.nan, 1.0),  # >60% missing -> dropped
        "constant": 7,                                        # constant -> dropped
        "leak": y,                                            # leaks the target
        "label": np.where(y == 1, "yes", "no"),
    })
    df.loc[rng.choice(n, 20, replace=False), "x2"] = np.nan
    p = workspace / "tab" / "data.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def _yolo(root: Path, n_train: int = 8, n_val: int = 3, dup: bool = False) -> Path:
    rng = np.random.default_rng(1)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (rng.random((96, 128, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(root / "images" / split / f"{split}{i}.jpg")
            lines = [f"{i % 2} 0.5 0.5 0.3 0.4", "1 0.25 0.25 0.1 0.1"] if i % 4 else []
            (root / "labels" / split / f"{split}{i}.txt").write_text("\n".join(lines))
    if dup:  # identical image in train and val -> split leakage
        (root / "images" / "val" / "dup.jpg").write_bytes((root / "images" / "train" / "train1.jpg").read_bytes())
        (root / "labels" / "val" / "dup.txt").write_text("0 0.5 0.5 0.3 0.4")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n  2: bird\n")
    return root


@pytest.fixture(scope="session")
def yolo_dir(workspace) -> Path:
    return _yolo(workspace / "yolo_ds", dup=True)
