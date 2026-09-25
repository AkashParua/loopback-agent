"""Fixed model zoo. The agent may only choose from here."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class ZooEntry:
    name: str
    tasks: tuple[str, ...]
    framework: str  # "sklearn" | "pytorch"
    library: str
    note: str
    requires: str | None = None

    @property
    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None


ZOO: dict[str, ZooEntry] = {e.name: e for e in [
    ZooEntry("logreg", ("classify",), "sklearn", "scikit-learn", "fast linear baseline"),
    ZooEntry("ridge", ("regress",), "sklearn", "scikit-learn", "fast linear baseline"),
    ZooEntry("random_forest", ("classify", "regress"), "sklearn", "scikit-learn",
             "strong non-linear baseline, robust to scaling"),
    ZooEntry("mlp", ("classify", "regress"), "pytorch", "lightning",
             "PyTorch MLP (Lightning), 2 hidden layers + BatchNorm + dropout"),
    ZooEntry("tabnet", ("classify", "regress"), "pytorch", "pytorch-tabnet",
             "TabNet attention MLP, good on mid-size tabular", requires="pytorch_tabnet"),
    ZooEntry("yolov8n", ("detect",), "pytorch", "ultralytics", "smallest/fastest detector"),
    ZooEntry("yolov8s", ("detect",), "pytorch", "ultralytics", "more capacity, ~3x slower than n"),
    ZooEntry("yolo11n", ("detect",), "pytorch", "ultralytics", "newer nano detector, better small objects"),
]}


def models_for(task: str) -> list[ZooEntry]:
    return [e for e in ZOO.values() if task in e.tasks and e.available]
