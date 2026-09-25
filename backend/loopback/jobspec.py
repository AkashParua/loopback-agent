"""The job spec is the only thing the agent hands to training (README: 'Model zoo')."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

Task = Literal["classify", "regress", "detect"]
Metric = Literal["accuracy", "f1_macro", "rmse", "mae", "r2", "mAP50", "mAP50-95"]

HIGHER_IS_BETTER = {"accuracy": True, "f1_macro": True, "r2": True, "mAP50": True,
                    "mAP50-95": True, "rmse": False, "mae": False}


class DataRef(BaseModel):
    format: Literal["tabular", "yolo", "coco"]
    path: str
    target: Optional[str] = None


class AdaptCfg(BaseModel):
    imgsz: Optional[int] = None
    class_map: Optional[list[str]] = None
    fliplr: Optional[float] = None


class TrainCfg(BaseModel):
    epochs: int = 50
    batch: int = 16
    patience: int = 10
    lr: Optional[float] = None
    dropout: Optional[float] = None
    weight_decay: Optional[float] = None
    device: str = "auto"


class JobSpec(BaseModel):
    task: Task
    framework: Literal["pytorch", "sklearn"] = "pytorch"
    model: str
    data: DataRef
    adapt: AdaptCfg = Field(default_factory=AdaptCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    target_metric: Metric
    target_value: Optional[float] = None
    reason: Optional[str] = None

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "JobSpec":
        return cls.model_validate(yaml.safe_load(text))

    def save(self, path: Path) -> None:
        path.write_text(self.to_yaml())

    @classmethod
    def load(cls, path: Path) -> "JobSpec":
        return cls.from_yaml(path.read_text())

    def fingerprint(self) -> str:
        """Identity of the *idea*: model + training knobs, not paths or prose."""
        core = {"task": self.task, "model": self.model,
                "adapt": self.adapt.model_dump(exclude_none=True),
                "train": self.train.model_dump(exclude_none=True, exclude={"device"})}
        return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def higher_is_better(self) -> bool:
        return HIGHER_IS_BETTER[self.target_metric]
