from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def score(metric: str, y_true, y_pred) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"not a tabular metric: {metric}")


def all_metrics(task: str, y_true, y_pred) -> dict[str, float]:
    names = ["accuracy", "f1_macro"] if task == "classify" else ["rmse", "mae", "r2"]
    return {m: round(score(m, y_true, y_pred), 5) for m in names}
