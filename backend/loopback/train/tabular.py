"""Tabular training: sklearn baselines, PyTorch MLP (Lightning), TabNet (pytorch-tabnet)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

from ..steps.adapt import TabularPlan, prepare_frame
from .metrics import score
from .runctx import RunContext, resolve_device


class TabularData:
    """Adapted splits -> numpy arrays, with the fitted preprocessor from step 4."""

    def __init__(self, adapted: Path):
        self.dir = adapted
        self.meta = json.loads((adapted / "meta.json").read_text())
        self.plan = TabularPlan.model_validate_json((adapted / "plan.json").read_text())
        self.pre = joblib.load(adapted / "preprocessor.joblib")
        self.task = self.meta["task"]
        self.classes = self.meta["classes"]
        self.splits = {s: self._load(s) for s in ("train", "val", "test")}
        ytr = self.splits["train"][1]
        self.y_mean = float(ytr.mean()) if self.task == "regress" else 0.0
        self.y_std = float(ytr.std() or 1.0) if self.task == "regress" else 1.0

    def _load(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        df = prepare_frame(pd.read_parquet(self.dir / f"{split}.parquet"), self.plan)
        X = self.pre.transform(df[self.meta["features_in"]]).astype(np.float32)
        y = df[self.meta["target"]]
        if self.task == "classify":
            index = {c: i for i, c in enumerate(self.classes)}
            y = y.astype(str).map(index).fillna(-1).astype(int).to_numpy()
        else:
            y = y.astype(float).to_numpy()
        return X, y

    def decode(self, y_idx: np.ndarray) -> np.ndarray:
        return np.array(self.classes)[y_idx] if self.task == "classify" else y_idx

    def class_weights(self) -> np.ndarray | None:
        if self.task != "classify" or self.meta.get("class_weight") != "balanced":
            return None
        y = self.splits["train"][1]
        present = np.unique(y)
        w = np.ones(len(self.classes), dtype=np.float32)
        w[present] = compute_class_weight("balanced", classes=present, y=y)
        return w


# ================================================================== sklearn

def train_sklearn(ctx: RunContext, data: TabularData) -> dict:
    spec = ctx.spec
    Xtr, ytr = data.splits["train"]
    Xva, yva = data.splits["val"]
    cw = "balanced" if data.meta.get("class_weight") == "balanced" else None
    model = {
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight=cw),
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: (RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=0)
                                  if data.task == "classify" else
                                  RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)),
    }[spec.model]()
    ctx.events.info("fit_start", model=spec.model, n_train=len(ytr))
    model.fit(Xtr, ytr)
    if data.task == "classify":
        tr_loss, va_loss = _safe_log_loss(model, Xtr, ytr), _safe_log_loss(model, Xva, yva)
    else:
        tr_loss = mean_squared_error(ytr, model.predict(Xtr))
        va_loss = mean_squared_error(yva, model.predict(Xva))
    val_metric = score(spec.target_metric, *_pair(data, yva, model.predict(Xva)))
    ctx.epoch_end(0, train_loss=tr_loss, val_loss=va_loss, val_metric=val_metric, lr=None)
    joblib.dump(model, ctx.dir / "model.joblib")
    return ctx.finish({"best_val_metric": val_metric, "best_epoch": 0, "artifact": "model.joblib"})


def _safe_log_loss(model, X, y) -> float | None:
    try:
        return float(log_loss(y, model.predict_proba(X), labels=model.classes_))
    except ValueError:  # e.g. a class present in val but not in train
        return None


def _pair(data: TabularData, y_true_idx, y_pred_idx):
    if data.task == "classify":
        return data.decode(np.asarray(y_true_idx)), data.decode(np.asarray(y_pred_idx))
    return y_true_idx, y_pred_idx


# ================================================================== Lightning MLP

def build_mlp(d_in: int, d_out: int, dropout: float, hidden=(256, 128)) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    prev = d_in
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        prev = h
    layers.append(torch.nn.Linear(prev, d_out))
    return torch.nn.Sequential(*layers)


def train_mlp(ctx: RunContext, data: TabularData) -> dict:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    from torch.utils.data import DataLoader, TensorDataset

    spec, cfg = ctx.spec, ctx.spec.train
    task = data.task
    metric = spec.target_metric
    higher = spec.higher_is_better
    Xtr, ytr = data.splits["train"]
    Xva, yva = data.splits["val"]
    d_out = len(data.classes) if task == "classify" else 1

    def tensors(X, y):
        yt = torch.tensor(y, dtype=torch.long) if task == "classify" else \
            torch.tensor((y - data.y_mean) / data.y_std, dtype=torch.float32).unsqueeze(1)
        return TensorDataset(torch.tensor(X), yt)

    cw = data.class_weights()

    class MLPModule(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = build_mlp(Xtr.shape[1], d_out, cfg.dropout or 0.0)
            self.loss_fn = (torch.nn.CrossEntropyLoss(weight=torch.tensor(cw) if cw is not None else None)
                            if task == "classify" else torch.nn.MSELoss())
            self._tl, self._tn = 0.0, 0
            self._vp, self._vy, self._vl = [], [], []
            self.epoch_stats: dict = {}

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, _):
            x, y = batch
            loss = self.loss_fn(self(x), y)
            self._tl += float(loss.detach()) * len(x)
            self._tn += len(x)
            return loss

        def validation_step(self, batch, _):
            x, y = batch
            out = self(x)
            self._vl.append(float(self.loss_fn(out, y)) * len(x))
            self._vp.append(out.argmax(1).cpu().numpy() if task == "classify"
                            else out.squeeze(1).cpu().numpy() * data.y_std + data.y_mean)
            self._vy.append(y.cpu().numpy() if task == "classify"
                            else y.squeeze(1).cpu().numpy() * data.y_std + data.y_mean)

        def on_validation_epoch_end(self):
            pred, true = np.concatenate(self._vp), np.concatenate(self._vy)
            val_metric = score(metric, *_pair(data, true, pred))
            val_loss = sum(self._vl) / len(true)
            self.epoch_stats = {"val_loss": val_loss, "val_metric": val_metric}
            self.log("val_metric", val_metric if math.isfinite(val_metric) else float("nan"))
            self._vp, self._vy, self._vl = [], [], []

        def configure_optimizers(self):
            opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr or 1e-3, weight_decay=cfg.weight_decay or 0.0)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max" if higher else "min", factor=0.5, patience=max(2, cfg.patience // 3))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_metric"}}

    class LoopbackLogger(L.Callback):
        def __init__(self):
            self._gn, self._gk = 0.0, 0
            self.best = None

        def on_before_optimizer_step(self, trainer, module, optimizer):
            sq = sum(float(p.grad.detach().pow(2).sum()) for p in module.parameters() if p.grad is not None)
            self._gn += math.sqrt(sq)
            self._gk += 1

        def on_train_epoch_end(self, trainer, module):
            train_loss = module._tl / max(1, module._tn)
            module._tl, module._tn = 0.0, 0
            vm = module.epoch_stats.get("val_metric")
            if vm is not None and math.isfinite(vm) and (
                    self.best is None or (vm > self.best[1] if higher else vm < self.best[1])):
                self.best = (trainer.current_epoch, vm)
                torch.save(module.net.state_dict(), ctx.dir / "model.pt")
            stop = ctx.epoch_end(
                trainer.current_epoch, train_loss=train_loss,
                val_loss=module.epoch_stats.get("val_loss"), val_metric=vm,
                lr=trainer.optimizers[0].param_groups[0]["lr"],
                grad_norm=self._gn / max(1, self._gk))
            self._gn, self._gk = 0.0, 0
            if stop:
                trainer.should_stop = True

    early = EarlyStopping(monitor="val_metric", mode="max" if higher else "min",
                          patience=cfg.patience, check_finite=True)
    logger_cb = LoopbackLogger()
    device = resolve_device(cfg.device)
    trainer = L.Trainer(max_epochs=cfg.epochs, accelerator="gpu" if device.startswith("cuda") else "cpu",
                        devices=1, callbacks=[logger_cb, early], logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                        default_root_dir=str(ctx.dir))
    bs = min(cfg.batch, len(Xtr))
    train_dl = DataLoader(tensors(Xtr, ytr), batch_size=bs, shuffle=True, drop_last=len(Xtr) > bs)
    val_dl = DataLoader(tensors(Xva, yva), batch_size=1024)
    ctx.events.info("fit_start", model="mlp", n_train=len(ytr), d_in=int(Xtr.shape[1]), device=device)
    module = MLPModule()
    trainer.fit(module, train_dl, val_dl)
    meta = {"d_in": int(Xtr.shape[1]), "d_out": d_out, "dropout": cfg.dropout or 0.0,
            "y_mean": data.y_mean, "y_std": data.y_std}
    (ctx.dir / "model_meta.json").write_text(json.dumps(meta))
    best_epoch, best = logger_cb.best or (None, None)
    return ctx.finish({"best_val_metric": best, "best_epoch": best_epoch, "artifact": "model.pt"},
                      patience_stop=early.stopped_epoch > 0)


# ================================================================== TabNet

def train_tabnet(ctx: RunContext, data: TabularData) -> dict:
    from pytorch_tabnet.callbacks import Callback
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    spec, cfg = ctx.spec, ctx.spec.train
    higher = spec.higher_is_better
    Xtr, ytr = data.splits["train"]
    Xva, yva = data.splits["val"]
    device = resolve_device(cfg.device)
    state = {"best": None}

    class Bridge(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if data.task == "classify":
                pred = self.trainer.predict(Xva)
            else:
                pred = self.trainer.predict(Xva).reshape(-1)
            vm = score(spec.target_metric, *_pair(data, yva, pred))
            if state["best"] is None or (vm > state["best"][1] if higher else vm < state["best"][1]):
                state["best"] = (epoch, vm)
            if ctx.epoch_end(epoch, train_loss=logs.get("loss"), val_metric=vm, lr=logs.get("lr")):
                self.trainer._stop_training = True

    kw = dict(verbose=0, device_name=device, optimizer_params={"lr": cfg.lr or 2e-2}, seed=0)
    if data.task == "classify":
        model = TabNetClassifier(**kw)
        fit_y, eval_y, metric = ytr, yva, ["accuracy"]
    else:
        model = TabNetRegressor(**kw)
        fit_y, eval_y, metric = ytr.reshape(-1, 1), yva.reshape(-1, 1), ["rmse"]
    bs = min(cfg.batch, len(Xtr))
    ctx.events.info("fit_start", model="tabnet", n_train=len(ytr), device=device)
    model.fit(Xtr, fit_y, eval_set=[(Xva, eval_y)], eval_name=["val"], eval_metric=metric,
              max_epochs=cfg.epochs, patience=cfg.patience, batch_size=bs,
              virtual_batch_size=max(16, bs // 4), drop_last=False, callbacks=[Bridge()],
              weights=1 if data.meta.get("class_weight") == "balanced" else 0)
    model.save_model(str(ctx.dir / "tabnet"))
    best_epoch, best = state["best"] or (None, None)
    ran = ctx.epoch + 1
    return ctx.finish({"best_val_metric": best, "best_epoch": best_epoch, "artifact": "tabnet.zip"},
                      patience_stop=ran < cfg.epochs and ctx.stop_reason is None)


def train_tabular(ctx: RunContext) -> dict:
    data = TabularData(Path(ctx.spec.data.path))
    if ctx.spec.model in ("logreg", "ridge", "random_forest"):
        return train_sklearn(ctx, data)
    if ctx.spec.model == "mlp":
        return train_mlp(ctx, data)
    if ctx.spec.model == "tabnet":
        return train_tabnet(ctx, data)
    raise ValueError(f"unknown tabular model {ctx.spec.model}")
