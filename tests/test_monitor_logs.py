import math

import pytest

from loopback.jobspec import DataRef, JobSpec, TrainCfg
from loopback.logs import TrainLog, digest, read_train_log
from loopback.steps.monitor import apply_change, signals


def spec(**train) -> JobSpec:
    return JobSpec(task="classify", model="mlp", data=DataRef(format="tabular", path="/x", target="y"),
                   train=TrainCfg(epochs=50, patience=6, lr=1e-3, dropout=0.2, **train),
                   target_metric="f1_macro", target_value=0.9)


def rows(vals, losses=None, val_losses=None):
    return [{"epoch": i, "val_metric": v, "train_loss": (losses or [1.0] * len(vals))[i],
             **({"val_loss": val_losses[i]} if val_losses else {})} for i, v in enumerate(vals)]


def kinds(rs, s=None):
    return {x.kind for x in signals(rs, s or spec())}


def test_nan_is_hard_stop():
    rs = rows([0.5, 0.6], losses=[1.0, float("nan")])
    sig = signals(rs, spec())
    assert sig[0].kind == "nan_loss" and sig[0].severity == "stop"


def test_plateau_then_flat():
    assert "plateau" in kinds(rows([0.5, 0.7] + [0.69] * 4))
    assert "flat" in kinds(rows([0.5, 0.7] + [0.69] * 7))


def test_diverging_and_overfitting():
    assert "diverging" in kinds(rows([0.5] * 6, losses=[1.0, 0.5, 0.4, 0.5, 0.7, 0.9]))
    rs = rows([0.5, 0.6, 0.62, 0.63, 0.64], losses=[1, 0.8, 0.6, 0.4, 0.2], val_losses=[0.9, 0.5, 0.6, 0.7, 0.8])
    assert "overfitting" in kinds(rs)


def test_target_hit_signal():
    assert "target_hit" in kinds(rows([0.5, 0.95]))


def test_apply_change_validates():
    s = spec()
    new, desc = apply_change(s, "lr", "0.0003")
    assert new.train.lr == 0.0003 and "lr" in desc and new.fingerprint() != s.fingerprint()
    with pytest.raises(ValueError):
        apply_change(s, "imgsz", "640")          # not applicable to mlp
    with pytest.raises(ValueError):
        apply_change(s, "model", "yolov8n")      # wrong task
    with pytest.raises(ValueError):
        apply_change(s, "lr", "abc")
    assert apply_change(s, "model", "random_forest")[0].framework == "sklearn"


def test_trainlog_nan_roundtrip_and_digest(tmp_path):
    log = TrainLog(tmp_path)
    for i in range(5):
        log.write(epoch=i, train_loss=1 / (i + 1), val_metric=0.5 + i / 10, lr=1e-3)
    log.write(epoch=5, train_loss=float("nan"), val_metric=0.9, lr=1e-3)
    rs = read_train_log(tmp_path)
    assert math.isnan(rs[-1]["train_loss"])
    text = digest(tmp_path, higher_is_better=True, tail=4, header="RUN x", signals=["NAN_LOSS at epoch 5"])
    assert "LOG (last 4 of 6 epochs" in text
    assert "best_val_metric=0.9" in text and "nonfinite_losses=1" in text
    assert "SIGNALS NAN_LOSS" in text
