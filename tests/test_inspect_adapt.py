import json

from loopback.steps.adapt import apply_detect, apply_tabular, plan_detect, plan_tabular, yolo_to_coco
from loopback.steps.context import grounded
from loopback.steps.inspect_data import card_digest, detect_kind, inspect


def test_tabular_card_flags_problems(tabular_csv):
    card = inspect(tabular_csv, target="label")
    flags = {c["name"]: c["flags"] for c in card["columns"]}
    assert "id_like" in flags["id"]
    assert "constant" in flags["constant"]
    assert "high_null" in flags["mostly_null"]
    assert card["target_stats"]["kind"] == "categorical"
    assert any(s["column"] == "leak" for s in card["leakage_suspects"])
    assert "possible leakage: leak" in " ".join(card["warnings"])
    digest = card_digest(card)
    assert "TARGET label categorical" in digest and "WARNINGS" in digest


def test_tabular_adapt_drops_and_splits(tabular_csv, workspace):
    card = inspect(tabular_csv, target="label")
    plan = plan_tabular(card, "label", "classify", user_drop=["color"])
    dropped = {d.column for d in plan.drop}
    assert {"id", "constant", "mostly_null", "leak", "color"} <= dropped
    diff = apply_tabular(plan, tabular_csv, workspace / "adapt_tab")
    assert sum(diff["split"].values()) == diff["rows"]["after"] == 400
    assert diff["columns"]["features_out"] == 2  # x1, x2
    meta = json.loads((workspace / "adapt_tab" / "meta.json").read_text())
    assert meta["classes"] == ["no", "yes"]


def test_yolo_card_detects_leak_and_empty(yolo_dir):
    assert detect_kind(yolo_dir)[0] == "yolo"
    card = inspect(yolo_dir)
    assert card["splits"]["train"]["images"] == 8
    assert card["overlap"]["by_hash"] == 1
    assert [c["name"] for c in card["classes"]] == ["cat", "dog", "bird"]
    assert any("zero boxes" in w for w in card["warnings"])  # bird
    assert any("split leakage" in w for w in card["warnings"])


def test_coco_roundtrip_and_conversion(yolo_dir, workspace):
    coco = workspace / "coco_ds"
    counts = yolo_to_coco(yolo_dir, coco)
    assert counts == {"train": 8, "val": 4}
    assert detect_kind(coco)[0] == "coco"
    card = inspect(coco)
    yolo_card = inspect(yolo_dir)
    assert {k: v["boxes"] for k, v in card["splits"].items()} == {k: v["boxes"] for k, v in yolo_card["splits"].items()}

    plan = plan_detect(card)
    assert plan.convert == "coco->yolo" and plan.dedupe_val and "bird" in plan.drop_classes
    plan.rename = {"dog": "pet", "cat": "pet"}  # merge two classes
    plan.class_map = ["pet"]
    diff = apply_detect(plan, coco, workspace / "adapt_det")
    assert diff["removed_val_duplicates"] == 1
    assert diff["classes_after"] == ["pet"]
    back = inspect(workspace / "adapt_det")
    assert back["kind"] == "yolo" and [c["name"] for c in back["classes"]] == ["pet"]
    assert back["overlap"]["by_hash"] == 0
    assert sum(s["boxes"] for s in back["splits"].values()) == sum(s["boxes"] for s in card["splits"].values()) - 1


def test_val_split_created_when_missing(workspace):
    from tests.conftest import _yolo
    root = _yolo(workspace / "noval", n_train=10, n_val=0)
    card = inspect(root)
    plan = plan_detect(card)
    assert plan.val_fraction == 0.2
    diff = apply_detect(plan, root, workspace / "noval_adapted")
    assert diff["images"] == {"train": 8, "val": 2}


def test_grounding_rejects_invented_numbers():
    src = "- age: numeric, unique=58, missing=0.0%\nTARGET y: 0=300 (61.6%)"
    assert grounded("y is 61.6% zeros", src)
    assert not grounded("age has 7.4% missing", src)
