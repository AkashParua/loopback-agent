import pytest
from pydantic import ValidationError

from harness.schemas import TASKS, MonitorVerdict, adapt_review_model, output_model, suggestions_model


def test_all_tasks_have_schemas():
    for t in TASKS:
        assert output_model(t, {"candidate_id": [0, 1], "columns": ["a"]}).model_json_schema()


def test_dynamic_enums_are_strict_in_schema_but_lenient_on_parse():
    S = suggestions_model([0, 3])
    schema = S.model_json_schema()
    assert schema["$defs"]["Pick"]["properties"]["candidate_id"]["enum"] == [0, 3]
    parsed = S.model_validate({"picks": [{"candidate_id": 9, "reason": "x"}, {"candidate_id": 3, "reason": "y"}]})
    assert [p.candidate_id for p in parsed.picks] == [3]
    with pytest.raises(ValidationError):  # nothing valid left -> retry
        S.model_validate({"picks": [{"candidate_id": 9, "reason": "x"}]})
    A = adapt_review_model(["age", "fare"])
    assert A.model_validate({"notes": "n", "extra_drop": ["id", "fare"]}).extra_drop == ["fare"]


def test_monitor_verdict_enum():
    with pytest.raises(ValidationError):
        MonitorVerdict.model_validate({"verdict": "maybe", "evidence_epoch": 1, "evidence": "e", "reason": "r"})
