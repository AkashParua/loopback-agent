"""Typed contracts for every LLM task.

Small models do best when the answer space is closed. Every task returns a
pydantic model; enums are Literals so Ollama's JSON-schema constrained decoding
can only emit valid values. Tasks whose valid values depend on the request
(candidate ids, column names) build their model at request time.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, create_model

Task = Literal["classify", "regress", "detect"]
Metric = Literal["accuracy", "f1_macro", "rmse", "mae", "r2", "mAP50", "mAP50-95"]
Param = Literal["lr", "batch", "epochs", "patience", "dropout", "weight_decay", "imgsz", "model"]
FailureMode = Literal[
    "nan_loss", "diverging", "dead_gradients", "plateau", "overfitting",
    "underfitting", "data_issue", "out_of_memory", "crash", "none",
]


# ---------------------------------------------------------------- requests

class TaskRequest(BaseModel):
    """Body for POST /v1/tasks/{task}. `context` is the digest the backend built."""

    context: str = Field(..., description="LLM-readable digest (data card, log tail, ...)")
    choices: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Closed sets for dynamic enums, e.g. {'candidate_id': [0,1,2]}",
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Stateless chat: the client owns and resends the whole history."""

    messages: list[ChatMessage]
    context: Optional[str] = Field(None, description="Current project digest to ground answers")


class ChatResponse(BaseModel):
    reply: str
    messages: list[ChatMessage]


# ---------------------------------------------------------------- outputs

class DataSummary(BaseModel):
    summary: str = Field(..., description="3-5 plain sentences about the dataset")
    task: Task
    target_metric: Metric
    risks: list[str] = Field(default_factory=list, max_length=5,
                             description="Concrete data risks, each citing a number from the card")
    questions: list[str] = Field(default_factory=list, max_length=3,
                                 description="Questions for the user, only if truly needed")


class Overrides(BaseModel):
    epochs: Optional[int] = Field(None, ge=1, le=300)
    batch: Optional[int] = Field(None, ge=1, le=512)
    lr: Optional[float] = Field(None, gt=0, le=1)
    imgsz: Optional[int] = Field(None, ge=128, le=1536)


class MonitorChange(BaseModel):
    param: Param
    value: str = Field(..., description="New value, e.g. '0.0003' or 'yolov8s'")


class MonitorVerdict(BaseModel):
    verdict: Literal["continue", "change", "stop"]
    evidence_epoch: int = Field(..., ge=0, description="Epoch of the log line that justifies the verdict")
    evidence: str = Field(..., description="Quote the measured value, e.g. 'val_metric 0.71 flat since epoch 7'")
    change: Optional[MonitorChange] = None
    reason: str = Field(..., description="One sentence")


class Diagnosis(BaseModel):
    failure_mode: FailureMode
    evidence: str = Field(..., description="The log line or metric that shows it")
    fix: str = Field(..., description="One concrete change to try next")


class NextStep(BaseModel):
    action: Literal["retrain", "finish"]
    change: Optional[MonitorChange] = None
    evidence: str
    reason: str


class ReportNarrative(BaseModel):
    headline: str = Field(..., description="One sentence: did we hit the target, and the final number")
    model_rationale: str = Field(..., description="Why this model fit this data, 1-2 sentences")
    curve_paragraph: str = Field(..., description="One paragraph describing the training curve")
    what_to_try_next: list[str] = Field(..., min_length=1, max_length=4)


# ---------------------------------------------------------------- dynamic outputs

def _literal(values: list[Any]):
    if not values:
        raise ValueError("choice list must not be empty")
    return Literal[tuple(values)]  # type: ignore[valid-type]


def _keep(allowed: set, key: Optional[str] = None):
    """Drop out-of-set items instead of failing the whole answer.

    With constrained decoding (Ollama >= 0.34) this never triggers; on engines that
    ignore the schema it turns a retry-loop into a partial, still-valid answer.
    The JSON schema the model sees keeps the strict enum.
    """
    def f(v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return [x for x in v if (x.get(key) if key and isinstance(x, dict) else x) in allowed]
    return BeforeValidator(f)


def suggestions_model(candidate_ids: list[int]) -> type[BaseModel]:
    Cid = _literal(candidate_ids)
    Pick = create_model(
        "Pick",
        candidate_id=(Cid, ...),
        reason=(str, Field(..., description="One line, cite a data fact")),
        overrides=(Overrides, Field(default_factory=Overrides)),
    )
    return create_model(
        "Suggestions",
        picks=(Annotated[list[Pick], _keep(set(candidate_ids), "candidate_id")],
               Field(..., min_length=1, max_length=3, description="Best first")),
    )


def adapt_review_model(columns: list[str]) -> type[BaseModel]:
    fields: dict[str, Any] = {
        "notes": (str, Field(..., description="2-3 sentences reviewing the plan")),
        "warnings": (list[str], Field(default_factory=list, max_length=4)),
    }
    if columns:
        fields["extra_drop"] = (
            Annotated[list[_literal(columns)], _keep(set(columns))],
            Field(default_factory=list, description="Extra columns to drop; only leakage or IDs"),
        )
    return create_model("AdaptReview", **fields)


STATIC_OUTPUTS: dict[str, type[BaseModel]] = {
    "data_summary": DataSummary,
    "monitor": MonitorVerdict,
    "diagnose": Diagnosis,
    "next_step": NextStep,
    "report": ReportNarrative,
}

DYNAMIC_OUTPUTS = {
    "suggest": lambda ch: suggestions_model(list(ch.get("candidate_id", []))),
    "adapt_review": lambda ch: adapt_review_model([str(c) for c in ch.get("columns", [])]),
}

TASKS = sorted([*STATIC_OUTPUTS, *DYNAMIC_OUTPUTS])


def output_model(task: str, choices: dict[str, list[Any]]) -> type[BaseModel]:
    if task in STATIC_OUTPUTS:
        return STATIC_OUTPUTS[task]
    if task in DYNAMIC_OUTPUTS:
        return DYNAMIC_OUTPUTS[task](choices)
    raise KeyError(task)
