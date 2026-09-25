"""Backend HTTP API. Every step is an explicit POST so the UI keeps full control."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import memory, pipeline, runs
from .agent_client import agent
from .config import AGENT_URL, WORKSPACE

watcher = pipeline.Watcher()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    watcher.start()
    yield
    watcher.stop()


app = FastAPI(title="Loopback backend", version="0.1.0", lifespan=lifespan)


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


async def call(fn, *args, **kwargs):
    """Run blocking pipeline code (pandas, LLM calls) off the event loop."""
    return await run_in_threadpool(_guard, fn, *args, **kwargs)


def _p(p: pipeline.Project) -> dict:
    return pipeline.as_dict(p)


# ------------------------------------------------------------------ meta

@app.get("/health")
async def health():
    return {"ok": True, "workspace": str(WORKSPACE), "agent_url": AGENT_URL,
            "agent": await run_in_threadpool(agent.health)}


@app.get("/datasets")
async def datasets():
    return await call(pipeline.list_datasets)


# ------------------------------------------------------------------ projects

@app.post("/projects")
async def create_project():
    return _p(await call(pipeline.create))


@app.get("/projects")
async def list_projects():
    return await call(pipeline.list_projects)


@app.get("/projects/{pid}")
async def get_project(pid: str):
    return _p(await call(pipeline.load, pid))


@app.get("/projects/{pid}/digest", response_class=PlainTextResponse)
async def project_digest(pid: str):
    return await call(pipeline.digest_for_chat, pid)


@app.get("/projects/{pid}/experiments")
async def experiments(pid: str):
    return await call(memory.history, pid)


# ------------------------------------------------------------------ step 1: data

class DatasetBody(BaseModel):
    name: str


class ScanBody(BaseModel):
    target: Optional[str] = None


class ConfirmBody(BaseModel):
    task: str
    target: Optional[str] = None
    target_metric: str
    target_value: Optional[float] = None


@app.post("/projects/{pid}/upload")
async def upload(pid: str, files: list[UploadFile] = File(...)):
    blobs = [(f.filename or "upload", await f.read()) for f in files]
    return _p(await call(pipeline.upload, pid, blobs))


@app.post("/projects/{pid}/dataset")
async def use_dataset(pid: str, body: DatasetBody):
    return _p(await call(pipeline.use_dataset, pid, body.name))


@app.post("/projects/{pid}/scan")
async def scan(pid: str, body: ScanBody):
    return _p(await call(pipeline.scan, pid, body.target))


@app.post("/projects/{pid}/confirm")
async def confirm(pid: str, body: ConfirmBody):
    return _p(await call(pipeline.confirm_data, pid, body.task, body.target, body.target_metric, body.target_value))


# ------------------------------------------------------------------ step 2: context

class ContextBody(BaseModel):
    notes: str = ""
    constraints: dict = Field(default_factory=dict)


@app.post("/projects/{pid}/context")
async def context(pid: str, body: ContextBody):
    return _p(await call(pipeline.run_context, pid, body.notes, body.constraints))


@app.post("/projects/{pid}/context/approve")
async def approve_context(pid: str):
    return _p(await call(pipeline.approve_context, pid))


# ------------------------------------------------------------------ step 3: suggest

class SuggestBody(BaseModel):
    feedback: str = ""


class ApproveSpecBody(BaseModel):
    index: int
    spec_yaml: Optional[str] = None


@app.post("/projects/{pid}/suggest")
async def suggest(pid: str, body: SuggestBody):
    return _p(await call(pipeline.run_suggest, pid, body.feedback))


@app.post("/projects/{pid}/suggest/approve")
async def approve_suggestion(pid: str, body: ApproveSpecBody):
    return _p(await call(pipeline.approve_suggestion, pid, body.index, body.spec_yaml))


# ------------------------------------------------------------------ step 4: adapt

class PlanBody(BaseModel):
    plan: dict


@app.post("/projects/{pid}/adapt/plan")
async def plan_adapt(pid: str):
    return _p(await call(pipeline.plan_adapt, pid))


@app.post("/projects/{pid}/adapt/apply")
async def apply_adapt(pid: str, body: PlanBody):
    return _p(await call(pipeline.apply_adapt, pid, body.plan))


# ------------------------------------------------------------------ step 5-6: train / watch

class TrainBody(BaseModel):
    spec_yaml: Optional[str] = None


class ControlBody(BaseModel):
    action: str


class VerdictBody(BaseModel):
    approve: bool


class ChangeBody(BaseModel):
    param: str
    value: str


@app.post("/projects/{pid}/train")
async def train(pid: str, body: TrainBody):
    return _p(await call(pipeline.start_training, pid, body.spec_yaml))


@app.get("/runs/{rid}")
async def get_run(rid: str):
    return await call(pipeline.run_view, rid)


@app.post("/runs/{rid}/control")
async def control(rid: str, body: ControlBody):
    return await call(runs.control, rid, body.action)


@app.post("/runs/{rid}/review")
async def review(rid: str):
    return await call(pipeline.review, rid)


@app.post("/runs/{rid}/verdict")
async def verdict(rid: str, body: VerdictBody):
    return await call(pipeline.resolve_verdict, rid, body.approve)


# ------------------------------------------------------------------ step 7-8

@app.post("/runs/{rid}/evaluate")
async def evaluate(rid: str):
    return await call(pipeline.evaluate, rid)


@app.post("/runs/{rid}/next")
async def next_step(rid: str):
    return await call(pipeline.next_step, rid)


@app.post("/runs/{rid}/next/start")
async def start_next(rid: str, body: ChangeBody):
    return await call(pipeline.start_next, rid, body.model_dump())


@app.post("/runs/{rid}/report", response_class=PlainTextResponse)
async def report(rid: str):
    return await call(pipeline.build_report, rid)


@app.get("/runs/{rid}/files/{name}")
async def run_file(rid: str, name: str):
    d = _guard(runs.run_dir, rid)
    f = (d / name).resolve()
    if f.parent != d or not f.is_file():
        raise HTTPException(404, name)
    return FileResponse(f)


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8100")))
