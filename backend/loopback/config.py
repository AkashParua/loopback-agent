from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.getenv("LOOPBACK_WORKSPACE", ROOT / "workspace")).resolve()
PROJECTS_DIR = WORKSPACE / "projects"
RUNS_DIR = WORKSPACE / "runs"
WEIGHTS_DIR = WORKSPACE / "weights"
DB_PATH = WORKSPACE / "memory.db"
DATASETS_DIR = Path(os.getenv("LOOPBACK_DATASETS", ROOT / "datasets")).resolve()

AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:8000")
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "240"))

# Training device: "auto" | "cpu" | "cuda" | "0". On a 4 GB GPU the LLM and a YOLO
# run do not fit together, so tabular defaults to CPU and detection to auto.
DEVICE = os.getenv("LOOPBACK_DEVICE", "auto")
MONITOR_EVERY = int(os.getenv("LOOPBACK_MONITOR_EVERY", "5"))  # epochs between LLM reviews
LOG_TAIL = int(os.getenv("LOOPBACK_LOG_TAIL", "12"))  # log lines the LLM sees

for d in (PROJECTS_DIR, RUNS_DIR, WEIGHTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
