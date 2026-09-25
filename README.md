# Loopback Agent — Bare Minimum Spec

An agent that looks at data, picks a model, adapts inputs, trains in PyTorch, watches logs, and writes an evaluation report. One UI screen per step.

**Local LLM:** Ollama · Qwen 3.5 4B · `Q4_K_M`

No Azure ML. No Slack. No custom trainers. Use existing libraries only.

---

## Setup

There are two ways to run Loopback:

- **Option A, Docker (recommended).** Three containers, no Python needed on your machine.
- **Option B, local.** A Python venv plus Ollama installed on your machine. Use this for development.

Both serve the same URLs:

| What | URL |
| --- | --- |
| UI | http://localhost:8501 |
| Backend API docs | http://localhost:8100/docs |
| Agent API docs | http://localhost:8000/docs |

### 1. Requirements

| | Minimum | Notes |
| --- | --- | --- |
| RAM | 16 GB | Ollama keeps the model in memory (about 4 GB). |
| Disk | 15 GB free | Model 3.4 GB, Docker images about 6 GB, datasets under 20 MB. |
| GPU | optional | Any NVIDIA GPU with 4 GB or more speeds up the LLM and YOLO training. Without one, everything runs on CPU, just slower. |
| OS | Linux, or Windows 11 with WSL2 | macOS works for CPU only. |
| Network | needed for first setup | Downloads the model, Python packages, datasets, and YOLO weights (on the first detection run). |

On Windows, run every command below inside your WSL2 shell (for example Ubuntu), not PowerShell.

### 2. Get the code

```bash
git clone <this-repo-url> loopback-agent
cd loopback-agent
```

### 3A. Option A: Docker

**1. Install Docker.**
- **Windows:** install Docker Desktop. In *Settings → Resources → WSL integration*, turn integration on for your distro.
- **Linux:** install Docker Engine and the Compose plugin.

Check that Docker works from your shell:

```bash
docker info --format '{{.ServerVersion}}'   # prints a version, not an error
docker compose version
```

If `docker info` says it can't connect to the Docker daemon, Docker isn't running, or WSL integration is off for this distro.

**2. (GPU only) Check that containers can see the GPU.**
- On Linux this needs the NVIDIA Container Toolkit.
- Docker Desktop's WSL2 backend supports GPUs out of the box.
- Rancher Desktop doesn't set up GPU passthrough by default.

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

If this prints your GPU, use the GPU command in step 4. If it fails, use the CPU command.

**3. Download the test datasets.** This runs inside the backend image, so you don't need Python on your machine:

```bash
docker compose build backend
docker compose run --rm --no-deps backend python scripts/fetch_datasets.py   # writes ./datasets
# add --large to also fetch african-wildlife (105 MB)
```

**4. Build and start everything:**

```bash
# CPU only
docker compose up -d --build

# NVIDIA GPU (the LLM and training both use it)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The first start takes a while:
- The images build (mostly PyTorch).
- The `agent` container downloads `qwen3.5:4b-q4_K_M` (3.4 GB) into the `ollama` volume. Later starts skip this.

Follow the download with:

```bash
docker compose logs -f agent      # wait for "Uvicorn running on http://0.0.0.0:8000"
```

**5. Open http://localhost:8501,** then check that everything is up (see [section 4](#4-check-that-it-works)).

**Day-to-day commands:**

```bash
docker compose ps                    # status (agent turns "healthy" once the model is loaded)
docker compose logs -f backend       # training / pipeline logs
docker compose stop                  # stop, keep everything
docker compose up -d                 # start again (no rebuild, no model download)
docker compose up -d --build         # after pulling new code
docker compose down                  # remove containers (the model volume and ./workspace stay)
docker compose down -v               # also delete the downloaded model
```

### 3B. Option B: local, without Docker

**1. Install Python 3.12** and create the venv:

```bash
python3 --version                                   # 3.12.x
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt       # CPU PyTorch + all services + pytest
```

For GPU training, swap in the CUDA build of PyTorch:

```bash
.venv/bin/pip uninstall -y torch torchvision
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True
```

**2. Install Ollama 0.34 or newer and pull the model.** Older versions ignore the JSON schema for Qwen 3.5. The agent still works on them, but more answers fail validation and fall back to rules.

```bash
curl -fsSL https://ollama.com/install.sh | sh        # installs or upgrades (Linux / WSL)
ollama --version                                     # 0.34 or newer
ollama pull qwen3.5:4b-q4_K_M
ollama run qwen3.5:4b-q4_K_M "say ok" --think=false  # smoke test
```

**3. Download the test datasets:**

```bash
.venv/bin/python scripts/fetch_datasets.py           # writes ./datasets (add --large for african-wildlife)
```

**4. Start the harness, backend, and UI together.** Ctrl+C stops all three.

```bash
scripts/dev.sh
```

`dev.sh` expects Ollama on `http://127.0.0.1:11434`. For another host, set `OLLAMA_URL=... scripts/dev.sh`.

Don't run `dev.sh` and the Docker setup at the same time: both use ports 8000, 8100 and 8501.

### 4. Check that it works

```bash
curl -s localhost:8000/health    # {"ok": true, "model": "qwen3.5:4b-q4_K_M", ...}
curl -s localhost:8100/health    # {"ok": true, ..., "agent": {"ok": true, ...}}
```

In the UI sidebar, both dots should be green: **backend** and **agent (Qwen 3.5 4B)**.

If the agent shows *"offline – rule-based fallbacks"*, the whole flow still works, but every agent answer comes from rules. See Troubleshooting.

Automated tests (Option B venv):

```bash
.venv/bin/pytest                                           # offline, about 20 s; LLM steps use rule fallbacks
LIVE_AGENT_URL=http://127.0.0.1:8000 .venv/bin/pytest -m live   # against the running agent
RUN_SLOW=1 .venv/bin/pytest -k detect                      # trains a tiny YOLO (downloads yolov8n.pt)
```

### 5. First run, step by step

1. In the sidebar, click **＋ New** to create a project.
2. **Data:** On *Open datasets*, pick `titanic` and click **Use this dataset**. You can also upload a `.csv` / `.parquet`, or a `.zip` of a YOLO or COCO folder. Check the detected task, target column and target metric, then click **Approve & continue**.
3. **Context:** Add domain notes and constraints: columns never to use, max epochs, "no horizontal flip", device. Click **Ask agent to read the context**, read its summary and risks, then **Approve**.
4. **Suggest:** Click **Suggest models**. You get 1–3 job specs from the fixed zoo, each with a reason. Edit the YAML if you like, then **Approve** one.
5. **Adapt:** Review the plan. For tabular data that's the dropped columns and encodings; for detection it's class renames and merges, `imgsz`, and the val split. Click **Approve & apply** to see the before/after diff, then **Continue to Train**.
6. **Train:** Click **Start training**. The loss and metric charts and the raw log tail update live. Use **Pause / Resume / Stop** as needed.
7. **Watch:** The agent reviews the log every 5 epochs; you can also click **Ask agent to review**. **Approve** a stop or change verdict, or **Reject** it. Approving a change starts a new run.
8. **Eval:** Click **Evaluate on hold-out split**. You get metrics plus a confusion matrix, residual plot, or grid of detection errors. Then click **What next?** The agent either proposes one untried change (approve it to start a new run and go back to Train) or says to finish.
9. **Report:** Click **Generate report**, then **Download report.md**.

The **💬 Ask the agent** panel in the sidebar answers questions about the current project. Its history lives only in your browser session; use **Save chat** / **Restore chat** to keep it.

### 6. Configuration (environment variables)

In Docker, set these under `environment:` in `docker-compose.yml`. Locally, export them before running `scripts/dev.sh`.

| Variable | Service | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_MODEL` | agent | `qwen3.5:4b-q4_K_M` | Ollama model tag. |
| `OLLAMA_URL` | agent | `http://127.0.0.1:11434` | Where the harness reaches Ollama. |
| `OLLAMA_CONTEXT_LENGTH` | agent (Ollama) | `8192` in the container | Context window. Larger uses more VRAM. |
| `OLLAMA_KEEP_ALIVE` | agent (Ollama) | `10m` in the container | How long the model stays loaded between calls. |
| `LLM_RETRIES` | agent | `2` | Retries after a validation failure. |
| `LLM_MAX_TOKENS` | agent | `900` | Cap on tokens per answer. |
| `LLM_TIMEOUT` | agent | `300` | Seconds per LLM call. |
| `AGENT_URL` | backend, frontend | `http://127.0.0.1:8000` (`http://agent:8000` in Docker) | Harness address. |
| `AGENT_TIMEOUT` | backend | `240` | Seconds before a step falls back to rules. |
| `LOOPBACK_WORKSPACE` | backend | `./workspace` (`/workspace` in Docker) | Projects, runs, memory, weights. |
| `LOOPBACK_DATASETS` | backend, fetch script | `./datasets` (`/datasets` in Docker) | Datasets listed on the Data screen. |
| `LOOPBACK_DEVICE` | backend | `auto` | Training device when a spec says `auto`: `auto`, `cpu`, `cuda`. |
| `LOOPBACK_MONITOR_EVERY` | backend | `5` | Epochs between automatic agent reviews. |
| `LOOPBACK_LOG_TAIL` | backend | `12` | Log lines the agent sees. |
| `BACKEND_URL` | frontend | `http://127.0.0.1:8100` (`http://backend:8100` in Docker) | Backend address. |
| `TORCH_INDEX` (build arg) | backend image | CPU wheels | Set by `docker-compose.gpu.yml` to the CUDA 12.6 wheels. |

### 7. Where things are stored

```
workspace/
  memory.db                  SQLite `experiments` table (every spec + result)
  weights/                   cached YOLO weights (yolov8n.pt, ...)
  projects/<project_id>/
    project.json             state of the 8 steps
    data/                    uploaded files
    adapted/                 output of the Adapt step (splits, preprocessor, or YOLO dataset + data.yaml)
  runs/<run_id>/
    job_spec.yaml  train.log  events.jsonl  metrics.json  status.json
    eval.json  verdict.json  report.md  *.png  worker.out  error.txt (on crash)
datasets/                    open test data (scripts/fetch_datasets.py)
```

To start fresh, stop the services and delete `workspace/`.

### 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `docker info`: *cannot connect to the Docker daemon* | Start Docker Desktop. On Windows, also enable WSL integration for your distro (step 3A.1). |
| `could not select device driver "nvidia"` | The GPU isn't visible to Docker (step 3A.2). Use the CPU command instead. |
| Sidebar shows *agent offline* | The model is still downloading (`docker compose logs -f agent`), or Ollama isn't running (`ollama ps`, Option B). The pipeline keeps working on rule fallbacks meanwhile. |
| Agent answers keep saying *rule-based fallback* with *answer rejected* | Expected sometimes: the backend rejected an answer that was ungrounded or invalid. If it happens constantly, check `ollama --version` is 0.34 or newer. |
| Each agent step takes 10–60 s | Normal for a 4B model partly on CPU. With 4 GB of VRAM, some model layers run on CPU. |
| `CUDA out of memory` during training | The LLM and YOLO share the GPU. Set `LOOPBACK_DEVICE=cpu`, pick `cpu` on the Context screen, or lower `batch` / `imgsz` in the job spec. |
| Upload rejected as too large | The Docker UI accepts up to 2 GB. With `dev.sh` the limit is Streamlit's 200 MB: add `--server.maxUploadSize=2000` to the Streamlit line in `scripts/dev.sh`. |
| First detection run fails to download `yolov8n.pt` | It needs internet once. Weights are then cached in `workspace/weights/`. |
| `workspace/` is owned by root after using Docker | `sudo chown -R $USER workspace datasets` |
| Port 8000 / 8100 / 8501 already in use | Another copy is running (`dev.sh` and Docker at the same time, for example). Stop one. |

### Architecture

| Container | Contents | Port |
| --- | --- | --- |
| `agent` | Ollama 0.34.4 + Qwen 3.5 4B `Q4_K_M`, with a **PydanticAI** harness behind FastAPI. It keeps no chat state: the client sends the chat history on every call. | 8000 |
| `backend` | The 8 steps below, training worker processes, SQLite memory, and `runs/`. | 8100 |
| `frontend` | Streamlit stepper. Chat history lives in the browser session and can be saved or restored as JSON. | 8501 |

**Agent endpoints:** `POST /v1/tasks/{data_summary|suggest|adapt_review|monitor|diagnose|next_step|report}` returns typed JSON. `POST /v1/chat` handles chat. `GET /v1/tasks/{task}/schema` returns a task's schema.

**How the 4B model is kept reliable:**
- Every task has a pydantic schema that Ollama enforces during decoding. Values that depend on the request (candidate ids, column names) become per-request enums.
- The model **ranks** options the rules generated. It never writes a spec from scratch.
- Invalid output is sent back to the model with the validation error for a retry.
- The backend checks every answer. A monitor verdict must cite an epoch that exists in the log, a proposed change must be valid for the model and not already tried, and a data claim that quotes a number missing from the data card is dropped.
- Every call has a rule-based fallback, and the UI shows whether the answer came from the LLM or from rules.
- Thinking is off (`reasoning_effort=none`). With it on, the model spent more than 1,500 tokens reasoning before answering.

**LLM-readable logs:** `structlog` writes JSON lines to `events.jsonl`, and `train.log` gets one JSON object per epoch. The LLM reads a digest built from both: a fixed-width table of the last N epochs, precomputed trends (best epoch, epochs since best, slope), the signals found (NaN, dead gradients, plateau, divergence, overfitting, underfitting), and the list of changes already tried.

### Test data (`scripts/fetch_datasets.py`)

| Dataset | Tests |
| --- | --- |
| `breast_cancer` (CSV) | clean binary classification |
| `titanic` (CSV) | nulls, ID columns, free text, high cardinality |
| `diabetes` (Parquet) | regression |
| `coco8` (YOLO) | 80 classes, mostly with zero boxes |
| `medical-pills` (YOLO) | real detection data (115 images) |
| `medical-pills-coco` (COCO) | the same boxes in COCO format, exercising COCO → YOLO |
| `african-wildlife` (YOLO, `--large`) | optional, 105 MB |

---

## Scope

| Allowed | Not in scope |
| --- | --- |
| PyTorch only | TensorFlow, JAX, Keras |
| Tabular data (CSV / Parquet) | Text, audio, video, 3D |
| Images in **YOLO** or **COCO** format | Custom annotation formats |
| Existing recipes: sklearn / Lightning / Ultralytics YOLO / torchvision | New architectures from scratch |
| Local training + Ollama | Cloud jobs, multi-node |

The agent **chooses the model**. The agent **adapts the input**. The user only approves.

---

## What it does

1. **Look at data** — schema, missing values, class balance, image sizes, split leakage.
2. **Understand context** — task type (classification / regression / detection), target metric, domain notes the user typed in.
3. **Suggest a model** from a fixed zoo, with a one-line reason.
4. **Adapt input / output** — column drops, encoding, resize, YOLO↔COCO conversion, class mapping.
5. **Train** in PyTorch. Write logs every step.
6. **Monitor** — LLM reads logs, suggests a change, or early-stops a dead run.
7. **Evaluate** — hold-out metrics + (for detection) missed-box samples.
8. **Report** — LLM writes what was tried, what changed, and the final numbers.

---

## Data

### Tabular
- Files: `.csv`, `.parquet`
- Agent inspects dtypes, nulls, cardinality, target balance.
- Agent builds a sklearn/PyTorch feature pipeline (impute → encode → scale).
- Task inferred: classification if target is categorical, else regression.

### Images (YOLO or COCO only)
- YOLO: `images/` + `labels/*.txt` + `data.yaml`
- COCO: `images/` + `annotations.json`
- Agent checks class counts, box sizes, empty images, train/val overlap.
- Training goes through **Ultralytics YOLO** (PyTorch under the hood). No custom detect heads.

---

## Model zoo (agent picks one)

**Tabular**
- MLP (PyTorch)
- TabNet-style MLP baseline (pre-existing recipe)
- Optional: Logistic / RandomForest as a fast sklearn baseline before the neural net

**Detection**
- YOLOv8n / YOLOv8s (Ultralytics)
- YOLOv11n if already installed

Agent output is a **job spec**, not raw training code:

```yaml
task: detect          # or classify / regress
framework: pytorch
model: yolov8n
data:
  format: yolo
  path: ./data
adapt:
  imgsz: 640
  class_map: [car, ped, bike]
train:
  epochs: 50
  batch: 16
  patience: 10        # early stop
target_metric: mAP50
target_value: 0.85
```

---

## The loop

```
Understand data → Decide (job spec) → Train (PyTorch)
        ↑                                    ↓
   Evaluation report ← Look at errors ← Watch logs / early stop
```

Stops when the target metric is hit, the user stops it, or the agent has no new evidence-based change left.

Rules the agent must follow:
- Every proposed change points at a measured log line or metric.
- Memory stores every job spec + result so the same failed idea is not retried.
- Early stop if loss is NaN, gradients die, or val metric is flat for `patience` epochs.

---

## LLM role (Ollama / Qwen 3.5 4B Q4_K_M)

The model never trains. It only reads structured context and emits JSON / markdown.

| When | LLM does |
| --- | --- |
| After data scan | Summarize the dataset and list 1–3 job specs |
| During training | Read the latest log window; say `continue` / `change X` / `stop` |
| After a failed run | Name the failure mode from the logs |
| After eval | Write the report |

Run locally:

```bash
ollama run qwen3.5:4b-q4_K_M
```

Keep prompts short. Feed only: data card, last N log lines, previous experiments, user target. Ask for JSON back.

---

## Training logs

Every run writes:

```
runs/<run_id>/
  job_spec.yaml
  train.log          # epoch, loss, lr, val metric
  metrics.json
  events.jsonl       # agent decisions (start / change / early_stop)
  report.md          # written after eval
```

Log line format (one JSON object per epoch):

```json
{"epoch": 12, "train_loss": 0.41, "val_metric": 0.71, "lr": 0.001}
```

The LLM is given the tail of `train.log` plus `events.jsonl`. That is its memory of the run.

---

## Evaluation

**Tabular:** accuracy / F1 / RMSE on the hold-out split. Confusion matrix or residual plot saved as an image.

**Detection:** `pycocotools` or Ultralytics val — mAP50, mAP50-95, per-class recall. Save a small grid of false negatives / false positives.

Then the LLM writes `report.md`: data facts, model chosen and why, adaptations applied, training curve in one paragraph, early-stop reason if any, final metrics, what to try next.

---

## UI — one screen per step

Clean, single-purpose pages. No terminal required.

| Step | Screen | User sees | User does |
| --- | --- | --- | --- |
| 1 | **Data** | File drop, detected type (tabular / YOLO / COCO), class balance, warnings | Confirm task + target metric |
| 2 | **Context** | Agent’s reading of the data + domain notes box | Add constraints (“do not flip class X”) |
| 3 | **Suggest** | 1–3 job specs with reasons | Approve one |
| 4 | **Adapt** | Diff of columns / image size / format conversion | Approve or edit |
| 5 | **Train** | Live loss + metric chart, raw log tail | Pause / stop |
| 6 | **Watch** | Agent verdict: continue / change / early stop | Approve the change |
| 7 | **Eval** | Metrics + error samples | — |
| 8 | **Report** | Generated markdown report | Download |

Same layout on every screen: left = facts, right = agent message + Approve / Reject.

---

## Stack (nothing else)

- **UI:** Streamlit or Gradio
- **Agent / LLM:** Ollama + Qwen 3.5 4B `Q4_K_M`
- **Train:** PyTorch · Lightning (tabular) · Ultralytics (YOLO/COCO)
- **Tabular prep:** pandas + sklearn
- **Metrics:** sklearn / pycocotools
- **Memory:** SQLite (`experiments` table)
- **Logs:** files under `runs/`

---

## Out of scope (on purpose)

Cloud compute, Slack approvals, custom model search, AutoML from scratch, multimodal fusion, generating new architectures, anything not PyTorch, any image format that is not YOLO or COCO.
