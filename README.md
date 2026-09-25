# Loopback Agent — Bare Minimum Spec

An agent that looks at data, picks a model, adapts inputs, trains in PyTorch, watches logs, and writes an evaluation report. One UI screen per step.

**Local LLM:** Ollama · Qwen 3.5 4B · `Q4_K_M`

No Azure ML. No Slack. No custom trainers. Use existing libraries only.

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
