---
name: mlflow-monitor
description: "Launch and manage MLflow UI for live monitoring of ML training experiments. Use when: (1) the user asks to start, launch, or open MLflow, (2) the user wants live monitoring of a training run, (3) the user asks to view MLflow experiment metrics or compare runs, (4) the user mentions MLflow in the context of monitoring or dashboards."
---

# MLflow Monitor

Launch the MLflow UI to browse and compare training runs logged via `src/mlflow_utils.py`.

## Prerequisites

- `mlflow` installed (`pip install mlflow`)
- Training code integrated with `src/mlflow_utils.py` (logs to `sqlite:///mlflow.db` in project root)

## Launch Steps

### 1. Kill any existing MLflow server on the target port

```bash
fuser -k 5000/tcp 2>/dev/null || true
```

### 2. Start MLflow UI as a persistent background process

```bash
nohup mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --host 0.0.0.0 --allowed-hosts all > /tmp/mlflow_ui.log 2>&1 &
```

Key flags:
- `--backend-store-uri sqlite:///mlflow.db` — SQLite database in project root (required for full UI features)
- `--port 5000` — serve on port 5000
- `--host 0.0.0.0` — accept connections from any IP (required on Lightning AI)
- `nohup ... &` — persist after shell exits

### 3. Verify the server is responding

```bash
sleep 3 && curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/
```

Expected: `200`. If not, check `/tmp/mlflow_ui.log`.

### 4. Report access URL

Tell the user: **http://localhost:5000**

On Lightning AI, the user accesses it via the studio's port-forwarding UI (look for port 5000).

## Stopping

```bash
fuser -k 5000/tcp 2>/dev/null || true
```

## Troubleshooting

- **Port in use**: Run the kill command from step 1, then retry
- **No data in UI**: Confirm `mlflow.db` exists in project root
- **Overview tab not working**: Must use `sqlite:///` URI, not file-based `mlruns/` path
