# Autoresearch

Autonomous ML experiment loop for Kaggle datasets. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An AI agent iteratively edits a single `train.py`, runs experiments, logs results to MLflow, and keeps only improvements — all unattended. The human writes strategy, the agent writes code.

## How It Works

```
prepare.py  (read-only)  — data loading, train/val split, evaluation function
train.py    (agent edits) — feature engineering, model, training loop
MLflow      (tracking)    — every experiment logged with metrics, params, timestamps
git worktree (isolation)  — one worktree per experiment, parallel agents possible
```

1. Agent reads `prepare.py` and the dataset
2. Modifies `train.py` with one idea
3. Runs the experiment, logs to MLflow
4. If score improved → commit and continue. If not → try next idea
5. Repeat until interrupted

## Datasets

| Project | Task | Target | Rows | Metric |
|---------|------|--------|------|--------|
| `job-salary-prediction` | Regression | `salary` | 250K | RMSE |
| `chocolate-sales` | Regression | `profit` | 1M | MAPE |
| `financial-fraud` | Binary classification | `is_fraud` | 1M | 1 - AUC-ROC |
| `florida-real-estate` | Regression | `lastSoldPrice` | 10.9K | RMSE |
| `student-mental-health` | Multiclass classification | `risk_level` | 1M | 1 - F1 (macro) |

All datasets sourced from [Kaggle](https://www.kaggle.com). Download via `kaggle datasets download` (see each project's `prepare.py` for the dataset slug).

## Results

### Job Salary Prediction

7 experiments, RMSE **27,498 → 5,071** (81.6% improvement)

| Model | RMSE |
|-------|------|
| LinearRegression (baseline) | 27,498 |
| RandomForest | 6,533 |
| HistGradientBoosting | 5,126 |
| LightGBM + Optuna | **5,071** |

## Quick Start

```bash
# Clone and enter
git clone https://github.com/detrin/autoresearch.git
cd autoresearch

# Download a dataset (requires Kaggle API credentials)
pip install kaggle
kaggle datasets download -d nalisha/job-salary-prediction-dataset -p kaggle/job-salary-prediction --unzip

# Setup project
cd kaggle/job-salary-prediction
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn mlflow lightgbm optuna

# Start MLflow (from repo root, in another terminal)
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5050

# Run baseline
python train.py

# Open MLflow dashboard
open http://localhost:5050
```

Then point your AI coding agent at the repo and tell it to read `kaggle/CLAUDE.md` and start the experiment loop.

## Architecture

See [`kaggle/CLAUDE.md`](kaggle/CLAUDE.md) for full details on the experiment loop, MLflow logging requirements, tooling, and conventions.

## License

MIT
