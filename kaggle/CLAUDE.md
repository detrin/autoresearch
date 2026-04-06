# Autoresearch — Kaggle Benchmarks

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Repo: [detrin/autoresearch](https://github.com/detrin/autoresearch).
Agent autonomously runs ML experiments in a loop. Human writes strategy, agent writes code.

## Architecture

### Per Project — Three Files

1. **`prepare.py`** (READ-ONLY) — Data loading, train/val split, evaluation function, fixed constants. Agent cannot modify this.
2. **`train.py`** (AGENT EDITS THIS) — The single file the agent modifies. Feature engineering, model, training loop. Everything fair game.
3. **`results.tsv`** (LOCAL) — Lightweight fallback log. Tab-separated: `commit`, `val_score`, `status`, `description`.

### Repo Root — MLflow Tracking Server

```
autoresearch/
├── kaggle/
│   ├── CLAUDE.md
│   ├── job-salary-prediction/
│   ├── chocolate-sales/
│   ├── financial-fraud/
│   ├── florida-real-estate/
│   └── student-mental-health/
├── mlflow/                  ← MLflow backend store
│   └── mlflow.db            ← SQLite database
├── mlruns/                  ← MLflow artifact store
└── .worktrees/              ← git worktrees live here
```

Start MLflow server from repo root:
```bash
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5050
```

All agents connect to `http://localhost:5050`.

### Git Worktrees — One Per Experiment

Each experiment runs in an isolated worktree. No resets, no reverts, no lost code.

```bash
git worktree add .worktrees/<project>-<experiment-name> -b autoresearch/<project>/<experiment-name>
```

Worktrees let multiple agents run experiments **in parallel** on different branches. Main branch stays clean.

## The Experiment Loop

### Setup Phase

1. Pick a project and agree on a run tag (e.g., `apr5-salary`)
2. Read all in-scope files for full context (`prepare.py`, `train.py`, data samples)
3. Create worktree: `git worktree add .worktrees/<tag> -b autoresearch/<tag>`
4. Run baseline (unmodified `train.py`) and log to both MLflow and `results.tsv`

### Loop (runs until interrupted)

1. **Examine** — MLflow dashboard, `results.tsv`, recent experiment outcomes
2. **Modify `train.py`** — one idea per experiment
3. **Git commit** the change in the worktree
4. **Run:** `python train.py > run.log 2>&1`
5. **Read results:** grep the metric from `run.log`
6. **Handle crashes:** if no metric found, `tail -n 50 run.log`, attempt fix, give up after 3 tries
7. **Log to MLflow:**
   - `mlflow.set_experiment("<project-name>")`
   - `mlflow.log_metric("val_score", score)`
   - `mlflow.log_param("commit", short_hash)`
   - `mlflow.log_param("description", what_changed)`
   - `mlflow.log_param("status", "keep" | "discard" | "crash")`
   - Timestamp is automatic — MLflow records `start_time` on every run
8. **Log to `results.tsv`** — same data, as lightweight fallback
9. **If score improved:** keep the commit, continue building on it
10. **If score equal or worse:** start next experiment from the last best commit (new worktree or revert in current)

## MLflow Logging Requirements

Every experiment run MUST log:

| Field | Type | Notes |
|-------|------|-------|
| `val_score` | metric | Primary evaluation metric |
| `commit` | param | Short git hash |
| `description` | param | One-line description of the change |
| `status` | param | `keep` / `discard` / `crash` |
| `project` | tag | Which dataset/project |
| `branch` | tag | Git branch name |

Timestamp is recorded automatically by MLflow (`start_time`, `end_time`). This lets us track optimization velocity — how quickly score improves over wall-clock time.

## Session Deadline

Every experiment session has a **hard deadline** enforced by `kaggle/deadline.py`. The human sets it before launching the agent.

### How it works

1. **Human sets deadline before agent launch:**
   ```bash
   python kaggle/deadline.py set 90   # 90-minute session
   ```

2. **Agent checks before EVERY experiment:**
   ```bash
   python kaggle/deadline.py check    # exit code 0 = keep going, 1 = stop
   ```
   Or from Python inside the experiment loop:
   ```python
   import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
   from deadline import check_deadline, minutes_remaining
   if not check_deadline():
       # stop loop, summarize results, exit
   ```

3. **Agent MUST stop when deadline is reached.** No "one more experiment." Log final results, summarize what worked, and exit cleanly.

4. **Human cleans up after:**
   ```bash
   python kaggle/deadline.py clear
   ```

### Per-run timeout

Individual `python train.py` runs still have a 10-minute timeout. Use the Bash tool's `timeout` parameter or shell `timeout 600`:
```bash
timeout 600 python train.py > run.log 2>&1
```

## Critical Rules

- **NEVER STOP** unless deadline is reached. Do not pause to ask the human. Run until deadline.
- **CHECK DEADLINE** before every experiment iteration. If `check_deadline()` returns False, stop immediately.
- **Per-run timeout:** If a single run exceeds 10 minutes, kill it and treat as crash.
- **Simplicity wins.** Equal score + simpler code = improvement.
- **Only modify `train.py`.** Never touch `prepare.py`.
- **One idea per experiment.** Atomic changes make results interpretable.
- **Always log to MLflow.** `results.tsv` is the fallback, MLflow is the source of truth.
- **When stuck:** re-read `prepare.py` and data, check MLflow for patterns in what worked/failed, try combining near-misses, try radical approaches.

## Projects

Each subfolder is independent with its own `pyproject.toml` and `.venv`.

| Project | Task | Target | Rows | Metric (lower=better) |
|---------|------|--------|------|-----------------------|
| `job-salary-prediction/` | Regression | `salary` | 250K | RMSE |
| `chocolate-sales/` | Regression | `profit` | 1M | MAPE |
| `financial-fraud/` | Binary classification | `is_fraud` | 1M | 1 - AUC-ROC |
| `florida-real-estate/` | Regression | `lastSoldPrice` | 10.9K | RMSE |
| `student-mental-health/` | Multiclass classification | `risk_level` | 1M | 1 - F1 (macro) |

## Tooling

The agent is free to install and use any library that helps improve the score. The following are recommended starting points, not restrictions:

| Library | Use case |
|---------|----------|
| `datasci-toolkit` (`detrin/datasci-toolkit`) | Preprocessing, feature engineering utilities |
| `optbinning` | Optimal binning for numerical/categorical features |
| `lightgbm` | Gradient boosting (fast, strong baseline) |
| `optuna` | Hyperparameter optimization |
| `shap` | Feature importance and model interpretability |

These are listed as `[project.optional-dependencies] recommended` in each `pyproject.toml`. To install: `pip install -e ".[recommended]"`.

The agent MAY install additional packages (xgboost, catboost, pytorch, feature-engine, category_encoders, etc.) if it believes they will improve the score. Add new deps to `pyproject.toml` before importing. The only constraint: never modify `prepare.py`.

## Conventions

- Each project: own `pyproject.toml`, own `.venv`
- No comments in code
- DRY — shared utilities go in top-level `autoresearch/` if needed
- Activate project venv before any work: `source <project>/.venv/bin/activate`
- All experiments comparable within a project (same data split, same eval function)
- Worktrees cleaned up after experiments are analyzed: `git worktree remove .worktrees/<tag>`

## Publishing

Only after the experiment loop produces a strong result:
1. Find best run in MLflow, note the commit hash
2. Cherry-pick or extract best `train.py` from that branch
3. Create a clean Kaggle notebook combining `prepare.py` + `train.py`
4. Execute notebook locally with `jupyter nbconvert --execute` to verify it runs clean
5. Push to Kaggle with `kaggle kernels push`
6. Check status with `kaggle kernels status <user>/<kernel-slug>` — must show `COMPLETE`

### Kaggle Data Paths

On Kaggle, datasets are mounted at `/kaggle/input/<dataset-slug>/`. Notebooks must handle both local and Kaggle paths:

```python
import os

DATA_FILE = "my_dataset.csv"
if os.path.exists(f"/kaggle/input/<dataset-slug>/{DATA_FILE}"):
    DATA_PATH = f"/kaggle/input/<dataset-slug>/{DATA_FILE}"
else:
    DATA_PATH = DATA_FILE
```

### Kaggle Kernel Metadata

Each project needs a `kernel-metadata.json` for pushing:

```json
{
  "id": "<kaggle-user>/<kernel-slug>",
  "title": "<Kernel Title>",
  "code_file": "notebook_executed.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["<owner>/<dataset-slug>"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

## Tooling Tradeoffs

- **Use Python 3.12+.** Python 3.9 on macOS causes SSL warnings, async issues, and jupyter timeout bugs. Kaggle runs 3.12.
- **MLflow must use port 5050.** Port 5000 conflicts with macOS Control Center.
- Large CSVs must be excluded from git history, not just `.gitignore` — `git filter-branch` needed after the fact.
- Optuna searches don't belong in published notebooks — bake the best params in directly. The search is the research; the notebook is the report.
- Use `glob.glob("/kaggle/input/**/<file>", recursive=True)` for robust Kaggle path detection.
