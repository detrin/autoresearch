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
│   ├── AGENTS.md
│   ├── deadline.py
│   ├── template/              ← notebook template & kernel-metadata scaffold
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
4. **Run baseline FIRST** — run the unmodified `train.py` before any changes. Log as experiment #0. This is the true baseline all improvements are measured against. Do NOT skip this step.

### Loop (runs until deadline)

1. **Check deadline** — `python kaggle/deadline.py check`. If False, go to Finalize Phase.
2. **Check remaining time** — if <15 minutes remaining, do NOT start Optuna, ensemble tuning, or any long-running experiment. Go to Finalize Phase.
3. **Examine** — MLflow dashboard, `results.tsv`, recent experiment outcomes
4. **Modify `train.py`** — one idea per experiment
5. **Git commit** the change in the worktree
6. **Run:** `timeout 600 python train.py > run.log 2>&1` (ALWAYS use `timeout 600`)
7. **Read results:** grep the metric from `run.log`
8. **Handle crashes:** if no metric found, `tail -n 50 run.log`, attempt fix, give up after 3 tries
9. **Log to MLflow:**
   - `mlflow.set_experiment("<project-name>")`
   - `mlflow.log_metric("val_score", score)`
   - `mlflow.log_param("commit", short_hash)`
   - `mlflow.log_param("description", what_changed)`
   - `mlflow.log_param("status", "keep" | "discard" | "crash")`
   - Timestamp is automatic — MLflow records `start_time` on every run
10. **Log to `results.tsv`** — same data, as lightweight fallback
11. **If score improved:** keep the commit, continue building on it
12. **If score equal or worse:** start next experiment from the last best commit (new worktree or revert in current)

### Mandatory Experiments

The agent MUST run these experiments (in any order) before free exploration:

1. **Baseline** — unmodified `train.py`, no changes
2. **datasci-toolkit experiment** — use `ContinuousOptimalBinning2D` on at least 2 feature pairs, or `OptimalBinner`/`QuantileBinner` for binning. Install with `pip install datasci-toolkit`.

After these two, the agent is free to explore any approach.

### Finalize Phase

When <10 minutes remain OR deadline is reached:

1. **Stop experimenting.** No new ideas.
2. **Identify best commit** from MLflow/results.tsv
3. **Checkout best commit's train.py** and run it one final time to confirm the score
4. **Write structured summary:**
   - Best commit hash and score
   - Improvement over baseline (%)
   - What worked (ranked by impact)
   - What didn't work
   - Ideas not tried (for next session)

## MLflow Logging Requirements

Every experiment run MUST log:

| Field | Type | Notes |
|-------|------|-------|
| `val_score` | metric | **Always `val_score`** — never use the project-specific name from `prepare.py` (e.g. `val_1-f1`, `val_rmse`). Using the project name creates a split column in MLflow where runs can't be compared. |
| `commit` | param | Short git hash |
| `description` | param | One-line description of the change |
| `status` | param | `keep` / `discard` / `crash` |
| `project` | tag | Which dataset/project |
| `branch` | tag | Git branch name |

**Critical:** `prepare.py` defines a project-specific `METRIC_NAME` (e.g. `"1-f1"`) and `print_results` prints `val_1-f1`. Ignore this name when logging to MLflow — always use `"val_score"` as the metric key. This keeps all experiments across all projects comparable in a single column.

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
- **ALWAYS use `timeout 600`** when running `python train.py`. Never run without it.
- **Baseline first.** Run unmodified `train.py` as experiment #0 before any changes.
- **Finalize with 10 min left.** Stop experimenting, confirm best score, write summary.
- **Per-run timeout:** If a single run exceeds 10 minutes, kill it and treat as crash.
- **Simplicity wins.** Equal score + simpler code = improvement.
- **Only modify `train.py`.** Never touch `prepare.py`.
- **One idea per experiment.** Atomic changes make results interpretable.
- **Always log to MLflow.** `results.tsv` is the fallback, MLflow is the source of truth.
- **Try datasci-toolkit.** At least one experiment must use it (binning, 2D features, etc.).
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

### Variable Naming (train.py and notebooks)

Consistent names across all projects:

| Variable | Convention | Example |
|----------|-----------|---------|
| Split data | `X_train`, `X_val`, `y_train`, `y_val` | always |
| Predictions | `preds_{model}` | `preds_lgb`, `preds_xgb`, `preds_cb` |
| Models | `model_{name}` | `model_lgb`, `model_xgb`, `model_cb` |
| Per-model scores | `{metric}_{name}` | `rmse_lgb`, `mape_xgb` |
| Ensemble predictions | `preds_ensemble` | always |
| Results list | `results` | list of `(name, score)` tuples |
| Feature columns | `feature_cols` | always |
| Categorical columns | `cat_cols` | always |
| Drop columns | `drop_cols` | always |

## Publishing

Only after the experiment loop produces a strong result:
1. Find best run in MLflow, note the commit hash
2. Cherry-pick or extract best `train.py` from that branch
3. Create a clean Kaggle notebook from `kaggle/template/notebook_template.ipynb` — fill in project-specific code
4. Execute notebook locally with `jupyter nbconvert --execute` to verify it runs clean
5. Push to Kaggle with `kaggle kernels push`
6. Check status with `kaggle kernels status <user>/<kernel-slug>` — must show `COMPLETE`

### Notebook Template

Use `kaggle/template/notebook_template.ipynb` as the starting point for every published notebook. The template enforces:

1. **Header** (markdown) — title, dataset link, task type, approach summary, best result table
2. **Setup & Data Loading** — imports, `find_file()` helper, load CSVs, print shape
3. **EDA** — target stats, null counts, 2x2 subplot (distribution, by-category, scatter, correlation)
4. **Data Preparation** — train/val split, feature engineering, label encoding
5. **Baseline** — LinearRegression with `.fillna(0)`, always include this
6. **Best Model(s)** — baked hyperparams (no Optuna at runtime), `%%time` on training cells
7. **Ensemble** — weighted blend with grid search, print individual model scores first
8. **Diagnostics** — feature importance bar + predicted vs actual scatter
9. **Results Summary** — DataFrame table + horizontal bar chart
10. **Conclusions** — what worked (ranked), what didn't, takeaway

Rules:
- Always use `find_file()` for Kaggle/local path portability
- Always `fillna(0)` for LinearRegression baseline
- Always `%%time` on model training cells
- Always print `df.info()` or null counts in EDA
- Bake Optuna-tuned params directly — no search at runtime
- Print individual model scores before ensemble so improvement from blending is visible
- Footer: `*Generated via [autoresearch](https://github.com/detrin/autoresearch)*`

### Kaggle Data Paths

Use the standard `find_file()` helper (defined in template):

```python
def find_file(name):
    kaggle_matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if kaggle_matches:
        return kaggle_matches[0]
    if os.path.exists(name):
        return name
    raise FileNotFoundError(f"Cannot find {name}")
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
