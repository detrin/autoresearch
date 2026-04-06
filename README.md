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
2. Runs unmodified `train.py` as baseline (experiment #0)
3. Modifies `train.py` with one idea per experiment
4. Runs the experiment, logs to MLflow
5. If score improved → commit and continue. If not → try next idea
6. Repeat until session deadline

## Results Overview

| Project | Task | Best Model | Score | Baseline | Improvement | Experiments | Notebook |
|---------|------|-----------|-------|----------|-------------|-------------|----------|
| Job Salary Prediction | Regression (RMSE) | LightGBM + Optuna | **5,071** | 27,498 | 81.6% | 7 | [Kaggle](https://www.kaggle.com/code/jetakow/job-salary-prediction-autoresearch) |
| Florida Real Estate | Regression (RMSE) | LGB+XGB+CB ensemble | **335,652** | 575,853 | 41.7% | 31 | [Kaggle](https://www.kaggle.com/code/jetakow/florida-real-estate-price-prediction-autoresearch) |
| Chocolate Sales | Regression (MAPE) | 2xLGB+XGB+CB ensemble | **0.00125** | 0.01026 | 87.8% | 9 | [Kaggle](https://www.kaggle.com/code/jetakow/chocolate-sales-profit-prediction-autoresearch) |
| Financial Fraud | Classification (AUC-ROC) | 5-model stacking | **0.8845** | 0.8663 | +2.1% | 13 | [Kaggle](https://www.kaggle.com/code/jetakow/financial-fraud-detection-autoresearch) |
| Student Mental Health | Multiclass (1-F1 macro) | 2x LightGBM ensemble | **0.000673** | 0.001981 | 66.0% | 15 | [Kaggle](https://www.kaggle.com/code/jetakow/student-mental-health-risk-prediction-autoresearch) |

## Datasets

| Project | Target | Rows | Metric | Source |
|---------|--------|------|--------|--------|
| `job-salary-prediction` | `salary` | 250K | RMSE | [Kaggle](https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset) |
| `chocolate-sales` | `profit` | 1M | MAPE | [Kaggle](https://www.kaggle.com/datasets/ssssws/chocolate-sales-dataset-2023-2024) |
| `financial-fraud` | `is_fraud` | 1M | 1 - AUC-ROC | [Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) |
| `florida-real-estate` | `lastSoldPrice` | 10.9K | RMSE | [Kaggle](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026) |
| `student-mental-health` | `risk_level` | 1M | 1 - F1 (macro) | [Kaggle](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout) |

Download via `kaggle datasets download` (see each project's `prepare.py` for the dataset slug).

## Quick Start

```bash
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

Then point your AI coding agent at the repo and tell it to read `kaggle/AGENTS.md` and start the experiment loop.

## Architecture

See [`kaggle/AGENTS.md`](kaggle/AGENTS.md) for full details on the experiment loop, MLflow logging, deadline system, notebook template, and conventions.

---

## Discussion

### Job Salary Prediction

**7 experiments · RMSE 27,498 → 5,071 · 81.6% improvement**

The salary prediction dataset was the first project run through autoresearch. The agent started with a LinearRegression baseline, then immediately switched to tree-based models.

| Experiment | Model | RMSE |
|-----------|-------|------|
| 0 | LinearRegression | 27,498 |
| 1 | RandomForest | 6,533 |
| 2 | HistGradientBoosting | 5,126 |
| 3 | LightGBM defaults | 5,089 |
| 4 | LightGBM + Optuna | **5,071** |

**What worked:** Switching from linear to tree-based models delivered 76% of total improvement in a single step. Optuna confirmed the agent was near the ceiling with marginal further gains.

**What didn't work:** Feature engineering added noise — the dataset is clean with no nulls and limited categorical richness. Further model complexity hit diminishing returns quickly.

**Takeaway:** On clean tabular data with limited feature engineering surface, model selection dominates. The agent correctly converged in 7 experiments.

---

### Florida Real Estate Price Prediction

**31 experiments · RMSE 575,853 → 335,652 · 41.7% improvement**

The most intensive session. The small dataset (10.9K rows) meant fast iteration but also higher overfitting risk.

| Experiment | Model / Feature | RMSE |
|-----------|----------------|------|
| 0 | LinearRegression | 575,853 |
| 1 | LightGBM basic | ~450,000 |
| 2 | + zip-code target encoding | ~390,000 |
| 3 | + log-transformed target | ~365,000 |
| 4 | + relative sqft features | ~350,000 |
| 5 | LGB+XGB+CB ensemble + Optuna | **335,652** |

**What worked:** Zip-code target encoding (smoothed mean/median prices per zip) was the single biggest lever — location signal that raw label-encoded zip codes couldn't provide. Log-transforming the target tamed the heavy right tail. Ensemble blending consistently beat any single model.

**What didn't work:** Raw zip codes as categorical features (too many unique values for 10.9K rows). Polynomial features added noise. The agent also didn't use datasci-toolkit despite explicit instructions — which motivated adding the mandatory experiments section to AGENTS.md.

**Takeaway:** For real estate, location encoding is everything. The 31-experiment count revealed the need for a finalize phase — gains plateaued around experiment 15 but the agent kept running minor variations.

---

### Chocolate Sales Profit Prediction

**9 experiments · MAPE 0.01026 → 0.00125 · 87.8% improvement**

The most dramatic improvement across all projects. The agent discovered that when `profit ≈ revenue - cost`, margin-derived features give the model near-direct signal.

| Experiment | Change | MAPE |
|-----------|--------|------|
| 0 | LightGBM baseline | 0.01026 |
| 1 | Keep IDs + 1000 trees | 0.00859 |
| 2 | Calendar features + margin features | 0.00239 |
| 3 | Interaction features + 255 leaves | 0.00217 |
| 4 | margin_per_unit / margin_per_weight | 0.00160 |
| 5 | LGB + XGB ensemble | 0.00131 |
| 6 | 2xLGB + XGB + CatBoost ensemble | **0.00125** |

**What worked:** Margin features (`revenue - cost`, `margin_pct`, `margin_per_unit`, `margin_per_weight`) were transformative — 3.5x MAPE reduction in one step. Keeping ID columns let gradient boosting learn per-entity patterns implicitly. Calendar features added temporal signal from the separate calendar table.

**What didn't work:** Target encoding on IDs caused overfitting (IDs already served this purpose). Very large trees (511+ leaves) timed out on 800K rows.

**Takeaway:** When the target has a near-algebraic relationship with available features, the right features matter more than the right model. The agent achieved 0.1% MAPE — near-perfect predictions — in just 9 experiments.

---

### Financial Fraud Detection

**13 experiments · AUC-ROC 0.8663 → 0.8845 · +2.1%**

The hardest dataset to move the needle on. Gains were small after the initial model upgrade.

| Experiment | Model | AUC-ROC |
|-----------|-------|---------|
| 0 | Logistic Regression | 0.8663 |
| 1 | LightGBM | ~0.883 |
| 2–10 | Feature engineering / tuning | ~0.884 |
| 11 | 5-model stacking (3xLGB + RF + ET) | **0.8845** |

**What worked:** Tree-based models immediately jumped to 0.883 — the +1.7% gain from model selection was most of the total improvement. Stacking diverse models (LGB + RandomForest + ExtraTrees) with a logistic regression meta-learner squeezed out the final 0.2%.

**What didn't work:** Feature engineering (transaction ratios, time features, merchant aggregations) added noise rather than signal. Target encoding and hyperparameter tuning both converged to the same ~0.884 ceiling.

**Takeaway:** This dataset has a hard ceiling around AUC 0.885 due to its synthetic nature — the fraud signal doesn't respond to feature engineering. Sometimes the most valuable finding is "the data caps out here."

---

### Student Mental Health Risk Prediction

**15 experiments · 1-F1 macro 0.001981 → 0.000673 · 66.0% improvement**

Multiclass classification (Low / Medium / High risk) on 1M rows of student wellbeing and academic data.

| Experiment | Model | 1-F1 macro |
|-----------|-------|-----------|
| 0 | LogisticRegression | 0.001981 |
| 1 | LightGBM | 0.000865 |
| 2 | LGB tuned (127 leaves, lr=0.02) | 0.000855 |
| 3 | LGB + XGB ensemble 0.6/0.4 | 0.000845 |
| 4 | 2xLGB + XGB, weights 0.4/0.4/0.2 | 0.000697 |
| 5 | 2xLGB optimized blend (XGB weight → 0) | **0.000673** |

**What worked:** Switching from LogisticRegression to LightGBM delivered +56% in one step. Two diverse LightGBM configs (231 vs 127 leaves, different regularization and seeds) blended at equal weight outperformed any single model or three-model ensemble — the optimal XGB weight converged to 0, leaving a clean 2-LGB blend.

**What didn't work:** CatBoost (0.005516 — 8x worse than LightGBM, likely from the numeric-heavy feature space). Feature engineering, target encoding, datasci-toolkit binning, and Optuna tuning all failed to improve on the tuned LGB baseline.

**Takeaway:** The dataset's features directly encode mental health signals, so the agent's best move was picking the right model family. Diversity within the same model family (hyperparameter variation) outperformed diversity across model families.

---

### Cross-Project Patterns

Across all five projects, consistent patterns emerged:

1. **Model selection delivers most of the improvement** — switching from linear models to gradient boosting is always the single biggest step (56–81% of total gain)
2. **Feature engineering is high-variance** — transformative for chocolate sales (margin features, 3.5x) and real estate (zip encoding), but added noise for fraud, salary, and mental health
3. **Ensembles reliably add 5–21%** on top of the best single model, but only after features are settled
4. **Diversity beats quantity** — two LGB models with different configs outperformed three models with the same family and different seeds
5. **Synthetic data has hard ceilings** — fraud and mental health datasets converged quickly; the signal is structured enough that near-perfect F1 is achievable but there's no room for clever feature engineering

## License

MIT
