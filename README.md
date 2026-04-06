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
| Student Mental Health | Multiclass (F1 macro) | — | — | — | — | — | *pending* |

## Datasets

| Project | Target | Rows | Metric | Source |
|---------|--------|------|--------|--------|
| `job-salary-prediction` | `salary` | 250K | RMSE | [Kaggle](https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset) |
| `chocolate-sales` | `profit` | 1M | MAPE | [Kaggle](https://www.kaggle.com/datasets/ssssws/chocolate-sales-dataset-2023-2024) |
| `financial-fraud` | `is_fraud` | 1M | 1 - AUC-ROC | [Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) |
| `florida-real-estate` | `lastSoldPrice` | 10.9K | RMSE | [Kaggle](https://www.kaggle.com/datasets/kanchana1990/florida-real-estate-sold-dataset-2026) |
| `student-mental-health` | `risk_level` | 1M | 1 - F1 (macro) | [Kaggle](https://www.kaggle.com/datasets/yashkhare2004/students-mental-health-and-academic-performance) |

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

**7 experiments, RMSE 27,498 → 5,071 (81.6% improvement)**

The salary prediction dataset was the first project run through autoresearch. The agent started with a LinearRegression baseline at RMSE 27,498, then immediately jumped to tree-based models.

**What the agent tried:**
- LinearRegression baseline → RMSE 27,498
- RandomForest → RMSE 6,533 (76% jump in a single step)
- HistGradientBoosting → RMSE 5,126
- LightGBM with default params → RMSE 5,089
- LightGBM + Optuna hyperparameter search → RMSE 5,071

**What worked:**
- Switching from linear to tree-based models was the single biggest lever — 76% of total improvement came from one model swap
- Optuna hyperparameter tuning provided marginal gains (<1% per iteration) but confirmed the agent was near the ceiling

**What didn't work:**
- Feature engineering attempts added noise — the dataset was clean with no nulls and limited categorical richness
- Further model complexity (deeper trees, more estimators) hit diminishing returns quickly

**Takeaway:** On clean tabular data with limited feature engineering surface, model selection dominates. The agent correctly converged in 7 experiments — most of the value came from experiment #2 (RandomForest).

---

### Florida Real Estate Price Prediction

**31 experiments, RMSE 575,853 → 335,652 (41.7% improvement)**

The most intensive session — 31 experiments over a 90-minute deadline. The small dataset (10.9K rows) meant fast iteration but also higher overfitting risk.

**What the agent tried:**
- LinearRegression baseline → RMSE 575,853
- LightGBM with basic features → RMSE ~450K
- Zip-code target encoding (smoothed means, medians, quantiles, price/sqft) → big drop
- Log-transformed target to handle right-skewed price distribution
- Property-type target encoding and relative features (sqft vs zip average)
- 3-model ensemble (LightGBM + XGBoost + CatBoost) with Optuna-tuned params
- Weighted blend optimization → RMSE 335,652

**What worked:**
- **Zip-code target encoding** was the single biggest lever — smoothed mean/median prices per zip gave the model location signal that raw zip codes couldn't provide
- **Log-transformed target** handled the heavy right tail in Florida housing prices, stabilizing predictions for both modest homes and mansions
- **Ensemble blending** consistently beat any single model — the diversity of LGB/XGB/CatBoost captured different patterns
- **Relative features** (e.g., sqft relative to zip average) added incremental value by capturing "above/below average for the area"

**What didn't work:**
- Raw zip codes as categorical features — too many unique values for LabelEncoder to be useful with only 10.9K rows
- datasci-toolkit binning — the agent didn't use it despite explicit instructions (led to adding mandatory experiment requirements in AGENTS.md)
- Polynomial features on numeric columns — added noise on this small dataset

**Takeaway:** For real estate, location encoding is everything. Target encoding with proper smoothing (to avoid leakage on small datasets) unlocked most of the improvement. The 31-experiment count also revealed the need for a deadline finalize phase — the agent kept running minor variations instead of stopping when gains plateaued.

---

### Chocolate Sales Profit Prediction

**9 experiments, MAPE 0.01026 → 0.00125 (87.8% improvement)**

The most dramatic improvement of any project. The agent discovered that when `profit ≈ revenue - cost`, margin-derived features give the model near-direct signal.

**What the agent tried:**
- LightGBM baseline (500 trees, 63 leaves) → MAPE 0.01026
- Keep ID columns + increase to 1000 trees → MAPE 0.00859 (+16%)
- Calendar features (merge order_date with calendar table) + margin features → MAPE 0.00239 (+77%)
- Interaction features + larger trees (255 leaves) → MAPE 0.00217
- Target encoding on IDs/categories → MAPE 0.00221 (worse)
- margin_per_unit and margin_per_weight → MAPE 0.00160
- 2-model ensemble (LGB + XGB) → MAPE 0.00131
- 4-model ensemble (2xLGB + XGB + CatBoost) → MAPE 0.00125

**What worked:**
- **Margin features** (`revenue - cost`, `margin_pct`, `margin_per_unit`, `margin_per_weight`) were transformative — 3.5x MAPE reduction in a single experiment. This is the clearest example of domain-aware feature engineering beating model complexity.
- **Keeping ID columns** (product_id, store_id, customer_id) let gradient boosting learn per-entity patterns — essentially implicit target encoding
- **Calendar features** from the separate calendar table added temporal signal the agent found by exploring the multi-table schema
- **4-model ensemble** with deliberately diverse hyperparameters squeezed out the last 21% improvement

**What didn't work:**
- **Target encoding** on IDs/categories — caused overfitting, slightly worse than label encoding. The ID columns already served this purpose implicitly.
- **Very large trees** (511+ leaves, 4000 estimators) — timeout issues on 800K training rows

**Takeaway:** When the target has a near-algebraic relationship with available features (`profit ≈ revenue - cost`), the right features matter more than the right model. The agent achieved 0.1% MAPE — near-perfect predictions — in just 9 experiments because it found the margin feature early.

---

### Financial Fraud Detection

**13 experiments, AUC-ROC 0.8663 → 0.8845 (+2.1%)**

The hardest dataset to improve on. The agent ran 13 experiments but gains were small after the initial model upgrade.

**What the agent tried:**
- Logistic Regression baseline → AUC-ROC 0.8663
- LightGBM → AUC-ROC ~0.883 (immediate jump)
- Feature engineering (ratios, interactions, aggregations) → no measurable gain
- Target encoding on merchant/category → no gain
- Various LightGBM hyperparameter configs → ~0.884
- 5-model stacking ensemble (3x LightGBM + RandomForest + ExtraTrees) → AUC-ROC 0.8845

**What worked:**
- **Tree-based models** immediately jumped to 0.883 AUC — the +1.7% gain from model selection was most of the total improvement
- **Stacking diverse models** (LGB + RF + ExtraTrees) with a logistic regression meta-learner squeezed out the final 0.2%

**What didn't work:**
- **Feature engineering** — every attempt (transaction ratios, time features, merchant aggregations) added noise rather than signal
- **Target encoding** — no improvement, likely because the synthetic data didn't have real merchant/category patterns
- **Hyperparameter tuning** — all tree-based approaches converged to ~0.884 regardless of settings

**Takeaway:** This dataset likely has a hard ceiling around AUC 0.885 due to its synthetic nature — the fraud signal is embedded in ways that don't respond to feature engineering. The agent correctly identified this plateau and shifted to ensembling as the only remaining lever. Sometimes the most valuable finding is "the data caps out here."

---

### Cross-Project Patterns

Across all four completed projects, consistent patterns emerged:

1. **Model selection delivers 70-80% of total improvement** — switching from linear models to gradient boosting is always the biggest single step
2. **Feature engineering is high-variance** — margin features transformed chocolate sales (3.5x), zip encoding transformed real estate, but nothing helped fraud detection. Domain knowledge determines whether FE has a surface to work with.
3. **Ensembles reliably add 5-20%** on top of the best single model — but only after feature engineering has been explored
4. **Hyperparameter tuning has diminishing returns** — Optuna finds marginal gains after model architecture and features are set
5. **The agent's biggest weakness is knowing when to stop** — the Florida session ran 31 experiments where gains plateaued after ~15, motivating the deadline finalize phase

## License

MIT
