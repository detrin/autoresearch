import mlflow
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

drop_cols = [TARGET, "listPrice"]
feature_cols = [c for c in train.columns if c not in drop_cols]

for df in [train, val]:
    df["property_age"] = 2026 - df["year_built"]
    df["sqft_per_bed"] = df["sqft"] / df["beds"].replace(0, np.nan)
    df["sqft_per_bath"] = df["sqft"] / df["baths"].replace(0, np.nan)
    df["total_rooms"] = df["beds"] + df["baths"]
    df["bed_bath_ratio"] = df["beds"] / df["baths"].replace(0, np.nan)
    df["log_sqft"] = np.log1p(df["sqft"])
    df["sqft_x_baths"] = df["sqft"] * df["baths"]

global_mean = train[TARGET].mean()
smoothing = 10

zip_stats = train.groupby("zip").agg(
    zip_te_mean=(TARGET, "mean"),
    zip_te_count=(TARGET, "count"),
    zip_te_std=(TARGET, "std"),
    zip_te_median=(TARGET, "median"),
    zip_sqft_mean=("sqft", "mean"),
    zip_beds_mean=("beds", "mean"),
    zip_age_mean=("property_age", "mean"),
    zip_te_q25=(TARGET, lambda x: x.quantile(0.25)),
    zip_te_q75=(TARGET, lambda x: x.quantile(0.75)),
).reset_index()
zip_stats["zip_te_std"] = zip_stats["zip_te_std"].fillna(0)
zip_stats["zip_te_smooth"] = (
    (zip_stats["zip_te_count"] * zip_stats["zip_te_mean"] + smoothing * global_mean)
    / (zip_stats["zip_te_count"] + smoothing)
)
zip_stats["zip_te_iqr"] = zip_stats["zip_te_q75"] - zip_stats["zip_te_q25"]
zip_stats["zip_price_per_sqft"] = zip_stats["zip_te_mean"] / zip_stats["zip_sqft_mean"].replace(0, np.nan)

merge_cols = [c for c in zip_stats.columns if c != "zip_te_mean"]
train = train.merge(zip_stats[merge_cols], on="zip", how="left")
val = val.merge(zip_stats[merge_cols], on="zip", how="left")
for c in merge_cols[1:]:
    fill = global_mean if any(k in c for k in ["smooth", "median", "q25", "q75"]) else 0
    val[c] = val[c].fillna(fill)

type_stats = train.groupby("type").agg(
    type_te_mean=(TARGET, "mean"),
    type_te_count=(TARGET, "count"),
    type_te_std=(TARGET, "std"),
    type_te_median=(TARGET, "median"),
    type_sqft_mean=("sqft", "mean"),
).reset_index()
type_stats["type_te_std"] = type_stats["type_te_std"].fillna(0)
type_stats["type_te_smooth"] = (
    (type_stats["type_te_count"] * type_stats["type_te_mean"] + smoothing * global_mean)
    / (type_stats["type_te_count"] + smoothing)
)
type_merge = [c for c in type_stats.columns if c != "type_te_mean"]
train = train.merge(type_stats[type_merge], on="type", how="left")
val = val.merge(type_stats[type_merge], on="type", how="left")
for c in type_merge[1:]:
    fill = global_mean if any(k in c for k in ["smooth", "median"]) else 0
    val[c] = val[c].fillna(fill)

for df in [train, val]:
    df["log_zip_te_smooth"] = np.log1p(df["zip_te_smooth"].clip(lower=0))
    df["sqft_vs_zip_avg"] = df["sqft"] / df["zip_sqft_mean"].replace(0, np.nan)
    df["beds_vs_zip_avg"] = df["beds"] / df["zip_beds_mean"].replace(0, np.nan)
    df["age_vs_zip_avg"] = df["property_age"] / df["zip_age_mean"].replace(0, np.nan)
    df["price_per_sqft_est"] = df["zip_price_per_sqft"] * df["sqft"]
    df["sqft_vs_type_avg"] = df["sqft"] / df["type_sqft_mean"].replace(0, np.nan)
    df["log_type_te_smooth"] = np.log1p(df["type_te_smooth"].clip(lower=0))

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio",
              "log_sqft", "sqft_x_baths",
              "zip_te_smooth", "zip_te_count", "zip_te_std", "zip_te_median",
              "zip_te_q25", "zip_te_q75", "zip_te_iqr", "zip_price_per_sqft",
              "log_zip_te_smooth", "zip_sqft_mean", "zip_beds_mean", "zip_age_mean",
              "sqft_vs_zip_avg", "beds_vs_zip_avg", "age_vs_zip_avg", "price_per_sqft_est",
              "type_te_smooth", "type_te_count", "type_te_std", "type_te_median",
              "type_sqft_mean", "sqft_vs_type_avg", "log_type_te_smooth"]
feature_cols = feature_cols + extra_cols

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
    encoders[col] = le

X_train = train[feature_cols].fillna(-1).values
X_val = val[feature_cols].fillna(-1).values
y_train_log = np.log1p(train[TARGET].values)
y_val = val[TARGET].values
y_val_log = np.log1p(y_val)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_lgb(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 150),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
        "random_state": 42,
        "verbosity": -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return evaluate(y_val, np.expm1(m.predict(X_val)))

study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=120, timeout=240)
best_lgb = study_lgb.best_params
best_lgb.update({"n_estimators": 3000, "random_state": 42, "verbosity": -1})

def objective_xgb(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
        "random_state": 42,
        "verbosity": 0,
        "tree_method": "hist",
    }
    m = xgb.XGBRegressor(**params, early_stopping_rounds=100)
    m.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    return evaluate(y_val, np.expm1(m.predict(X_val)))

study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=123))
study_xgb.optimize(objective_xgb, n_trials=80, timeout=180)
best_xgb = study_xgb.best_params
best_xgb.update({"n_estimators": 3000, "random_state": 42, "verbosity": 0, "tree_method": "hist"})

def objective_cb(trial):
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 50.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "random_seed": 42,
        "verbose": 0,
    }
    m = CatBoostRegressor(**params)
    m.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), early_stopping_rounds=100)
    return evaluate(y_val, np.expm1(m.predict(X_val)))

study_cb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=777))
study_cb.optimize(objective_cb, n_trials=50, timeout=180)
best_cb = study_cb.best_params
best_cb.update({"iterations": 3000, "random_seed": 42, "verbose": 0})

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))
val_lgb = np.zeros(len(X_val))
val_xgb = np.zeros(len(X_val))
val_cb = np.zeros(len(X_val))

for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_train)):
    X_tr, X_te = X_train[tr_idx], X_train[te_idx]
    y_tr, y_te = y_train_log[tr_idx], y_train_log[te_idx]

    m1 = lgb.LGBMRegressor(**best_lgb)
    m1.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb[te_idx] = m1.predict(X_te)
    val_lgb += m1.predict(X_val) / 5

    m2 = xgb.XGBRegressor(**best_xgb, early_stopping_rounds=100)
    m2.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    oof_xgb[te_idx] = m2.predict(X_te)
    val_xgb += m2.predict(X_val) / 5

    m3 = CatBoostRegressor(**best_cb)
    m3.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=100)
    oof_cb[te_idx] = m3.predict(X_te)
    val_cb += m3.predict(X_val) / 5

oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cb])
val_stack = np.column_stack([val_lgb, val_xgb, val_cb])

meta = Ridge(alpha=1.0)
meta.fit(oof_stack, y_train_log)
preds_log = meta.predict(val_stack)
preds = np.expm1(preds_log)

score = evaluate(y_val, preds)
print_results(score)

desc = (f"stacking Ridge meta, 5-fold OOF, "
        f"lgb={study_lgb.best_value:.0f} xgb={study_xgb.best_value:.0f} cb={study_cb.best_value:.0f}")

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "Stacking-Ridge-LGB+XGB+CB")
    mlflow.log_param("description", desc[:250])
    mlflow.log_param("status", "keep")
