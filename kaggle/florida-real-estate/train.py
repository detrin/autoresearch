import mlflow
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
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
    df["decade_built"] = (df["year_built"] // 10) * 10
    df["is_condo"] = (df["type"] == "condos").astype(int)
    df["is_single_family"] = (df["type"] == "single_family").astype(int)

global_mean = train[TARGET].mean()
smoothing = 10

def compute_te(train_df, val_df, col, target, smooth, gm):
    stats = train_df.groupby(col)[target].agg(["mean", "count", "std", "median"]).reset_index()
    stats.columns = [col, f"{col}_te_mean", f"{col}_te_count", f"{col}_te_std", f"{col}_te_median"]
    stats[f"{col}_te_std"] = stats[f"{col}_te_std"].fillna(0)
    stats[f"{col}_te_smooth"] = (
        (stats[f"{col}_te_count"] * stats[f"{col}_te_mean"] + smooth * gm)
        / (stats[f"{col}_te_count"] + smooth)
    )
    merge_cols = [c for c in stats.columns if c != f"{col}_te_mean"]
    tr = train_df.merge(stats[merge_cols], on=col, how="left")
    vl = val_df.merge(stats[merge_cols], on=col, how="left")
    for c in merge_cols[1:]:
        fill = gm if any(k in c for k in ["smooth", "median"]) else 0
        vl[c] = vl[c].fillna(fill)
    return tr, vl, merge_cols[1:]

te_cols = []
for col in ["zip", "type", "decade_built"]:
    train, val, new_cols = compute_te(train, val, col, TARGET, smoothing, global_mean)
    te_cols.extend(new_cols)

zip_agg = train.groupby("zip").agg(
    zip_sqft_mean=("sqft", "mean"),
    zip_beds_mean=("beds", "mean"),
    zip_age_mean=("property_age", "mean"),
).reset_index()
train = train.merge(zip_agg, on="zip", how="left")
val = val.merge(zip_agg, on="zip", how="left")
for c in ["zip_sqft_mean", "zip_beds_mean", "zip_age_mean"]:
    val[c] = val[c].fillna(0)

zip_price = train.groupby("zip")[TARGET].agg(["mean"]).reset_index()
zip_price.columns = ["zip", "zip_price_mean"]
zip_sqft_agg = train.groupby("zip")["sqft"].mean().reset_index()
zip_sqft_agg.columns = ["zip", "zip_sqft_mean2"]
zip_ppsqft = zip_price.merge(zip_sqft_agg, on="zip")
zip_ppsqft["zip_price_per_sqft"] = zip_ppsqft["zip_price_mean"] / zip_ppsqft["zip_sqft_mean2"].replace(0, np.nan)
train = train.merge(zip_ppsqft[["zip", "zip_price_per_sqft"]], on="zip", how="left")
val = val.merge(zip_ppsqft[["zip", "zip_price_per_sqft"]], on="zip", how="left")
val["zip_price_per_sqft"] = val["zip_price_per_sqft"].fillna(0)

zip_q = train.groupby("zip")[TARGET].agg([
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.75),
]).reset_index()
zip_q.columns = ["zip", "zip_te_q25", "zip_te_q75"]
zip_q["zip_te_iqr"] = zip_q["zip_te_q75"] - zip_q["zip_te_q25"]
train = train.merge(zip_q, on="zip", how="left")
val = val.merge(zip_q, on="zip", how="left")
for c in ["zip_te_q25", "zip_te_q75", "zip_te_iqr"]:
    val[c] = val[c].fillna(global_mean if "q" in c else 0)

for df in [train, val]:
    df["log_zip_te_smooth"] = np.log1p(df["zip_te_smooth"].clip(lower=0))
    df["sqft_vs_zip_avg"] = df["sqft"] / df["zip_sqft_mean"].replace(0, np.nan)
    df["beds_vs_zip_avg"] = df["beds"] / df["zip_beds_mean"].replace(0, np.nan)
    df["age_vs_zip_avg"] = df["property_age"] / df["zip_age_mean"].replace(0, np.nan)
    df["price_per_sqft_est"] = df["zip_price_per_sqft"] * df["sqft"]
    df["sqft_vs_type_avg"] = df["sqft"] / df["type_sqft_mean"].replace(0, np.nan) if "type_sqft_mean" in df.columns else np.nan
    df["log_type_te_smooth"] = np.log1p(df["type_te_smooth"].clip(lower=0))

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio",
              "log_sqft", "sqft_x_baths", "decade_built", "is_condo", "is_single_family",
              "zip_sqft_mean", "zip_beds_mean", "zip_age_mean", "zip_price_per_sqft",
              "zip_te_q25", "zip_te_q75", "zip_te_iqr",
              "log_zip_te_smooth", "sqft_vs_zip_avg", "beds_vs_zip_avg", "age_vs_zip_avg",
              "price_per_sqft_est", "log_type_te_smooth"] + te_cols

feature_cols = list(dict.fromkeys(feature_cols + extra_cols))

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
    encoders[col] = le

X_train = train[feature_cols].fillna(-1)
X_val = val[feature_cols].fillna(-1)
y_train_log = np.log1p(train[TARGET])
y_val = val[TARGET]
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

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=120, timeout=300)
best_lgb = study_lgb.best_params
best_lgb.update({"n_estimators": 3000, "random_state": 42, "verbosity": -1})
model_lgb = lgb.LGBMRegressor(**best_lgb)
model_lgb.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
preds_lgb = np.expm1(model_lgb.predict(X_val))

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

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=80, timeout=240)
best_xgb = study_xgb.best_params
best_xgb.update({"n_estimators": 3000, "random_state": 42, "verbosity": 0, "tree_method": "hist"})
model_xgb = xgb.XGBRegressor(**best_xgb, early_stopping_rounds=100)
model_xgb.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
preds_xgb = np.expm1(model_xgb.predict(X_val))

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

study_cb = optuna.create_study(direction="minimize")
study_cb.optimize(objective_cb, n_trials=50, timeout=240)
best_cb = study_cb.best_params
best_cb.update({"iterations": 3000, "random_seed": 42, "verbose": 0})
model_cb = CatBoostRegressor(**best_cb)
model_cb.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), early_stopping_rounds=100)
preds_cb = np.expm1(model_cb.predict(X_val))

def objective_blend(trial):
    w1 = trial.suggest_float("lgb_w", 0.0, 1.0)
    w2 = trial.suggest_float("xgb_w", 0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2
    blend = w1 * preds_lgb + w2 * preds_xgb + w3 * preds_cb
    return evaluate(y_val, blend)

study_blend = optuna.create_study(direction="minimize")
study_blend.optimize(objective_blend, n_trials=300)
w1 = study_blend.best_params["lgb_w"]
w2 = study_blend.best_params["xgb_w"]
w3 = 1.0 - w1 - w2

preds = w1 * preds_lgb + w2 * preds_xgb + w3 * preds_cb
preds = np.clip(preds, 0, None)
score = evaluate(y_val, preds)
print_results(score)

desc = (f"3-ens+decade_built_te+is_condo w={w1:.2f}/{w2:.2f}/{w3:.2f}, "
        f"lgb={study_lgb.best_value:.0f} xgb={study_xgb.best_value:.0f} cb={study_cb.best_value:.0f}")

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGB+XGB+CB-decade-te")
    mlflow.log_param("description", desc[:250])
    mlflow.log_param("status", "keep")
