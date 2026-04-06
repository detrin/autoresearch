import mlflow
import numpy as np
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

X_train = train[feature_cols].fillna(-1)
X_val = val[feature_cols].fillna(-1)
y_train_log = np.log1p(train[TARGET])
y_val = val[TARGET]
y_val_log = np.log1p(y_val)

SEEDS = list(range(10))
all_preds_log = []

lgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "num_leaves": 100,
    "max_depth": 13,
    "min_child_samples": 20,
    "subsample": 0.65,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.4,
    "reg_lambda": 10.0,
    "verbosity": -1,
}

xgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "min_child_weight": 20,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "verbosity": 0,
    "tree_method": "hist",
}

cb_params = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "bagging_temperature": 0.5,
    "random_strength": 1.0,
    "verbose": 0,
}

for seed in SEEDS:
    lgb_params["random_state"] = seed
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    all_preds_log.append(m.predict(X_val))

    xgb_params["random_state"] = seed
    m = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    m.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    all_preds_log.append(m.predict(X_val))

    cb_params["random_seed"] = seed
    m = CatBoostRegressor(**cb_params)
    m.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), early_stopping_rounds=100)
    all_preds_log.append(m.predict(X_val))

avg_log_pred = np.mean(all_preds_log, axis=0)
preds = np.expm1(avg_log_pred)

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "30-model-avg")
    mlflow.log_param("description", f"10 seeds x 3 models (LGB+XGB+CB) averaged in log space, baked params")
    mlflow.log_param("status", "keep")
