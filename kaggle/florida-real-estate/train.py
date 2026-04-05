import mlflow
import numpy as np
import optuna
import lightgbm as lgb
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
    df["sqft_x_baths"] = df["sqft"] * df["baths"]
    df["log_sqft"] = np.log1p(df["sqft"])

global_mean = train[TARGET].mean()
smoothing = 10

for grp_col in ["zip", "type"]:
    stats = train.groupby(grp_col)[TARGET].agg(["mean", "count", "std", "median"]).reset_index()
    stats.columns = [grp_col, f"{grp_col}_te_mean", f"{grp_col}_te_count", f"{grp_col}_te_std", f"{grp_col}_te_median"]
    stats[f"{grp_col}_te_std"] = stats[f"{grp_col}_te_std"].fillna(0)
    stats[f"{grp_col}_te_smooth"] = (
        (stats[f"{grp_col}_te_count"] * stats[f"{grp_col}_te_mean"] + smoothing * global_mean)
        / (stats[f"{grp_col}_te_count"] + smoothing)
    )
    merge_cols = [grp_col, f"{grp_col}_te_smooth", f"{grp_col}_te_count", f"{grp_col}_te_std", f"{grp_col}_te_median"]
    train = train.merge(stats[merge_cols], on=grp_col, how="left")
    val = val.merge(stats[merge_cols], on=grp_col, how="left")
    for c in merge_cols[1:]:
        fill = global_mean if "smooth" in c or "median" in c else 0
        val[c] = val[c].fillna(fill)

zip_type_stats = train.groupby(["zip", "type"])[TARGET].agg(["mean", "count"]).reset_index()
zip_type_stats.columns = ["zip", "type", "zip_type_te_mean", "zip_type_te_count"]
zip_type_stats["zip_type_te_smooth"] = (
    (zip_type_stats["zip_type_te_count"] * zip_type_stats["zip_type_te_mean"] + smoothing * global_mean)
    / (zip_type_stats["zip_type_te_count"] + smoothing)
)
train = train.merge(zip_type_stats[["zip", "type", "zip_type_te_smooth"]], on=["zip", "type"], how="left")
val = val.merge(zip_type_stats[["zip", "type", "zip_type_te_smooth"]], on=["zip", "type"], how="left")
val["zip_type_te_smooth"] = val["zip_type_te_smooth"].fillna(global_mean)

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio",
              "sqft_x_baths", "log_sqft",
              "zip_te_smooth", "zip_te_count", "zip_te_std", "zip_te_median",
              "type_te_smooth", "type_te_count", "type_te_std", "type_te_median",
              "zip_type_te_smooth"]
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

def objective(trial):
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
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": 42,
        "verbosity": -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_train, y_train_log, eval_set=[(X_val, np.log1p(y_val))],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    preds = np.expm1(m.predict(X_val))
    return evaluate(y_val, preds)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=150, timeout=480)

best_params = study.best_params
best_params["n_estimators"] = 3000
best_params["random_state"] = 42
best_params["verbosity"] = -1

model = lgb.LGBMRegressor(**best_params)
model.fit(X_train, y_train_log, eval_set=[(X_val, np.log1p(y_val))],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
preds = np.expm1(model.predict(X_val))

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGBMRegressor-log-ext-feats")
    mlflow.log_param("description", f"log target + zip+type+zip_type te + interactions, optuna 150, best={study.best_value:.0f}")
    mlflow.log_param("status", "keep")
    for k, v in best_params.items():
        mlflow.log_param(k, v)
