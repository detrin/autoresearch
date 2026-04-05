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

global_mean = train[TARGET].mean()
smoothing = 10
zip_stats = train.groupby("zip")[TARGET].agg(["mean", "count", "std"]).reset_index()
zip_stats.columns = ["zip", "zip_te_mean", "zip_te_count", "zip_te_std"]
zip_stats["zip_te_std"] = zip_stats["zip_te_std"].fillna(0)
zip_stats["zip_te_smooth"] = (
    (zip_stats["zip_te_count"] * zip_stats["zip_te_mean"] + smoothing * global_mean)
    / (zip_stats["zip_te_count"] + smoothing)
)
train = train.merge(zip_stats[["zip", "zip_te_smooth", "zip_te_count", "zip_te_std"]], on="zip", how="left")
val = val.merge(zip_stats[["zip", "zip_te_smooth", "zip_te_count", "zip_te_std"]], on="zip", how="left")
val["zip_te_smooth"] = val["zip_te_smooth"].fillna(global_mean)
val["zip_te_count"] = val["zip_te_count"].fillna(0)
val["zip_te_std"] = val["zip_te_std"].fillna(0)

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio",
              "zip_te_smooth", "zip_te_count", "zip_te_std"]
feature_cols = feature_cols + extra_cols

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
    encoders[col] = le

X_train, y_train = train[feature_cols].fillna(-1), train[TARGET]
X_val, y_val = val[feature_cols].fillna(-1), val[TARGET]

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
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return evaluate(y_val, m.predict(X_val))

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, timeout=480)

best_params = study.best_params
best_params["n_estimators"] = 3000
best_params["random_state"] = 42
best_params["verbosity"] = -1

model = lgb.LGBMRegressor(**best_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGBMRegressor-optuna200")
    mlflow.log_param("description", f"optuna 200 trials, zip te, wider search, best_trial={study.best_value:.0f}")
    mlflow.log_param("status", "keep")
    for k, v in best_params.items():
        mlflow.log_param(k, v)
