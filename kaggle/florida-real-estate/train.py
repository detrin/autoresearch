import mlflow
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
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

X_train = train[feature_cols].fillna(-1)
X_val = val[feature_cols].fillna(-1)
y_train_log = np.log1p(train[TARGET])
y_val = val[TARGET]
y_val_log = np.log1p(y_val)

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

optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=100, timeout=300)

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
study_xgb.optimize(objective_xgb, n_trials=80, timeout=300)

best_xgb = study_xgb.best_params
best_xgb.update({"n_estimators": 3000, "random_state": 42, "verbosity": 0, "tree_method": "hist"})
model_xgb = xgb.XGBRegressor(**best_xgb, early_stopping_rounds=100)
model_xgb.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
preds_xgb = np.expm1(model_xgb.predict(X_val))

def objective_blend(trial):
    w = trial.suggest_float("lgb_weight", 0.1, 0.9)
    blend = w * preds_lgb + (1 - w) * preds_xgb
    return evaluate(y_val, blend)

study_blend = optuna.create_study(direction="minimize")
study_blend.optimize(objective_blend, n_trials=50)
best_w = study_blend.best_params["lgb_weight"]

preds = best_w * preds_lgb + (1 - best_w) * preds_xgb
score = evaluate(y_val, preds)
print_results(score)

desc = f"lgb+xgb ensemble w={best_w:.2f}, lgb_best={study_lgb.best_value:.0f}, xgb_best={study_xgb.best_value:.0f}"

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGB+XGB-ensemble")
    mlflow.log_param("description", desc)
    mlflow.log_param("status", "keep")
    mlflow.log_param("lgb_weight", best_w)
