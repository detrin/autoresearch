import mlflow
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.preprocessing import OrdinalEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

cat_cols = train.select_dtypes(include="object").columns.tolist()
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols])
val[cat_cols] = oe.transform(val[cat_cols])

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

cat_indices = [X_train.columns.tolist().index(c) for c in cat_cols]

sample_idx = np.random.RandomState(42).choice(len(X_train), size=50000, replace=False)
X_sample, y_sample = X_train.iloc[sample_idx], y_train.iloc[sample_idx]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_sample, y_sample, categorical_feature=cat_indices)
    preds = model.predict(X_val)
    return evaluate(y_val, preds)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, timeout=180)

best_params = study.best_params
best_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
model = lgb.LGBMRegressor(**best_params)
model.fit(X_train, y_train, categorical_feature=cat_indices)
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)
print(f"best_params: {best_params}")

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGBMRegressor")
    mlflow.log_param("description", "LGBM Optuna 30 trials subsample tuning")
    mlflow.log_params({k: str(v) for k, v in study.best_params.items()})
    mlflow.log_param("status", "keep")
