import mlflow
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import TargetEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

cat_cols = train.select_dtypes(include="object").columns.tolist()
te = TargetEncoder(smooth="auto", random_state=42)
train[cat_cols] = te.fit_transform(train[cat_cols], train[TARGET])
val[cat_cols] = te.transform(val[cat_cols])

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

params = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 127,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_samples": 20,
    "reg_alpha": 1e-5,
    "reg_lambda": 1e-3,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGBMRegressor")
    mlflow.log_param("description", "LGBM + TargetEncoder (moderate params)")
    mlflow.log_params({k: str(v) for k, v in params.items()})
    mlflow.log_param("status", "keep")
