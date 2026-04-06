import mlflow
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

drop_cols = [TARGET]
feature_cols = [c for c in train.columns if c not in drop_cols]

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
cat_indices = [feature_cols.index(c) for c in cat_cols]

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
y_train = train[TARGET]
y_val = val[TARGET]

for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_val[col] = X_val[col].astype(str)

model_cb = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    cat_features=cat_indices,
    eval_metric="TotalF1",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=0,
)
model_cb.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
)

preds_cb = model_cb.predict(X_val).flatten()
score = evaluate(y_val, preds_cb)
print_results(score)

results = [("cb", score)]

with mlflow.start_run():
    mlflow.log_metric("val_1-f1", score)
    mlflow.log_param("model", "CatBoost")
    mlflow.log_param("description", "CatBoost with raw cat features, TotalF1 eval, early stopping")
    mlflow.log_param("status", "keep")
