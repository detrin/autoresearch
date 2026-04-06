import mlflow
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from optbinning import MulticlassOptimalBinning
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

drop_cols = [TARGET]
feature_cols = [c for c in train.columns if c not in drop_cols]

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
    encoders[col] = le

bin_features = ["burnout_score", "mental_health_index", "stress_level", "anxiety_score", "depression_score"]
bin_encoders = {}
for col in bin_features:
    optb = MulticlassOptimalBinning(name=col, max_n_prebins=10, min_prebin_size=0.05, time_limit=30)
    optb.fit(train[col].values, train[TARGET].values)
    train[f"{col}_bin"] = optb.transform(train[col].values, metric="indices")
    val[f"{col}_bin"] = optb.transform(val[col].values, metric="indices")
    bin_encoders[col] = optb

feature_cols_ext = feature_cols + [f"{col}_bin" for col in bin_features]

le_target = LabelEncoder()
y_train = le_target.fit_transform(train[TARGET])
y_val = le_target.transform(val[TARGET])

X_train = train[feature_cols_ext]
X_val = val[feature_cols_ext]

model_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)

preds_lgb = model_lgb.predict(X_val)
score = evaluate(val[TARGET], le_target.inverse_transform(preds_lgb))
print_results(score)

results = [("lgb_binned", score)]

with mlflow.start_run():
    mlflow.log_metric("val_1-f1", score)
    mlflow.log_param("model", "LightGBM+OptimalBinning")
    mlflow.log_param("description", "LightGBM + MulticlassOptimalBinning features on 5 numeric cols")
    mlflow.log_param("status", "keep")
