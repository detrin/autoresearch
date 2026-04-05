import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

def add_features(df):
    df["exp_sq"] = df["experience_years"] ** 2
    df["exp_skills"] = df["experience_years"] * df["skills_count"]
    df["exp_certs"] = df["experience_years"] * df["certifications"]
    df["skills_certs"] = df["skills_count"] * df["certifications"]
    return df

train = add_features(train)
val = add_features(val)

cat_cols = train.select_dtypes(include="object").columns.tolist()
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols])
val[cat_cols] = oe.transform(val[cat_cols])

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

cat_indices = [X_train.columns.tolist().index(c) for c in cat_cols]

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(X_train, y_train, categorical_feature=cat_indices)
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGBMRegressor")
    mlflow.log_param("description", "LGBM 1000 + feature eng (exp_sq, interactions)")
    mlflow.log_param("status", "discard")
