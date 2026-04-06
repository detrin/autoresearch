import mlflow
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

def add_features(df):
    df = df.copy()
    df["anxiety_x_depression"] = df["anxiety_score"] * df["depression_score"]
    df["stress_x_exam"] = df["stress_level"] * df["exam_pressure"]
    df["mental_burnout_ratio"] = df["mental_health_index"] / (df["burnout_score"] + 1)
    df["sleep_stress_ratio"] = df["sleep_hours"] / (df["stress_level"] + 1)
    df["social_anxiety_diff"] = df["social_support"] - df["anxiety_score"]
    df["total_distress"] = df["anxiety_score"] + df["depression_score"] + df["stress_level"]
    df["academic_pressure"] = df["exam_pressure"] * df["family_expectation"]
    df["wellbeing"] = df["sleep_hours"] + df["physical_activity"] + df["social_support"]
    return df

train = add_features(train)
val = add_features(val)

drop_cols = [TARGET]
feature_cols = [c for c in train.columns if c not in drop_cols]

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
    encoders[col] = le

le_target = LabelEncoder()
y_train = le_target.fit_transform(train[TARGET])
y_val = le_target.transform(val[TARGET])

X_train = train[feature_cols]
X_val = val[feature_cols]

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

results = [("lgb_feat_eng", score)]

with mlflow.start_run():
    mlflow.log_metric("val_1-f1", score)
    mlflow.log_param("model", "LightGBM+FeatEng")
    mlflow.log_param("description", "LightGBM + feature engineering: interactions, ratios, composite scores")
    mlflow.log_param("status", "keep")
