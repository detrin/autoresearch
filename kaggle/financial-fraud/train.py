import mlflow
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

high_card_cols = ["city", "country"]
for col in high_card_cols:
    means = train.groupby(col)[TARGET].mean()
    global_mean = train[TARGET].mean()
    counts = train.groupby(col)[TARGET].count()
    smoothing = 100
    smooth_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
    train[f"{col}_target_enc"] = train[col].map(smooth_means).fillna(global_mean)
    val[f"{col}_target_enc"] = val[col].map(smooth_means).fillna(global_mean)

drop_cols = [TARGET, "transaction_id", "user_id", "organization", "transaction_timestamp"]
feature_cols = [c for c in train.columns if c not in drop_cols]

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
    encoders[col] = le

X_train, y_train = train[feature_cols], train[TARGET]
X_val, y_val = val[feature_cols], val[TARGET]

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
probs = model.predict_proba(X_val)[:, 1]

score = evaluate(y_val, probs)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_1-auc_roc", score)
    mlflow.log_param("model", "LGBMClassifier")
    mlflow.log_param("description", "lgbm + target encoding city/country")
    mlflow.log_param("status", "keep")
