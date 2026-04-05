import mlflow
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

train["amount_fee_ratio"] = train["transaction_amount"] / (train["fee_amount"] + 1e-6)
val["amount_fee_ratio"] = val["transaction_amount"] / (val["fee_amount"] + 1e-6)

train["log_amount"] = np.log1p(train["transaction_amount"])
val["log_amount"] = np.log1p(val["transaction_amount"])

train["log_fee"] = np.log1p(train["fee_amount"])
val["log_fee"] = np.log1p(val["fee_amount"])

train["amount_x_fee"] = train["transaction_amount"] * train["fee_amount"]
val["amount_x_fee"] = val["transaction_amount"] * val["fee_amount"]

user_freq = train["user_id"].value_counts().to_dict()
train["user_tx_count"] = train["user_id"].map(user_freq)
val["user_tx_count"] = val["user_id"].map(user_freq).fillna(0)

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

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale = neg_count / pos_count

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    scale_pos_weight=scale,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(X_train, y_train)
probs = model.predict_proba(X_val)[:, 1]

score = evaluate(y_val, probs)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_1-auc_roc", score)
    mlflow.log_param("model", "LGBMClassifier")
    mlflow.log_param("description", "lgbm + feature eng: amount/fee ratio, log amounts, user freq")
    mlflow.log_param("status", "keep")
