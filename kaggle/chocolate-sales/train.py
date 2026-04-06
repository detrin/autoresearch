import subprocess, mlflow, lightgbm as lgb, pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

calendar = pd.read_csv("calendar.csv")
calendar.rename(columns={"date": "order_date"}, inplace=True)
train = train.merge(calendar, on="order_date", how="left")
val = val.merge(calendar, on="order_date", how="left")

for df in [train, val]:
    df["margin"] = df["revenue"] - df["cost"]
    df["margin_pct"] = df["margin"] / (df["revenue"] + 1e-9)
    df["revenue_per_unit"] = df["revenue"] / (df["quantity"] + 1e-9)
    df["cost_per_unit"] = df["cost"] / (df["quantity"] + 1e-9)
    df["cost_ratio"] = df["cost"] / (df["revenue"] + 1e-9)
    df["disc_revenue"] = df["quantity"] * df["unit_price"]
    df["disc_amount"] = df["disc_revenue"] - df["revenue"]
    df["price_x_qty"] = df["unit_price"] * df["quantity"]
    df["weight_x_qty"] = df["weight_g"] * df["quantity"]
    df["cocoa_x_price"] = df["cocoa_percent"] * df["unit_price"]
    df["margin_per_unit"] = df["margin"] / (df["quantity"] + 1e-9)
    df["margin_per_weight"] = df["margin"] / (df["weight_g"] + 1e-9)
    df["discount_flag"] = (df["discount"] > 0).astype(int)

drop_cols = [TARGET, "order_id", "order_date"]
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

model = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.05, num_leaves=255, random_state=42, n_jobs=-1,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
description = "LightGBM: 255 leaves, +margin_per_unit/weight, discount_flag"

with mlflow.start_run():
    mlflow.log_metric("val_mape", score)
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("commit", commit)
    mlflow.log_param("description", description)
    mlflow.log_param("status", "keep")

with open("results.tsv", "a") as f:
    f.write(f"{commit}\t{score:.6f}\tkeep\t{description}\n")
