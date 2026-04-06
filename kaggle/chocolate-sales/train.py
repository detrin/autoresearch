import subprocess, mlflow, lightgbm as lgb, pandas as pd
from sklearn.preprocessing import LabelEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

calendar = pd.read_csv("calendar.csv")
calendar.rename(columns={"date": "order_date"}, inplace=True)
train = train.merge(calendar, on="order_date", how="left")
val = val.merge(calendar, on="order_date", how="left")

train["margin"] = train["revenue"] - train["cost"]
val["margin"] = val["revenue"] - val["cost"]
train["margin_pct"] = train["margin"] / (train["revenue"] + 1e-9)
val["margin_pct"] = val["margin"] / (val["revenue"] + 1e-9)
train["revenue_per_unit"] = train["revenue"] / (train["quantity"] + 1e-9)
val["revenue_per_unit"] = val["revenue"] / (val["quantity"] + 1e-9)
train["cost_per_unit"] = train["cost"] / (train["quantity"] + 1e-9)
val["cost_per_unit"] = val["cost"] / (val["quantity"] + 1e-9)

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

model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=127, random_state=42, n_jobs=-1,
                           min_child_samples=50, subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
description = "LightGBM: +calendar feats, +margin/cost feats, 2000 trees"

with mlflow.start_run():
    mlflow.log_metric("val_mape", score)
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("commit", commit)
    mlflow.log_param("description", description)
    mlflow.log_param("status", "keep")

with open("results.tsv", "a") as f:
    f.write(f"{commit}\t{score:.6f}\tkeep\t{description}\n")
