import subprocess, mlflow, lightgbm as lgb, xgboost as xgb, pandas as pd, numpy as np
from catboost import CatBoostRegressor
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
    df["cost_per_weight"] = df["cost"] / (df["weight_g"] + 1e-9)
    df["rev_per_weight"] = df["revenue"] / (df["weight_g"] + 1e-9)
    df["profit_proxy"] = df["revenue"] * (1 - df["cost_ratio"])

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

lgb_model = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.05, num_leaves=255, random_state=42, n_jobs=-1,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
lgb_preds = lgb_model.predict(X_val)

lgb_model2 = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=127, random_state=123, n_jobs=-1,
    min_child_samples=50, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5
)
lgb_model2.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
lgb2_preds = lgb_model2.predict(X_val)

xgb_model = xgb.XGBRegressor(
    n_estimators=1500, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, tree_method="hist"
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)
xgb_preds = xgb_model.predict(X_val)

cb_model = CatBoostRegressor(
    iterations=1500, learning_rate=0.05, depth=8, random_seed=42,
    l2_leaf_reg=3.0, subsample=0.8, verbose=500
)
cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cb_preds = cb_model.predict(X_val)

all_preds = [lgb_preds, lgb2_preds, xgb_preds, cb_preds]
names = ["LGB1", "LGB2", "XGB", "CB"]
for n, p in zip(names, all_preds):
    print(f"{n}: {evaluate(y_val, p):.6f}")

best_score = 1.0
best_w = None
for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.05, 0.4, 0.05):
        for w3 in np.arange(0.05, 0.4, 0.05):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < 0.05 or w4 > 0.4:
                continue
            p = w1 * lgb_preds + w2 * lgb2_preds + w3 * xgb_preds + w4 * cb_preds
            s = evaluate(y_val, p)
            if s < best_score:
                best_score = s
                best_w = (w1, w2, w3, w4)

preds = best_w[0] * lgb_preds + best_w[1] * lgb2_preds + best_w[2] * xgb_preds + best_w[3] * cb_preds
score = evaluate(y_val, preds)
print(f"Weights: LGB1={best_w[0]:.2f}, LGB2={best_w[1]:.2f}, XGB={best_w[2]:.2f}, CB={best_w[3]:.2f}")
print_results(score)

commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
description = f"4-model ensemble LGB1+LGB2+XGB+CB w={best_w[0]:.2f}/{best_w[1]:.2f}/{best_w[2]:.2f}/{best_w[3]:.2f}"

with mlflow.start_run():
    mlflow.log_metric("val_mape", score)
    mlflow.log_param("model", "LGB1+LGB2+XGB+CB")
    mlflow.log_param("commit", commit)
    mlflow.log_param("description", description)
    mlflow.log_param("status", "keep")

with open("results.tsv", "a") as f:
    f.write(f"{commit}\t{score:.6f}\tkeep\t{description}\n")
