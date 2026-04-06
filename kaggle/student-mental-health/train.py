import mlflow
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

le_target = LabelEncoder()
y_train = le_target.fit_transform(train[TARGET])
y_val = le_target.transform(val[TARGET])

X_train = train[feature_cols]
X_val = val[feature_cols]

lgb_configs = [
    {"learning_rate": 0.024, "num_leaves": 231, "min_child_samples": 11, "subsample": 0.92, "colsample_bytree": 0.79, "reg_alpha": 0.33, "reg_lambda": 0.049, "random_state": 42},
    {"learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 123},
    {"learning_rate": 0.024, "num_leaves": 231, "min_child_samples": 11, "subsample": 0.92, "colsample_bytree": 0.79, "reg_alpha": 0.33, "reg_lambda": 0.049, "random_state": 777},
    {"learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 999},
]

lgb_preds = []
lgb_scores = []
for i, cfg in enumerate(lgb_configs):
    m = lgb.LGBMClassifier(
        n_estimators=2000,
        n_jobs=-1,
        verbose=-1,
        **cfg,
    )
    m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
    )
    p = m.predict_proba(X_val)
    s = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(p, axis=1)))
    lgb_preds.append(p)
    lgb_scores.append(s)
    print(f"  lgb{i+1}: {s:.6f}")

model_xgb = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    verbosity=0,
)
model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
preds_xgb = model_xgb.predict_proba(X_val)
score_xgb = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(preds_xgb, axis=1)))
print(f"  xgb: {score_xgb:.6f}")

lgb_avg = np.mean(lgb_preds, axis=0)
preds_ensemble = 0.8 * lgb_avg + 0.2 * preds_xgb
preds_final = le_target.inverse_transform(np.argmax(preds_ensemble, axis=1))
score = evaluate(val[TARGET], preds_final)
print_results(score)

results = [(f"lgb{i+1}", s) for i, s in enumerate(lgb_scores)] + [("xgb", score_xgb), ("ensemble", score)]
for name, s in results:
    print(f"  {name}: {s:.6f}")

with mlflow.start_run():
    mlflow.log_metric("val_1-f1", score)
    mlflow.log_param("model", "4xLGB+XGB_ensemble")
    mlflow.log_param("description", "4 LGB models (2 configs x 2 seeds) + XGB ensemble, 0.8/0.2")
    mlflow.log_param("status", "keep")
