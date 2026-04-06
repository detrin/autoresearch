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

model_lgb = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=255,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
)

model_xgb = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
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
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

preds_lgb = model_lgb.predict_proba(X_val)
preds_xgb = model_xgb.predict_proba(X_val)

score_lgb = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(preds_lgb, axis=1)))
score_xgb = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(preds_xgb, axis=1)))

best_score = 1.0
best_w = 0.6
for w in np.arange(0.5, 1.01, 0.05):
    blend = w * preds_lgb + (1 - w) * preds_xgb
    s = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(blend, axis=1)))
    if s < best_score:
        best_score = s
        best_w = w

preds_ensemble = best_w * preds_lgb + (1 - best_w) * preds_xgb
preds_final = le_target.inverse_transform(np.argmax(preds_ensemble, axis=1))
score = evaluate(val[TARGET], preds_final)
print_results(score)

results = [("lgb", score_lgb), ("xgb", score_xgb), ("ensemble", score)]
for name, s in results:
    print(f"  {name}: {s:.6f}")
print(f"  best_weight_lgb: {best_w:.2f}")

with mlflow.start_run():
    mlflow.log_metric("val_1-f1", score)
    mlflow.log_param("model", "LGB+XGB_ensemble_tuned")
    mlflow.log_param("description", f"LGB(255 leaves) + XGB(depth 8) ensemble, weight search, best_w={best_w:.2f}")
    mlflow.log_param("status", "keep")
