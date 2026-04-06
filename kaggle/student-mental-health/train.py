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

model_lgb1 = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.024,
    num_leaves=231,
    min_child_samples=11,
    subsample=0.92,
    colsample_bytree=0.79,
    reg_alpha=0.33,
    reg_lambda=0.049,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
model_lgb1.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
)

model_lgb2 = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=123,
    verbose=-1,
)
model_lgb2.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
)

model_xgb = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    verbosity=0,
)
model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

preds_lgb1 = model_lgb1.predict_proba(X_val)
preds_lgb2 = model_lgb2.predict_proba(X_val)
preds_xgb = model_xgb.predict_proba(X_val)

score_lgb1 = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(preds_lgb1, axis=1)))
score_lgb2 = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(preds_lgb2, axis=1)))
score_xgb = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(preds_xgb, axis=1)))

best_score = 1.0
best_w1, best_w2 = 0.4, 0.4
for w1 in np.arange(0.3, 0.7, 0.05):
    for w2 in np.arange(0.3, 0.7, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 0.5:
            continue
        blend = w1 * preds_lgb1 + w2 * preds_lgb2 + w3 * preds_xgb
        s = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(blend, axis=1)))
        if s < best_score:
            best_score = s
            best_w1, best_w2 = w1, w2

best_w3 = 1.0 - best_w1 - best_w2
preds_ensemble = best_w1 * preds_lgb1 + best_w2 * preds_lgb2 + best_w3 * preds_xgb
preds_final = le_target.inverse_transform(np.argmax(preds_ensemble, axis=1))
score = evaluate(val[TARGET], preds_final)
print_results(score)

results = [("lgb1", score_lgb1), ("lgb2", score_lgb2), ("xgb", score_xgb), ("ensemble", score)]
for name, s in results:
    print(f"  {name}: {s:.6f}")
print(f"  best_weights: {best_w1:.2f}/{best_w2:.2f}/{best_w3:.2f}")

with mlflow.start_run():
    mlflow.log_metric("val_1-f1", score)
    mlflow.log_param("model", "LGB1+LGB2+XGB_optimized_weights")
    mlflow.log_param("description", f"3-model ensemble: optimized weight search, tuned XGB, best={best_w1:.2f}/{best_w2:.2f}/{best_w3:.2f}")
    mlflow.log_param("status", "keep")
