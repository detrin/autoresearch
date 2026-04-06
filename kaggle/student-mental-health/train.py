import mlflow
import lightgbm as lgb
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
    {"learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 456},
]

all_preds = []
all_scores = []
for i, cfg in enumerate(lgb_configs):
    m = lgb.LGBMClassifier(n_estimators=2000, n_jobs=-1, verbose=-1, **cfg)
    m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
    )
    p = m.predict_proba(X_val)
    s = evaluate(val[TARGET], le_target.inverse_transform(np.argmax(p, axis=1)))
    all_preds.append(p)
    all_scores.append(s)
    print(f"  lgb{i+1}: {s:.6f}")

preds_ensemble = 0.5 * all_preds[0] + 0.25 * all_preds[1] + 0.25 * all_preds[2]
preds_final = le_target.inverse_transform(np.argmax(preds_ensemble, axis=1))
score = evaluate(val[TARGET], preds_final)
print_results(score)

results = [(f"lgb{i+1}", s) for i, s in enumerate(all_scores)] + [("ensemble", score)]
for name, s in results:
    print(f"  {name}: {s:.6f}")

with mlflow.start_run():
    mlflow.log_metric("val_1-f1", score)
    mlflow.log_param("model", "3xLGB_v2_ensemble")
    mlflow.log_param("description", "3 LGB: lgb1 + lgb2(seed123) + lgb2(seed456), weights 0.5/0.25/0.25")
    mlflow.log_param("status", "keep")
