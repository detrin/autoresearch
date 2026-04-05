import mlflow
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

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

lgbm1 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
lgbm2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=43, n_jobs=-1, verbose=-1)
lgbm3 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.1, num_leaves=31, random_state=44, n_jobs=-1, verbose=-1)
lgbm4 = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=255, min_child_samples=20, subsample=0.7, colsample_bytree=0.6, random_state=45, n_jobs=-1, verbose=-1)
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
et = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42)

models = [lgbm1, lgbm2, lgbm3, lgbm4, rf, et]
weights = [0.2, 0.2, 0.1, 0.2, 0.15, 0.15]

for m in models:
    m.fit(X_train, y_train)

preds = [m.predict_proba(X_val)[:, 1] for m in models]
probs = sum(w * p for w, p in zip(weights, preds))

score = evaluate(y_val, probs)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_1-auc_roc", score)
    mlflow.log_param("model", "Ensemble(4xLGBM+RF+ET)")
    mlflow.log_param("description", "6-model ensemble: 4 lgbm variants + rf + et")
    mlflow.log_param("status", "keep")
