import mlflow
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

drop_cols = [TARGET, "transaction_id", "user_id", "organization", "transaction_timestamp"]
feature_cols = [c for c in train.columns if c not in drop_cols]

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()

cat_indices = [feature_cols.index(c) for c in cat_cols]

encoders = {}
train_enc = train.copy()
val_enc = val.copy()
for col in cat_cols:
    le = LabelEncoder()
    train_enc[col] = le.fit_transform(train_enc[col].astype(str))
    val_enc[col] = le.transform(val_enc[col].astype(str))
    encoders[col] = le

X_train_enc, y_train = train_enc[feature_cols], train_enc[TARGET]
X_val_enc, y_val = val_enc[feature_cols], val_enc[TARGET]

X_train_cat = train[feature_cols].copy()
X_val_cat = val[feature_cols].copy()
for col in cat_cols:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_val_cat[col] = X_val_cat[col].astype(str)

lgbm1 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
lgbm2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=43, n_jobs=-1, verbose=-1)
cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8, random_seed=44, verbose=0, cat_features=cat_indices)
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)

lgbm1.fit(X_train_enc, y_train)
lgbm2.fit(X_train_enc, y_train)
cb.fit(X_train_cat, y_train)
rf.fit(X_train_enc, y_train)
et.fit(X_train_enc, y_train)

p_lgbm1 = lgbm1.predict_proba(X_val_enc)[:, 1]
p_lgbm2 = lgbm2.predict_proba(X_val_enc)[:, 1]
p_cb = cb.predict_proba(X_val_cat)[:, 1]
p_rf = rf.predict_proba(X_val_enc)[:, 1]
p_et = et.predict_proba(X_val_enc)[:, 1]

preds = np.array([p_lgbm1, p_lgbm2, p_cb, p_rf, p_et])

def neg_auc(w):
    w = np.abs(w)
    w = w / w.sum()
    blend = (w[:, None] * preds).sum(axis=0)
    return -roc_auc_score(y_val, blend)

res = minimize(neg_auc, x0=np.ones(5) / 5, method="Nelder-Mead")
best_w = np.abs(res.x)
best_w = best_w / best_w.sum()

probs = (best_w[:, None] * preds).sum(axis=0)

score = evaluate(y_val, probs)
print_results(score)
print(f"Optimized weights: {best_w}")

with mlflow.start_run():
    mlflow.log_metric("val_1-auc_roc", score)
    mlflow.log_param("model", "Ensemble(2xLGBM+CatBoost+RF+ET)")
    mlflow.log_param("description", "5-model ensemble with catboost, opt weights")
    mlflow.log_param("status", "keep")
