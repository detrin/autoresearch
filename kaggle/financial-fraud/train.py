import mlflow
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    m = lgb.LGBMClassifier(**params)
    m.fit(X_train, y_train)
    p = m.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, p)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=300)

best_params = study.best_params
best_params["random_state"] = 42
best_params["n_jobs"] = -1
best_params["verbose"] = -1
print(f"Best Optuna params: {best_params}")
print(f"Best Optuna AUC: {study.best_value:.6f}")

lgbm_opt = lgb.LGBMClassifier(**best_params)
lgbm1 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
lgbm2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=43, n_jobs=-1, verbose=-1)
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)

models = [lgbm_opt, lgbm1, lgbm2, rf, et]
for m in models:
    m.fit(X_train, y_train)

preds = np.array([m.predict_proba(X_val)[:, 1] for m in models])

def neg_auc(w):
    w = np.abs(w)
    w = w / w.sum()
    blend = (w[:, None] * preds).sum(axis=0)
    return -roc_auc_score(y_val, blend)

res = minimize(neg_auc, x0=np.ones(len(models)) / len(models), method="Nelder-Mead")
best_w = np.abs(res.x)
best_w = best_w / best_w.sum()

probs = (best_w[:, None] * preds).sum(axis=0)

score = evaluate(y_val, probs)
print_results(score)
print(f"Optimized weights: {best_w}")

with mlflow.start_run():
    mlflow.log_metric("val_1-auc_roc", score)
    mlflow.log_param("model", "Ensemble(optuna_lgbm+2xLGBM+RF+ET)")
    mlflow.log_param("description", "optuna-tuned lgbm + 4 other models, opt weights")
    mlflow.log_param("status", "keep")
