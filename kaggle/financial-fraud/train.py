import mlflow
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
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

model = lgb.LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
probs = model.predict_proba(X_val)[:, 1]

score = evaluate(y_val, probs)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_1-auc_roc", score)
    mlflow.log_param("model", "LGBMClassifier")
    mlflow.log_param("description", "lgbm 3000 trees lr=0.03 auc eval early_stop=100")
    mlflow.log_param("status", "keep")
