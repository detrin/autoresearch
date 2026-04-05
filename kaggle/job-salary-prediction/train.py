import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

cat_cols = train.select_dtypes(include="object").columns.tolist()
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols])
val[cat_cols] = oe.transform(val[cat_cols])

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

categorical_mask = [col in cat_cols for col in X_train.columns]

model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=8,
    categorical_features=categorical_mask,
    random_state=42,
)
model.fit(X_train, y_train)
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "HistGradientBoostingRegressor")
    mlflow.log_param("description", "HGBR 500 iter lr=0.05 depth=8 ordinal cats")
    mlflow.log_param("status", "keep")
