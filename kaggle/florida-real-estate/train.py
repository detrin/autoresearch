import mlflow
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from prepare import load_data, evaluate, print_results, TARGET, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

train, val = load_data()

drop_cols = [TARGET, "listPrice"]
feature_cols = [c for c in train.columns if c not in drop_cols]

for df in [train, val]:
    df["property_age"] = 2026 - df["year_built"]
    df["sqft_per_bed"] = df["sqft"] / df["beds"].replace(0, np.nan)
    df["sqft_per_bath"] = df["sqft"] / df["baths"].replace(0, np.nan)
    df["total_rooms"] = df["beds"] + df["baths"]
    df["bed_bath_ratio"] = df["beds"] / df["baths"].replace(0, np.nan)

global_mean = train[TARGET].mean()
zip_stats = train.groupby("zip")[TARGET].agg(["mean", "count", "std"]).reset_index()
zip_stats.columns = ["zip", "zip_target_mean", "zip_count", "zip_target_std"]
zip_stats["zip_target_std"] = zip_stats["zip_target_std"].fillna(0)

smoothing = 10
zip_stats["zip_target_smooth"] = (
    (zip_stats["zip_count"] * zip_stats["zip_target_mean"] + smoothing * global_mean)
    / (zip_stats["zip_count"] + smoothing)
)

train = train.merge(zip_stats[["zip", "zip_target_smooth", "zip_count", "zip_target_std"]], on="zip", how="left")
val = val.merge(zip_stats[["zip", "zip_target_smooth", "zip_count", "zip_target_std"]], on="zip", how="left")
val["zip_target_smooth"] = val["zip_target_smooth"].fillna(global_mean)
val["zip_count"] = val["zip_count"].fillna(0)
val["zip_target_std"] = val["zip_target_std"].fillna(0)

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio",
              "zip_target_smooth", "zip_count", "zip_target_std"]
feature_cols = feature_cols + extra_cols

cat_cols = train[feature_cols].select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    val[col] = le.transform(val[col].astype(str))
    encoders[col] = le

X_train, y_train = train[feature_cols].fillna(-1), train[TARGET]
X_val, y_val = val[feature_cols].fillna(-1), val[TARGET]

model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=63, random_state=42, verbosity=-1)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGBMRegressor")
    mlflow.log_param("description", "target encoding for zip + basic feat eng")
    mlflow.log_param("status", "keep")
