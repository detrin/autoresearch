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
    df["sqft_x_baths"] = df["sqft"] * df["baths"]
    df["age_x_sqft"] = df["property_age"] * df["sqft"]
    df["has_garage"] = (df["garage"] > 0).astype(int)
    df["is_new"] = (df["property_age"] <= 5).astype(int)

global_mean = train[TARGET].mean()
smoothing = 10
zip_stats = train.groupby("zip")[TARGET].agg(["mean", "count", "std", "median"]).reset_index()
zip_stats.columns = ["zip", "zip_te_mean", "zip_te_count", "zip_te_std", "zip_te_median"]
zip_stats["zip_te_std"] = zip_stats["zip_te_std"].fillna(0)
zip_stats["zip_te_smooth"] = (
    (zip_stats["zip_te_count"] * zip_stats["zip_te_mean"] + smoothing * global_mean)
    / (zip_stats["zip_te_count"] + smoothing)
)
train = train.merge(zip_stats[["zip", "zip_te_smooth", "zip_te_count", "zip_te_std", "zip_te_median"]], on="zip", how="left")
val = val.merge(zip_stats[["zip", "zip_te_smooth", "zip_te_count", "zip_te_std", "zip_te_median"]], on="zip", how="left")
for c in ["zip_te_smooth", "zip_te_median"]:
    val[c] = val[c].fillna(global_mean)
for c in ["zip_te_count", "zip_te_std"]:
    val[c] = val[c].fillna(0)

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio",
              "sqft_x_baths", "age_x_sqft", "has_garage", "is_new",
              "zip_te_smooth", "zip_te_count", "zip_te_std", "zip_te_median"]
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

params = {
    "n_estimators": 2000,
    "learning_rate": 0.021,
    "num_leaves": 107,
    "max_depth": 13,
    "min_child_samples": 22,
    "subsample": 0.626,
    "colsample_bytree": 0.503,
    "reg_alpha": 0.369,
    "reg_lambda": 9.581,
    "random_state": 42,
    "verbosity": -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
preds = model.predict(X_val)

score = evaluate(y_val, preds)
print_results(score)

with mlflow.start_run():
    mlflow.log_metric("val_rmse", score)
    mlflow.log_param("model", "LGBMRegressor-tuned")
    mlflow.log_param("description", "tuned params + interaction feats + zip median + has_garage + is_new")
    mlflow.log_param("status", "keep")
