import mlflow
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from datasci_toolkit import OptimalBinner
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
smoothing = 10
zip_stats = train.groupby("zip")[TARGET].agg(["mean", "count", "std"]).reset_index()
zip_stats.columns = ["zip", "zip_te_mean", "zip_te_count", "zip_te_std"]
zip_stats["zip_te_std"] = zip_stats["zip_te_std"].fillna(0)
zip_stats["zip_te_smooth"] = (
    (zip_stats["zip_te_count"] * zip_stats["zip_te_mean"] + smoothing * global_mean)
    / (zip_stats["zip_te_count"] + smoothing)
)
train = train.merge(zip_stats[["zip", "zip_te_smooth", "zip_te_count", "zip_te_std"]], on="zip", how="left")
val = val.merge(zip_stats[["zip", "zip_te_smooth", "zip_te_count", "zip_te_std"]], on="zip", how="left")
val["zip_te_smooth"] = val["zip_te_smooth"].fillna(global_mean)
val["zip_te_count"] = val["zip_te_count"].fillna(0)
val["zip_te_std"] = val["zip_te_std"].fillna(0)

bin_cols = ["sqft", "year_built", "beds", "baths", "property_age"]
y_train_series = train[TARGET]
train_pl = pl.from_pandas(train)
val_pl = pl.from_pandas(val)
y_pl = pl.from_pandas(y_train_series).to_series()

bin_feature_names = []
for col in bin_cols:
    try:
        x_series = train_pl[col].cast(pl.Float64)
        binner = OptimalBinner().fit(x_series, y_pl)
        train_binned = binner.transform(x_series)
        val_binned = binner.transform(val_pl[col].cast(pl.Float64))
        bin_name = f"{col}_bin"
        train[bin_name] = train_binned.to_pandas().values
        val[bin_name] = val_binned.to_pandas().values
        bin_feature_names.append(bin_name)
    except Exception:
        pass

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio",
              "zip_te_smooth", "zip_te_count", "zip_te_std"] + bin_feature_names
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
    mlflow.log_param("description", "baked optuna params + OptimalBinner 1D features")
    mlflow.log_param("status", "keep")
