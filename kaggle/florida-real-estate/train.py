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
smoothing = 10

def target_encode(train_df, val_df, col, target, smooth=10):
    stats = train_df.groupby(col)[target].agg(["mean", "count", "std", "median"]).reset_index()
    stats.columns = [col, f"{col}_te_mean", f"{col}_te_count", f"{col}_te_std", f"{col}_te_median"]
    stats[f"{col}_te_std"] = stats[f"{col}_te_std"].fillna(0)
    gm = train_df[target].mean()
    stats[f"{col}_te_smooth"] = (
        (stats[f"{col}_te_count"] * stats[f"{col}_te_mean"] + smooth * gm)
        / (stats[f"{col}_te_count"] + smooth)
    )
    out_cols = [col, f"{col}_te_smooth", f"{col}_te_count", f"{col}_te_std", f"{col}_te_median"]
    tr = train_df.merge(stats[out_cols], on=col, how="left")
    vl = val_df.merge(stats[out_cols], on=col, how="left")
    for c in out_cols[1:]:
        fill = gm if "smooth" in c or "mean" in c or "median" in c else 0
        vl[c] = vl[c].fillna(fill)
    return tr, vl, out_cols[1:]

te_feature_cols = []
for col in ["zip", "type", "sub_type"]:
    train, val, new_cols = target_encode(train, val, col, TARGET, smooth=smoothing)
    te_feature_cols.extend(new_cols)

extra_cols = ["property_age", "sqft_per_bed", "sqft_per_bath", "total_rooms", "bed_bath_ratio"] + te_feature_cols
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
    mlflow.log_param("description", "target enc zip+type+sub_type with median+std+smooth")
    mlflow.log_param("status", "keep")
