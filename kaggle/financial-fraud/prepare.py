import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "improved_fraud_dataset.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "is_fraud"
METRIC_NAME = "1-auc_roc"
MLFLOW_TRACKING_URI = "http://localhost:5050"
MLFLOW_EXPERIMENT = "financial-fraud"


def load_data():
    df = pd.read_csv(DATA_PATH)
    train, val = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET])
    return train, val


def evaluate(y_true, y_prob):
    return 1.0 - roc_auc_score(y_true, y_prob)


def print_results(score):
    print(f"val_{METRIC_NAME}: {score:.6f}")
