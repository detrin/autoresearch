import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "student_mental_health_burnout_1M.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "risk_level"
METRIC_NAME = "1-f1"
MLFLOW_TRACKING_URI = "http://localhost:5050"
MLFLOW_EXPERIMENT = "student-mental-health"


def load_data():
    df = pd.read_csv(DATA_PATH)
    train, val = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET])
    return train, val


def evaluate(y_true, y_pred):
    return 1.0 - f1_score(y_true, y_pred, average="macro")


def print_results(score):
    print(f"val_{METRIC_NAME}: {score:.6f}")
