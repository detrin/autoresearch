import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "florida_real_estate_sold_properties_ultimate.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "lastSoldPrice"
METRIC_NAME = "rmse"
MLFLOW_TRACKING_URI = "http://localhost:5050"
MLFLOW_EXPERIMENT = "florida-real-estate"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["sanitized_text"], errors="ignore")
    train, val = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return train, val


def evaluate(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def print_results(score):
    print(f"val_{METRIC_NAME}: {score:.6f}")
