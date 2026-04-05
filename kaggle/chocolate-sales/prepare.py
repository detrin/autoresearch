import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

DATA_DIR = os.path.dirname(__file__)
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "profit"
METRIC_NAME = "mape"
MLFLOW_TRACKING_URI = "http://localhost:5050"
MLFLOW_EXPERIMENT = "chocolate-sales"


def load_data():
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))

    df = sales.merge(products, on="product_id", how="left")
    df = df.merge(stores, on="store_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")

    train, val = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return train, val


def evaluate(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def print_results(score):
    print(f"val_{METRIC_NAME}: {score:.6f}")
