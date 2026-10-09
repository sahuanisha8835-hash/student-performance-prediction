import pandas as pd
from pathlib import Path


def load_data():
    project_folder = Path(__file__).resolve().parent.parent
    data_path = project_folder / "data" / "student_data.csv"

    data = pd.read_csv(data_path)
    return data


def preprocess_data(data):
    X = data.drop("result", axis=1)
    y = data["result"]

    X_encoded = pd.get_dummies(X, dtype=int)
    y_encoded = y.map({"Fail": 0, "Pass": 1})

    return X_encoded, y_encoded