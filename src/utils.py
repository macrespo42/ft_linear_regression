import pandas as pd
import numpy as np


def load(path: str) -> pd.DataFrame:
    """load(path: str) -> Dataset
    open a dataset and return it"""
    data_file = None
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("path isn't a csv file")
        data_file = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "File not found, try to run the script from the root of the project"
        )
    return data_file


def get_thetas() -> tuple[float, float]:
    theta0 = 0
    theta1 = 0
    try:
        df = load("dataset/thetas.csv")
        theta0, theta1 = (df["theta0"][0], df["theta1"][0])
    except Exception:
        print(
            """WARNING: the model is not trained, it must lead to incorrects results.
To train the model run train.py\n"""
        )
    return theta0, theta1


def rmse(actual: np.ndarray, predicted: np.ndarray):
    n = len(actual)
    total = 0

    for i in range(n):
        total += (predicted[i] - actual[i]) ** 2

    meanSquaredError = total / n
    return meanSquaredError**0.5
