import numpy as np
from pandas import DataFrame
from linear_function import linear_function
from parse_data import load

import matplotlib.pyplot as plt
from describe import TrainingData


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1500) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.theta0 = 0.0  # b
        self.theta1 = 0.0  # w

    def _compute_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        total_cost = 0
        m = len(x)
        for i in range(m):
            f = linear_function(self.theta0, self.theta1, x[i])
            total_cost += (f - y[i]) ** 2
        return total_cost / (2 * m)

    def _compute_gradient(self, x: np.ndarray, y: np.ndarray) -> float:
        m = len(x)

        dj_theta0 = 0
        dj_theta1 = 0

        for i in range(m):
            prediction = linear_function(self.theta0, self.theta1, x[i])

            theta0_i = (prediction - y[i]) * x[i]
            theta1_i = prediction - y[i]

            dj_theta0 += theta0_i
            dj_theta1 += theta1_i

        dj_theta0 /= m
        dj_theta1 /= m

        return dj_theta0, dj_theta1

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float]:
        for i in range(self.n_iters):
            dj_dw, dj_db = self._compute_gradient(x, y)

            self.theta1 -= self.learning_rate * dj_dw
            self.theta0 -= self.learning_rate * dj_db
        return self.theta1, self.theta0

    def save(self) -> None:
        thetas = DataFrame(data={"theta0": [self.theta0], "theta1": [self.theta1]})
        thetas.to_csv("dataset/thetas.csv")
        print(thetas)


def train() -> float:
    df = None
    try:
        df = load("dataset/data.csv")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    x_train, y_train = (df["km"].values, df["price"].values)

    data = TrainingData(x_train, y_train)
    x_norm, y_norm = data.normalize()

    if x_train.shape != y_train.shape:
        raise AssertionError("Training data as incorrect shapes")

    model = LinearRegression()
    # without normalize
    theta1, theta0 = model.fit(x_train, y_train)

    # wit normalize
    # theta1, theta0 = model.fit(x_norm, y_norm)
    # theta0, theta1 = data.denormalize(theta0, theta1)
    model.save()


if __name__ == "__main__":
    train()
