import copy

import numpy as np
from linear_function import linear_function
from parse_data import load, normalize_dataset


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1500) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def _compute_cost(
        theta0: int, theta1: int, x_train: np.ndarray, y_train: np.ndarray
    ):
        total_cost = 0
        m = len(x_train)
        for i in range(m):
            f = linear_function(theta0, theta1, x_train[i])
            total_cost += (f - y_train[i]) ** 2
        return total_cost / (2 * m)

    def compute_gradient(
        theta0: int, theta1: int, x_train: np.ndarray, y_train: np.ndarray
    ) -> float:
        m = len(x_train)

        dj_theta0 = 0
        dj_theta1 = 0

        for i in range(m):
            prediction = linear_function(theta0, theta1, x_train[i])

            theta0_i = (prediction - y_train[i]) * x_train[i]
            theta1_i = prediction - y_train[i]

            dj_theta0 += theta0_i
            dj_theta1 += theta1_i
            # print(f"cost: {compute_cost(theta0_i, theta1_i, x_train, y_train)}")

        dj_theta0 /= m
        dj_theta1 /= m

        return dj_theta0, dj_theta1

    def fit(self, x: np.ndarray, y: np.ndarray, theta1: float, theta0: float):
        w = copy.deepcopy(theta1)
        b = theta0

        for i in range(self.n_iters):
            dj_dw, dj_db = self._compute_gradient(b, w, x, y)

            w = w - self.learning_rate * dj_dw
            b = b - self.learning_rate * dj_db
        return w, b


def train(mileage: int) -> float:
    theta0 = 0  # b
    theta1 = 0  # w

    df = load("dataset/data.csv")
    normalized_df = df

    x_train, y_train = (normalized_df["km"], normalized_df["price"])
    if x_train.shape != y_train.shape:
        raise AssertionError("Training data as incorrect shapes")

    model = LinearRegression()
    w, b = model.fit(x_train, y_train, theta0, theta1)

    # test train
    m = x_train.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w * x_train[i] + b

    predict1 = 3.5 * w + b
    print("For population = 35,000, we predict a profit of $%.2f" % (predict1 * 10000))

    predict2 = 7.0 * w + b
    print("For population = 70,000, we predict a profit of $%.2f" % (predict2 * 10000))


if __name__ == "__main__":
    train(13000)
