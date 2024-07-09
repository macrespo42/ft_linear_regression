import numpy as np
from pandas import DataFrame
from linear_function import linear_function
from utils import load, rmse

from describe import TrainingData


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.theta0 = 0.0  # b
        self.theta1 = 0.0  # w

    def _compute_cost(self, x: np.ndarray, y: np.ndarray) -> float:
        """_compute_cost(self, x: np.ndarray, y: np.ndarray) -> float
        compute the cost J of the actual thetas
        """
        total_cost = 0
        m = len(x)
        for i in range(m):
            f = linear_function(self.theta0, self.theta1, x[i])
            total_cost += (f - y[i]) ** 2
        return total_cost / (2 * m)

    def _compute_gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """_compute_gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]
        compute the dj_theta0 and dj_theta1 for the gradient descent algorithm
        """
        m = len(x)

        dj_theta0: float = 0.0
        dj_theta1: float = 0.0

        for i in range(m):
            prediction = linear_function(self.theta0, self.theta1, x[i])

            theta0_i = (prediction - y[i]) * x[i]
            theta1_i = prediction - y[i]

            dj_theta0 += theta0_i
            dj_theta1 += theta1_i

        dj_theta0 /= m
        dj_theta1 /= m

        return (dj_theta0, dj_theta1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """fit(self, x: np.ndarray, y: np.ndarray) -> None
        compute the best theta0 and theta1 using gradient descent algorithm
        """
        for _ in range(self.n_iters):
            dj_theta1, dj_theta0 = self._compute_gradient(x, y)

            self.theta1 -= self.learning_rate * dj_theta1
            self.theta0 -= self.learning_rate * dj_theta0

    def save(self) -> None:
        """save(self) -> None
        save the thetas to a csv file
        """
        thetas = DataFrame(data={"theta0": [self.theta0], "theta1": [self.theta1]})
        thetas.to_csv("dataset/thetas.csv", index=False)


def train() -> None:
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
    model.fit(x_norm, y_norm)

    model.theta0, model.theta1 = data.denormalize(model.theta0, model.theta1)

    model.save()

    y_pred = []
    for i in range(len(x_train)):
        y_pred.append(linear_function(model.theta0, model.theta1, x_train[i]))

    print(f"Root Mean Squared Error (RMSE): {rmse(y_train, y_pred)}")


if __name__ == "__main__":
    train()
