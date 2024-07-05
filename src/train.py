from linear_function import linear_function
from numpy import ndarray
from pandas import DataFrame
from parse_data import load_data


def f(x):
    return 60 * x


def mean_squared_error(data: DataFrame, function) -> float:
    sosr = 0
    for x, y in data:
        residual = y - f(x)
        sosr += residual**2
    return sosr / len(data)


def compute_cost(tetha0: int, tetha1: int, x_train: ndarray, y_train: ndarray):
    total_cost = 0
    m = len(x_train)
    for i in range(m):
        f = linear_function(tetha0, tetha1, x_train[i])
        total_cost += (f - y_train[i]) ** 2
    return total_cost / (2 * m)


def compute_gradient(
    tetha0: int, tetha1: int, x_train: ndarray, y_train: ndarray
) -> float:
    m = len(x_train)

    dj_tetha0 = 0
    dj_tetha1 = 0

    for i in range(m):
        f = linear_function(tetha0, tetha1, x_train[i])
        dj_tetha0 += (f - y_train[i]) * x_train[i]
        dj_tetha1 += f - y_train[i]

    dj_tetha0 /= m
    dj_tetha1 /= m
    return dj_tetha0, dj_tetha1


def train(mileage: int) -> float:
    tetha0 = 0  # b
    tetha1 = 0  # w
    # mileage = x
    x_train, y_train = load_data()
    if x_train.shape != y_train.shape:
        raise AssertionError("Training data as incorrect shapes")
    cost = compute_cost(tetha0, tetha1, x_train, y_train)
    tmp_dj_dw, tmp_dj_db = compute_gradient(tetha0, tetha1, x_train, y_train)
    print(f"Actual cost is: {cost}")
    print(f"Gradient is {tmp_dj_dw} - {tmp_dj_db}")


if __name__ == "__main__":
    train(4200)
