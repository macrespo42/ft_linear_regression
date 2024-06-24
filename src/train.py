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


def compute_cost(tetha0: int, tetha1: int, m: int, mileage: int):
    total_cost = 0
    return total_cost


def train(mileage: int) -> float:
    tetha0 = 0
    tetha1 = 0
    x_train, y_train = load_data()
    if x_train.shape != y_train.shape:
        raise AssertionError("Training data as incorrect shapes")
    m = x_train.shape[0]
    # cost = compute_cost(tetha0, tetha1, m, mileage)
    print(f"recap:\ntetha0: {tetha0}\ntetha1:{tetha1}\nm:{m}\nmileage:{mileage}")
    # print(f"Actual cost is: {cost}")


if __name__ == "__main__":
    train(4200)
