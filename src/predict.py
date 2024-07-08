from linear_function import linear_function
from parse_data import load


def get_mileage() -> int:
    print("Please enter a mileage: ")
    mileage = input()
    return float(mileage)


def get_thetas() -> tuple[float]:
    theta0 = 0
    theta1 = 0
    try:
        df = load("dataset/thetas.csv")
        theta0, theta1 = (df["theta0"][0], df["theta1"][0])
    except Exception:
        print(
            """WARNING: the model is not trained, it must lead to incorrects predictions.
To train the model run train.py\n"""
        )
    return theta0, theta1


def predict_price() -> float:
    mileage = None
    theta0, theta1 = get_thetas()
    try:
        mileage = get_mileage()
    except Exception as e:
        print(f"Error: {e}")
    estimated_price = linear_function(theta0, theta1, mileage)
    return estimated_price


def main():
    estimated_price = predict_price()
    print(f"This car worth {estimated_price} euros")


if __name__ == "__main__":
    main()
