from linear_function import linear_function
from utils import get_thetas


def get_mileage() -> int:
    print("Please enter a mileage: ")
    mileage = input()
    return float(mileage)


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
