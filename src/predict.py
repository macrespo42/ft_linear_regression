from linear_function import linear_function
from utils import get_thetas


def get_mileage() -> float:
    """get_mileage() -> float
    Get mileage to predict via user input
    """
    print("Please enter a mileage: ")
    mileage = input()
    return float(mileage)


def predict_price() -> float:
    """predict_price() -> float
    Predict price of a car with a given mileage
    work well only if the model was trained before
    """
    mileage = None
    theta0, theta1 = get_thetas()
    try:
        mileage = get_mileage()
    except Exception as e:
        print(f"Error: {e}")
    if not mileage:
        return 0.0
    estimated_price = linear_function(theta0, theta1, mileage)
    return estimated_price


def main():
    """main function"""
    estimated_price = predict_price()
    print(f"This car worth {estimated_price} euros")


if __name__ == "__main__":
    main()
