from linear_function import linear_function


def get_mileage() -> int:
    print("Please enter a mileage: ")
    mileage = input()
    return int(mileage)


def predict_price() -> float:
    mileage = None
    tetha0 = 0
    tetha1 = 0
    try:
        mileage = get_mileage()
    except Exception as e:
        print(f"Error: {e}")
    estimated_price = linear_function(tetha0, tetha1, mileage)
    return estimated_price


def main():
    estimated_price = predict_price()
    print(f"This car worth {estimated_price} euros")


if __name__ == "__main__":
    main()
