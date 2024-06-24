import matplotlib.pyplot as plt
from parse_data import load


def show_dataset() -> None:
    dataset = load("dataset/data.csv")

    plt.title("Price of cars per km")
    plt.scatter(dataset["km"], dataset["price"])
    plt.ylabel("price (in eur)")
    plt.xlabel("mileage (in km)")
    plt.show()


def main():
    show_dataset()


if __name__ == "__main__":
    main()
