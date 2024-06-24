import pandas as pd
import matplotlib.pyplot as plt


def load(path: str):
    """load(path: str) -> Dataset
    open a dataset and return it dimensions"""
    data_file = None
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("path isn't a csv file")
        data_file = pd.read_csv(path)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    print(f"Loadind dataset of dimensions {data_file.shape}")
    return data_file


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
