import matplotlib.pyplot as plt
from utils import load, get_thetas
from linear_function import linear_function
import pandas as pd
import numpy as np


def show_dataset(df: pd.DataFrame) -> None:
    """def show_dataset(df: pd.dataset) -> None
    plot the training datas before any training
    """

    plt.title("Price of cars per km")
    plt.scatter(df["km"], df["price"])
    plt.ylabel("price (in eur)")
    plt.xlabel("mileage (in km)")
    plt.show()


def show_linear_regression(df: pd.DataFrame) -> None:
    """show_linear_regression(x, y, theta0, theta1) -> None
    Plot the training data and the line of the linear regression
    """

    x, y = (df["km"].values, df["price"].values)

    x = np.array(x)
    y = np.array(y)

    theta0, theta1 = get_thetas()

    y_predict = []
    for i in range(len(x)):
        y_predict.append(linear_function(theta0, theta1, x[i]))

    plt.title("Linear regression line of predictions")
    plt.scatter(x, y)
    plt.plot(x, y_predict, color="r")
    plt.ylabel("price (in eur)")
    plt.xlabel("mileage (in km)")
    plt.show()


def main():
    """main function"""
    df = None
    try:
        df = load("dataset/data.csv")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    loop = True
    while loop:
        print("Which graph do you want to show ?")
        print("1: A plot of the training dataset")
        print("2: Same as 1 but also show the line from the linear regression")
        print("3: Quit the program")

        choice = input()
        if choice == "1":
            show_dataset(df)
        elif choice == "2":
            show_linear_regression(df)
        elif choice == "3":
            print("Bye! :)")
            exit(0)
        else:
            continue


if __name__ == "__main__":
    main()
