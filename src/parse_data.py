import pandas as pd


def load(path: str) -> pd.DataFrame:
    """load(path: str) -> Dataset
    open a dataset and return it"""
    data_file = None
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("path isn't a csv file")
        data_file = pd.read_csv(path)
    except FileNotFoundError:
        print("File not found, try to run the script from the root of the project")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    return data_file


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_min_max_scaled = df.copy()

    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (
            df_min_max_scaled[column] - df_min_max_scaled[column].min()
        ) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())

    return df_min_max_scaled


def load_data(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    """load_data() -> tuple[pd.DataFrame]
    return a tuple of Dataframe, the first one
    is the km and the second one the price
    """
    return (df["km"], df["price"])


if __name__ == "__main__":
    print(load_data())
