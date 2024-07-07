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
