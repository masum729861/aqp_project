# exact_query.py

from __future__ import annotations
import time
import pandas as pd


class ExactQueryProcessor:
    """
    Executes aggregate queries on the full dataset.
    Supported aggregates: COUNT, SUM, AVG
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def count(self, column: str) -> dict:
        start = time.perf_counter()
        value = self.df[column].count()
        elapsed = time.perf_counter() - start

        return {
            "method": "Exact Query",
            "query": f"COUNT({column})",
            "result": int(value),
            "time_seconds": elapsed
        }

    def sum(self, column: str) -> dict:
        start = time.perf_counter()
        value = self.df[column].sum()
        elapsed = time.perf_counter() - start

        return {
            "method": "Exact Query",
            "query": f"SUM({column})",
            "result": float(value),
            "time_seconds": elapsed
        }

    def avg(self, column: str) -> dict:
        start = time.perf_counter()
        value = self.df[column].mean()
        elapsed = time.perf_counter() - start

        return {
            "method": "Exact Query",
            "query": f"AVG({column})",
            "result": float(value),
            "time_seconds": elapsed
        }


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    # Example usage
    csv_file = "dataset/sales.csv"
    numeric_column = "sales"

    df = load_dataset(csv_file)
    processor = ExactQueryProcessor(df)

    print(processor.count(numeric_column))
    print(processor.sum(numeric_column))
    print(processor.avg(numeric_column))