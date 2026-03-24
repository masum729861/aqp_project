# uniform_sampling.py

from __future__ import annotations
import time
import pandas as pd


class UniformSampler:
    """
    Performs approximate query processing using uniform random sampling.
    """

    def __init__(self, dataframe: pd.DataFrame, sample_fraction: float = 0.1, random_state: int = 42):
        if not (0 < sample_fraction <= 1):
            raise ValueError("sample_fraction must be between 0 and 1.")

        self.df = dataframe.copy()
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    def _get_sample(self) -> pd.DataFrame:
        return self.df.sample(frac=self.sample_fraction, random_state=self.random_state)

    def count(self, column: str) -> dict:
        start = time.perf_counter()

        sample_df = self._get_sample()
        sample_count = sample_df[column].count()

        # Scale up the count estimate
        estimated_count = sample_count / self.sample_fraction

        elapsed = time.perf_counter() - start

        return {
            "method": "Uniform Sampling",
            "query": f"COUNT({column})",
            "sample_fraction": self.sample_fraction,
            "sample_size": len(sample_df),
            "estimated_result": float(estimated_count),
            "time_seconds": elapsed
        }

    def sum(self, column: str) -> dict:
        start = time.perf_counter()

        sample_df = self._get_sample()
        sample_sum = sample_df[column].sum()

        # Scale up the sum estimate
        estimated_sum = sample_sum / self.sample_fraction

        elapsed = time.perf_counter() - start

        return {
            "method": "Uniform Sampling",
            "query": f"SUM({column})",
            "sample_fraction": self.sample_fraction,
            "sample_size": len(sample_df),
            "estimated_result": float(estimated_sum),
            "time_seconds": elapsed
        }

    def avg(self, column: str) -> dict:
        start = time.perf_counter()

        sample_df = self._get_sample()
        estimated_avg = sample_df[column].mean()

        elapsed = time.perf_counter() - start

        return {
            "method": "Uniform Sampling",
            "query": f"AVG({column})",
            "sample_fraction": self.sample_fraction,
            "sample_size": len(sample_df),
            "estimated_result": float(estimated_avg),
            "time_seconds": elapsed
        }


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    csv_file = "dataset/sales.csv"
    numeric_column = "sales"

    df = load_dataset(csv_file)
    sampler = UniformSampler(df, sample_fraction=0.1, random_state=42)

    print(sampler.count(numeric_column))
    print(sampler.sum(numeric_column))
    print(sampler.avg(numeric_column))