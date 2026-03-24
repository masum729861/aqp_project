from __future__ import annotations
import time
import pandas as pd


class StratifiedSampler:
    """
    Performs approximate query processing using stratified sampling.
    The dataset is divided by a group column, and samples are taken from each group.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        strata_column: str,
        sample_fraction: float = 0.1,
        random_state: int = 42
    ):
        if not (0 < sample_fraction <= 1):
            raise ValueError("sample_fraction must be between 0 and 1.")

        if strata_column not in dataframe.columns:
            raise ValueError(f"strata_column '{strata_column}' not found in dataframe.")

        self.df = dataframe.copy()
        self.strata_column = strata_column
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    def _get_sample(self) -> pd.DataFrame:
        sampled_parts = []

        for _, group in self.df.groupby(self.strata_column):
            n = max(1, int(len(group) * self.sample_fraction))
            n = min(n, len(group))
            sampled_group = group.sample(n=n, random_state=self.random_state)
            sampled_parts.append(sampled_group)

        return pd.concat(sampled_parts, ignore_index=True)

    def count(self, column: str) -> dict:
        start = time.perf_counter()

        sample_df = self._get_sample()
        estimated_count = 0.0

        for group_name, full_group in self.df.groupby(self.strata_column):
            sample_group = sample_df[sample_df[self.strata_column] == group_name]
            if len(sample_group) == 0:
                continue

            non_null_ratio = sample_group[column].count() / len(sample_group)
            estimated_count += non_null_ratio * len(full_group)

        elapsed = time.perf_counter() - start

        return {
            "method": "Stratified Sampling",
            "query": f"COUNT({column})",
            "strata_column": self.strata_column,
            "sample_fraction": self.sample_fraction,
            "sample_size": len(sample_df),
            "estimated_result": float(estimated_count),
            "time_seconds": elapsed
        }

    def sum(self, column: str) -> dict:
        start = time.perf_counter()

        sample_df = self._get_sample()
        estimated_sum = 0.0

        for group_name, full_group in self.df.groupby(self.strata_column):
            sample_group = sample_df[sample_df[self.strata_column] == group_name]
            if len(sample_group) == 0:
                continue

            group_weight = len(full_group) / len(sample_group)
            estimated_sum += sample_group[column].sum() * group_weight

        elapsed = time.perf_counter() - start

        return {
            "method": "Stratified Sampling",
            "query": f"SUM({column})",
            "strata_column": self.strata_column,
            "sample_fraction": self.sample_fraction,
            "sample_size": len(sample_df),
            "estimated_result": float(estimated_sum),
            "time_seconds": elapsed
        }

    def avg(self, column: str) -> dict:
        start = time.perf_counter()

        sample_df = self._get_sample()
        weighted_sum = 0.0
        total_size = 0

        for group_name, full_group in self.df.groupby(self.strata_column):
            sample_group = sample_df[sample_df[self.strata_column] == group_name]
            if len(sample_group) == 0:
                continue

            weighted_sum += sample_group[column].mean() * len(full_group)
            total_size += len(full_group)

        estimated_avg = weighted_sum / total_size if total_size > 0 else 0.0
        elapsed = time.perf_counter() - start

        return {
            "method": "Stratified Sampling",
            "query": f"AVG({column})",
            "strata_column": self.strata_column,
            "sample_fraction": self.sample_fraction,
            "sample_size": len(sample_df),
            "estimated_result": float(estimated_avg),
            "time_seconds": elapsed
        }


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    csv_file = "../dataset/sales.csv"
    numeric_column = "sales"
    strata_column = "region"

    df = load_dataset(csv_file)
    sampler = StratifiedSampler(df, strata_column=strata_column, sample_fraction=0.1, random_state=42)

    print(sampler.count(numeric_column))
    print(sampler.sum(numeric_column))
    print(sampler.avg(numeric_column))