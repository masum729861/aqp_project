# adaptive_sampling.py

from __future__ import annotations
import time
import math
import pandas as pd


class AdaptiveSampler:
    """
    Variance-aware adaptive sampling for approximate AVG / SUM / COUNT queries.

    Idea:
    - Start with a small sample
    - Estimate result and confidence interval
    - Increase sample size until relative error is below threshold
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        initial_fraction: float = 0.05,
        max_fraction: float = 1.0,
        step_fraction: float = 0.05,
        error_threshold: float = 0.05,
        confidence_z: float = 1.96,
        random_state: int = 42
    ):
        if not (0 < initial_fraction <= 1):
            raise ValueError("initial_fraction must be between 0 and 1.")
        if not (0 < max_fraction <= 1):
            raise ValueError("max_fraction must be between 0 and 1.")
        if initial_fraction > max_fraction:
            raise ValueError("initial_fraction cannot be greater than max_fraction.")
        if step_fraction <= 0:
            raise ValueError("step_fraction must be positive.")
        if error_threshold <= 0:
            raise ValueError("error_threshold must be positive.")

        self.df = dataframe.copy()
        self.initial_fraction = initial_fraction
        self.max_fraction = max_fraction
        self.step_fraction = step_fraction
        self.error_threshold = error_threshold
        self.confidence_z = confidence_z
        self.random_state = random_state

    def _sample_df(self, frac: float) -> pd.DataFrame:
        return self.df.sample(frac=frac, random_state=self.random_state)

    def _mean_confidence_interval(self, sample_series: pd.Series) -> tuple[float, float]:
        """
        Returns:
            (sample_mean, margin_of_error)
        """
        n = len(sample_series)
        if n == 0:
            return 0.0, float("inf")
        if n == 1:
            return float(sample_series.iloc[0]), float("inf")

        mean_val = sample_series.mean()
        std_val = sample_series.std(ddof=1)
        margin = self.confidence_z * (std_val / math.sqrt(n))

        return float(mean_val), float(margin)

    def avg(self, column: str) -> dict:
        start = time.perf_counter()

        frac = self.initial_fraction
        best_result = None

        while frac <= self.max_fraction + 1e-9:
            sample_df = self._sample_df(frac)
            sample_series = sample_df[column].dropna()

            mean_estimate, margin = self._mean_confidence_interval(sample_series)

            relative_error = abs(margin / mean_estimate) if mean_estimate != 0 else float("inf")

            best_result = {
                "method": "Adaptive Sampling",
                "query": f"AVG({column})",
                "sample_fraction": frac,
                "sample_size": len(sample_df),
                "estimated_result": mean_estimate,
                "confidence_interval_lower": mean_estimate - margin,
                "confidence_interval_upper": mean_estimate + margin,
                "relative_error_estimate": relative_error
            }

            if relative_error <= self.error_threshold:
                break

            frac += self.step_fraction

        elapsed = time.perf_counter() - start
        best_result["time_seconds"] = elapsed
        return best_result

    def sum(self, column: str) -> dict:
        start = time.perf_counter()

        total_rows = len(self.df)
        frac = self.initial_fraction
        best_result = None

        while frac <= self.max_fraction + 1e-9:
            sample_df = self._sample_df(frac)
            sample_series = sample_df[column].dropna()

            mean_estimate, margin = self._mean_confidence_interval(sample_series)

            estimated_sum = mean_estimate * total_rows
            sum_margin = margin * total_rows

            relative_error = abs(sum_margin / estimated_sum) if estimated_sum != 0 else float("inf")

            best_result = {
                "method": "Adaptive Sampling",
                "query": f"SUM({column})",
                "sample_fraction": frac,
                "sample_size": len(sample_df),
                "estimated_result": estimated_sum,
                "confidence_interval_lower": estimated_sum - sum_margin,
                "confidence_interval_upper": estimated_sum + sum_margin,
                "relative_error_estimate": relative_error
            }

            if relative_error <= self.error_threshold:
                break

            frac += self.step_fraction

        elapsed = time.perf_counter() - start
        best_result["time_seconds"] = elapsed
        return best_result

    def count(self, column: str) -> dict:
        start = time.perf_counter()

        total_rows = len(self.df)
        frac = self.initial_fraction
        best_result = None

        while frac <= self.max_fraction + 1e-9:
            sample_df = self._sample_df(frac)
            non_null_count = sample_df[column].count()

            estimated_count = non_null_count / frac

            # For COUNT, simple approximate margin
            # Here we estimate using binomial proportion p = non_null/sample_size
            sample_size = len(sample_df)
            if sample_size <= 1:
                margin = float("inf")
                relative_error = float("inf")
            else:
                p_hat = non_null_count / sample_size
                std_error = math.sqrt((p_hat * (1 - p_hat)) / sample_size)
                margin = self.confidence_z * std_error * total_rows
                relative_error = abs(margin / estimated_count) if estimated_count != 0 else float("inf")

            best_result = {
                "method": "Adaptive Sampling",
                "query": f"COUNT({column})",
                "sample_fraction": frac,
                "sample_size": sample_size,
                "estimated_result": float(estimated_count),
                "confidence_interval_lower": float(estimated_count - margin),
                "confidence_interval_upper": float(estimated_count + margin),
                "relative_error_estimate": relative_error
            }

            if relative_error <= self.error_threshold:
                break

            frac += self.step_fraction

        elapsed = time.perf_counter() - start
        best_result["time_seconds"] = elapsed
        return best_result


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    csv_file = "dataset/sales.csv"
    numeric_column = "sales"

    df = load_dataset(csv_file)

    sampler = AdaptiveSampler(
        df,
        initial_fraction=0.05,
        max_fraction=0.50,
        step_fraction=0.05,
        error_threshold=0.05,
        confidence_z=1.96,
        random_state=42
    )

    print(sampler.count(numeric_column))
    print(sampler.sum(numeric_column))
    print(sampler.avg(numeric_column))