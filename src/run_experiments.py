from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from exact_query import ExactQueryProcessor
from uniform_sampling import UniformSampler
from stratified_sampling import StratifiedSampler
from adaptive_sampling import AdaptiveSampler


# =========================================================
# Project paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "sales.csv"

RESULT_TABLE_DIR = BASE_DIR / "results" / "tables"
RESULT_FIG_DIR = BASE_DIR / "results" / "figures"


# =========================================================
# Configuration
# =========================================================
NUMERIC_COLUMN = "sales"
STRATA_COLUMN = "region"

# Larger-scale experiment settings
DATASET_SIZE_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
NUM_TRIALS = 10
BASE_RANDOM_STATE = 42

# Fixed sampling settings
UNIFORM_SAMPLE_FRACTION = 0.10
STRATIFIED_SAMPLE_FRACTION = 0.10

# Improved adaptive settings for stronger stopping behavior
ADAPTIVE_INITIAL_FRACTION = 0.05
ADAPTIVE_MAX_FRACTION = 0.80
ADAPTIVE_STEP_FRACTION = 0.05
ADAPTIVE_ERROR_THRESHOLD = 0.10
ADAPTIVE_CONFIDENCE_Z = 1.96


# =========================================================
# Helper functions
# =========================================================
def ensure_directories() -> None:
    RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATASET_PATH}\n\n"
            "Please create sales.csv in the dataset folder."
        )

    if DATASET_PATH.stat().st_size == 0:
        raise ValueError(
            f"Dataset file is empty:\n{DATASET_PATH}"
        )

    try:
        df = pd.read_csv(DATASET_PATH)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Dataset file has no readable CSV columns:\n{DATASET_PATH}"
        )

    if df.empty:
        raise ValueError("Dataset loaded, but contains no rows.")

    return df


def validate_columns(df: pd.DataFrame) -> None:
    if NUMERIC_COLUMN not in df.columns:
        raise ValueError(
            f"Numeric column '{NUMERIC_COLUMN}' not found. Available columns: {list(df.columns)}"
        )

    if STRATA_COLUMN not in df.columns:
        raise ValueError(
            f"Strata column '{STRATA_COLUMN}' not found. Available columns: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df[NUMERIC_COLUMN]):
        raise ValueError(f"Column '{NUMERIC_COLUMN}' must be numeric.")


def relative_error(exact_value: float, estimated_value: float) -> float:
    if exact_value == 0:
        return 0.0 if estimated_value == 0 else float("inf")
    return abs(exact_value - estimated_value) / abs(exact_value)


def confidence_interval_from_trials(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))

    if len(values) == 1:
        return (values[0], values[0])

    series = pd.Series(values, dtype=float)
    mean_val = float(series.mean())
    std_val = float(series.std(ddof=1))
    margin = z * (std_val / math.sqrt(len(values)))
    return (mean_val - margin, mean_val + margin)


def get_subset(df: pd.DataFrame, fraction: float, random_state: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return df.copy().reset_index(drop=True)
    return df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)


def summarize_method_rows(df_rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = ["dataset_fraction", "dataset_rows", "aggregate", "method"]

    for keys, group in df_rows.groupby(group_cols):
        dataset_fraction, dataset_rows, aggregate, method = keys

        est_values = group["estimated_result"].tolist()
        ci_low, ci_high = confidence_interval_from_trials(est_values)

        summary_rows.append(
            {
                "dataset_fraction": dataset_fraction,
                "dataset_rows": dataset_rows,
                "aggregate": aggregate,
                "method": method,
                "mean_estimated_result": float(group["estimated_result"].mean()),
                "std_estimated_result": float(group["estimated_result"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_time_seconds": float(group["time_seconds"].mean()),
                "std_time_seconds": float(group["time_seconds"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_relative_error": float(group["relative_error"].mean()),
                "std_relative_error": float(group["relative_error"].std(ddof=1)) if len(group) > 1 else 0.0,
                "estimate_ci_lower": float(ci_low),
                "estimate_ci_upper": float(ci_high),
                "trials": int(len(group)),
            }
        )

    return (
        pd.DataFrame(summary_rows)
        .sort_values(by=["dataset_fraction", "aggregate", "method"])
        .reset_index(drop=True)
    )


# =========================================================
# Upgrade 2: clearer adaptive stopping trace
# =========================================================
def adaptive_trace_avg(
    df: pd.DataFrame,
    column: str,
    exact_avg: float,
    initial_fraction: float,
    max_fraction: float,
    step_fraction: float,
    z_value: float,
    threshold: float,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frac = initial_fraction
    stop_reached = False

    while frac <= max_fraction + 1e-12:
        sample_df = df.sample(frac=frac, random_state=random_state)
        series = sample_df[column].dropna()

        n = len(series)
        if n <= 1:
            estimated_avg = float(series.iloc[0]) if n == 1 else float("nan")
            margin = float("inf")
            estimated_rel_error = float("inf")
        else:
            estimated_avg = float(series.mean())
            std_val = float(series.std(ddof=1))
            margin = z_value * (std_val / math.sqrt(n))
            estimated_rel_error = abs(margin / estimated_avg) if estimated_avg != 0 else float("inf")

        actual_rel_error = relative_error(exact_avg, estimated_avg) if pd.notna(estimated_avg) else float("inf")
        stop_condition_met = bool(estimated_rel_error <= threshold)

        rows.append(
            {
                "sample_fraction": round(frac, 4),
                "sample_size": int(len(sample_df)),
                "estimated_avg": estimated_avg,
                "ci_lower": estimated_avg - margin if math.isfinite(margin) else float("nan"),
                "ci_upper": estimated_avg + margin if math.isfinite(margin) else float("nan"),
                "estimated_relative_error": estimated_rel_error,
                "actual_relative_error": actual_rel_error,
                "target_error_threshold": threshold,
                "stop_condition_met": stop_condition_met,
            }
        )

        if stop_condition_met:
            stop_reached = True
            break

        frac += step_fraction

    if not stop_reached and rows:
        rows[-1]["stop_condition_met"] = False

    return pd.DataFrame(rows)


# =========================================================
# Experiment execution
# =========================================================
def run_single_trial(
    df: pd.DataFrame,
    trial_id: int,
    dataset_fraction: float,
    trial_random_state: int,
) -> list[dict[str, Any]]:
    subset_df = get_subset(df, dataset_fraction, trial_random_state)
    dataset_rows = len(subset_df)

    exact = ExactQueryProcessor(subset_df)
    uniform = UniformSampler(
        subset_df,
        sample_fraction=UNIFORM_SAMPLE_FRACTION,
        random_state=trial_random_state,
    )
    stratified = StratifiedSampler(
        subset_df,
        strata_column=STRATA_COLUMN,
        sample_fraction=STRATIFIED_SAMPLE_FRACTION,
        random_state=trial_random_state,
    )
    adaptive = AdaptiveSampler(
        subset_df,
        initial_fraction=ADAPTIVE_INITIAL_FRACTION,
        max_fraction=ADAPTIVE_MAX_FRACTION,
        step_fraction=ADAPTIVE_STEP_FRACTION,
        error_threshold=ADAPTIVE_ERROR_THRESHOLD,
        confidence_z=ADAPTIVE_CONFIDENCE_Z,
        random_state=trial_random_state,
    )

    rows: list[dict[str, Any]] = []

    experiments = [
        ("COUNT", exact.count(NUMERIC_COLUMN), [uniform.count(NUMERIC_COLUMN), stratified.count(NUMERIC_COLUMN), adaptive.count(NUMERIC_COLUMN)]),
        ("SUM", exact.sum(NUMERIC_COLUMN), [uniform.sum(NUMERIC_COLUMN), stratified.sum(NUMERIC_COLUMN), adaptive.sum(NUMERIC_COLUMN)]),
        ("AVG", exact.avg(NUMERIC_COLUMN), [uniform.avg(NUMERIC_COLUMN), stratified.avg(NUMERIC_COLUMN), adaptive.avg(NUMERIC_COLUMN)]),
    ]

    for aggregate_name, exact_result, approx_results in experiments:
        exact_value = float(exact_result["result"])

        rows.append(
            {
                "trial_id": trial_id,
                "random_state": trial_random_state,
                "dataset_fraction": dataset_fraction,
                "dataset_rows": dataset_rows,
                "aggregate": aggregate_name,
                "method": "Exact Query",
                "exact_result": exact_value,
                "estimated_result": exact_value,
                "relative_error": 0.0,
                "time_seconds": float(exact_result["time_seconds"]),
                "sample_fraction": 1.0,
                "sample_size": dataset_rows,
                "confidence_interval_lower": exact_value,
                "confidence_interval_upper": exact_value,
                "relative_error_estimate": 0.0,
            }
        )

        for result in approx_results:
            estimated_value = float(result["estimated_result"])
            rows.append(
                {
                    "trial_id": trial_id,
                    "random_state": trial_random_state,
                    "dataset_fraction": dataset_fraction,
                    "dataset_rows": dataset_rows,
                    "aggregate": aggregate_name,
                    "method": result["method"],
                    "exact_result": exact_value,
                    "estimated_result": estimated_value,
                    "relative_error": float(relative_error(exact_value, estimated_value)),
                    "time_seconds": float(result["time_seconds"]),
                    "sample_fraction": result.get("sample_fraction", None),
                    "sample_size": result.get("sample_size", None),
                    "confidence_interval_lower": result.get("confidence_interval_lower", None),
                    "confidence_interval_upper": result.get("confidence_interval_upper", None),
                    "relative_error_estimate": result.get("relative_error_estimate", None),
                }
            )

    return rows


# =========================================================
# Plotting functions
# =========================================================
def plot_method_time_comparison(summary_df: pd.DataFrame) -> None:
    plot_df = (
        summary_df[summary_df["dataset_fraction"] == 1.0]
        .groupby("method", as_index=False)["mean_time_seconds"]
        .mean()
        .sort_values("mean_time_seconds")
    )

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["method"], plot_df["mean_time_seconds"])
    plt.title("Average Execution Time by Method")
    plt.xlabel("Method")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RESULT_FIG_DIR / "method_execution_time_comparison.png", dpi=300)
    plt.close()


def plot_method_error_comparison(summary_df: pd.DataFrame) -> None:
    plot_df = (
        summary_df[summary_df["dataset_fraction"] == 1.0]
        .groupby("method", as_index=False)["mean_relative_error"]
        .mean()
        .sort_values("mean_relative_error")
    )

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["method"], plot_df["mean_relative_error"])
    plt.title("Average Relative Error by Method")
    plt.xlabel("Method")
    plt.ylabel("Relative Error")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RESULT_FIG_DIR / "method_relative_error_comparison.png", dpi=300)
    plt.close()


def plot_scalability_time(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["aggregate"] == "AVG"].copy()

    plt.figure(figsize=(9, 5))
    for method in plot_df["method"].unique():
        method_df = plot_df[plot_df["method"] == method].sort_values("dataset_rows")
        plt.plot(
            method_df["dataset_rows"],
            method_df["mean_time_seconds"],
            marker="o",
            label=method,
        )

    plt.title("Dataset Size vs Execution Time (AVG Query)")
    plt.xlabel("Dataset Rows")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_FIG_DIR / "dataset_size_vs_execution_time.png", dpi=300)
    plt.close()


def plot_scalability_error(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["aggregate"] == "AVG"].copy()

    plt.figure(figsize=(9, 5))
    for method in plot_df["method"].unique():
        method_df = plot_df[plot_df["method"] == method].sort_values("dataset_rows")
        plt.plot(
            method_df["dataset_rows"],
            method_df["mean_relative_error"],
            marker="o",
            label=method,
        )

    plt.title("Dataset Size vs Relative Error (AVG Query)")
    plt.xlabel("Dataset Rows")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_FIG_DIR / "dataset_size_vs_relative_error.png", dpi=300)
    plt.close()


def plot_adaptive_trace(trace_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(
        trace_df["sample_size"],
        trace_df["actual_relative_error"],
        marker="o",
        label="Actual Relative Error",
    )
    plt.plot(
        trace_df["sample_size"],
        trace_df["estimated_relative_error"],
        marker="o",
        label="Estimated Relative Error",
    )
    plt.axhline(
        trace_df["target_error_threshold"].iloc[0],
        linestyle="--",
        label="Target Error Threshold",
    )
    plt.title("Adaptive Sampling: Sample Size vs Error")
    plt.xlabel("Sample Size")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_FIG_DIR / "adaptive_sample_size_vs_error.png", dpi=300)
    plt.close()


# =========================================================
# Main
# =========================================================
def main() -> None:
    print("Loading dataset...")
    df = load_dataset()
    validate_columns(df)
    ensure_directories()

    print(f"Dataset loaded: {df.shape}")
    print(f"Running advanced experiments with {NUM_TRIALS} trials...")

    all_rows: list[dict[str, Any]] = []

    for dataset_fraction in DATASET_SIZE_FRACTIONS:
        print(f"Processing dataset fraction: {dataset_fraction:.2f}")

        for trial_id in range(1, NUM_TRIALS + 1):
            trial_random_state = BASE_RANDOM_STATE + trial_id + int(dataset_fraction * 1000)

            trial_rows = run_single_trial(
                df=df,
                trial_id=trial_id,
                dataset_fraction=dataset_fraction,
                trial_random_state=trial_random_state,
            )
            all_rows.extend(trial_rows)

    detailed_df = pd.DataFrame(all_rows)
    summary_df = summarize_method_rows(detailed_df)

    # Upgrade 2 trace on full dataset
    exact_full = ExactQueryProcessor(df)
    exact_avg_full = float(exact_full.avg(NUMERIC_COLUMN)["result"])

    trace_df = adaptive_trace_avg(
        df=df,
        column=NUMERIC_COLUMN,
        exact_avg=exact_avg_full,
        initial_fraction=ADAPTIVE_INITIAL_FRACTION,
        max_fraction=ADAPTIVE_MAX_FRACTION,
        step_fraction=ADAPTIVE_STEP_FRACTION,
        z_value=ADAPTIVE_CONFIDENCE_Z,
        threshold=ADAPTIVE_ERROR_THRESHOLD,
        random_state=BASE_RANDOM_STATE,
    )

    # Save tables
    detailed_path = RESULT_TABLE_DIR / "advanced_experiment_results_detailed.csv"
    summary_path = RESULT_TABLE_DIR / "advanced_experiment_results_summary.csv"
    trace_path = RESULT_TABLE_DIR / "adaptive_sampling_trace.csv"

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)

    # Save plots
    plot_method_time_comparison(summary_df)
    plot_method_error_comparison(summary_df)
    plot_scalability_time(summary_df)
    plot_scalability_error(summary_df)
    plot_adaptive_trace(trace_df)

    print("\nSaved tables:")
    print(detailed_path)
    print(summary_path)
    print(trace_path)

    print("\nSaved figures:")
    print(RESULT_FIG_DIR / "method_execution_time_comparison.png")
    print(RESULT_FIG_DIR / "method_relative_error_comparison.png")
    print(RESULT_FIG_DIR / "dataset_size_vs_execution_time.png")
    print(RESULT_FIG_DIR / "dataset_size_vs_relative_error.png")
    print(RESULT_FIG_DIR / "adaptive_sample_size_vs_error.png")

    print("\nTop-level summary (full dataset only):")
    full_summary = summary_df[summary_df["dataset_fraction"] == 1.0].copy()
    print(
        full_summary[
            [
                "aggregate",
                "method",
                "mean_estimated_result",
                "mean_time_seconds",
                "mean_relative_error",
                "estimate_ci_lower",
                "estimate_ci_upper",
                "trials",
            ]
        ].to_string(index=False)
    )

    print("\nAdaptive trace preview:")
    print(trace_df.to_string(index=False))

    print("\nAdvanced experiment completed successfully.")


if __name__ == "__main__":
    main()