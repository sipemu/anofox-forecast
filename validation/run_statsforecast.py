#!/usr/bin/env python3
"""Run statsforecast models on synthetic time series data.

This script loads the synthetic time series from CSV files and runs
equivalent statsforecast models, saving the results for comparison
with the Rust implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    RandomWalkWithDrift,
    SimpleExponentialSmoothing,
    Holt,
    HoltWinters,
    ARIMA,
    AutoARIMA,
    AutoETS,
    Theta,
    CrostonClassic,
    CrostonSBA,
    TSB,
)

# Configuration
HORIZON = 12
SEASONAL_PERIOD = 12
CONFIDENCE_LEVELS = [80, 90, 95]

# Directories
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "statsforecast"

# Series types
SERIES_TYPES = [
    "stationary",
    "trend",
    "seasonal",
    "trend_seasonal",
    "seasonal_negative",       # Has negative values - tests fallback to additive
    "multiplicative_seasonal", # True multiplicative seasonality
]


def load_series(series_type: str) -> pd.DataFrame:
    """Load a time series from CSV and format for statsforecast."""
    path = DATA_DIR / f"{series_type}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # statsforecast expects columns: unique_id, ds, y
    df = df.rename(columns={"timestamp": "ds", "value": "y"})
    df["unique_id"] = series_type

    return df[["unique_id", "ds", "y"]]


def get_models():
    """Create list of statsforecast models to run.

    Returns list of (name, model, has_native_intervals) tuples.
    Models without native interval support will only produce point forecasts.
    """
    return [
        # Baseline models - have native intervals
        ("Naive", Naive(), True),
        ("SeasonalNaive", SeasonalNaive(season_length=SEASONAL_PERIOD), True),
        ("RandomWalkWithDrift", RandomWalkWithDrift(), True),

        # Exponential smoothing
        ("SES", SimpleExponentialSmoothing(alpha=0.1), False),  # No native intervals
        ("Holt", Holt(season_length=SEASONAL_PERIOD), True),
        ("HoltWinters", HoltWinters(season_length=SEASONAL_PERIOD, error_type="A"), True),

        # ARIMA models - have native intervals
        ("ARIMA_1_1_1", ARIMA(order=(1, 1, 1)), True),
        ("AutoARIMA", AutoARIMA(season_length=SEASONAL_PERIOD), True),

        # ETS - has native intervals
        ("AutoETS", AutoETS(season_length=SEASONAL_PERIOD), True),

        # Theta - has native intervals
        ("Theta", Theta(season_length=SEASONAL_PERIOD), True),

        # Intermittent demand models - no native intervals
        ("Croston", CrostonClassic(), False),
        ("CrostonSBA", CrostonSBA(), False),
        ("TSB", TSB(alpha_d=0.1, alpha_p=0.1), False),
    ]


def run_forecasts(df: pd.DataFrame, series_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all models on a single series and return forecasts.

    Returns:
        Tuple of (point_forecasts_df, confidence_intervals_df)
    """
    models = get_models()

    point_results = []
    ci_results = []

    for model_name, model, has_native_intervals in models:
        try:
            print(f"  Running {model_name}...")

            # Create StatsForecast object with single model
            sf = StatsForecast(
                models=[model],
                freq="MS",  # Month start
                n_jobs=1,
            )

            # Fit and predict
            sf.fit(df)

            # Get point forecasts
            point_forecast = sf.predict(h=HORIZON)

            # Extract model column name
            cols = point_forecast.columns.tolist()
            model_col = [c for c in cols if c not in ["unique_id", "ds"]][0]

            # Store point forecasts
            for i, row in point_forecast.reset_index().iterrows():
                step = i + 1
                point_results.append({
                    "series_type": series_type,
                    "model": model_name,
                    "step": step,
                    "forecast": row[model_col],
                })

            # Get confidence intervals only for models with native support
            if has_native_intervals:
                for level in CONFIDENCE_LEVELS:
                    forecast_ci = sf.predict(h=HORIZON, level=[level])

                    lo_col = f"{model_col}-lo-{level}"
                    hi_col = f"{model_col}-hi-{level}"

                    for i, row in forecast_ci.reset_index().iterrows():
                        step = i + 1
                        ci_results.append({
                            "series_type": series_type,
                            "model": model_name,
                            "step": step,
                            "level": level,
                            "lower": row[lo_col] if lo_col in row.index else np.nan,
                            "upper": row[hi_col] if hi_col in row.index else np.nan,
                        })

            status = "✓" if has_native_intervals else "✓ (point only)"
            print(f"    {status} {model_name}")

        except Exception as e:
            print(f"    ✗ {model_name}: {e}")
            continue

    return pd.DataFrame(point_results), pd.DataFrame(ci_results)


def main():
    """Run all statsforecast models on all series types."""
    print("=== statsforecast Validation ===\n")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_point_results = []
    all_ci_results = []

    for series_type in SERIES_TYPES:
        print(f"Processing {series_type} series...")

        # Check if data exists
        data_path = DATA_DIR / f"{series_type}.csv"
        if not data_path.exists():
            print(f"  Error: Data file not found: {data_path}")
            print("  Run 'python generate_data.py' first to create the data files.")
            continue

        # Load data
        df = load_series(series_type)
        print(f"  Loaded {len(df)} observations")

        # Run forecasts
        point_df, ci_df = run_forecasts(df, series_type)

        all_point_results.append(point_df)
        all_ci_results.append(ci_df)

        print()

    # Combine and save results
    print("Writing results...")

    if all_point_results:
        point_forecasts = pd.concat(all_point_results, ignore_index=True)
        point_path = RESULTS_DIR / "point_forecasts.csv"
        point_forecasts.to_csv(point_path, index=False)
        print(f"  ✓ Point forecasts: {point_path}")

    if all_ci_results:
        ci_forecasts = pd.concat(all_ci_results, ignore_index=True)
        ci_path = RESULTS_DIR / "confidence_intervals.csv"
        ci_forecasts.to_csv(ci_path, index=False)
        print(f"  ✓ Confidence intervals: {ci_path}")

    print("\n=== statsforecast Complete ===")
    if all_point_results:
        print(f"Total forecasts: {len(point_forecasts)} point forecasts")


if __name__ == "__main__":
    main()
