#!/usr/bin/env python3
"""
Generate reference outputs from Nixtla statsforecast for comparison with Rust implementation.

This script generates JSON files containing:
- Test data with known exogenous effects
- Fitted coefficients from statsforecast models
- Forecasts for comparison

Usage:
    python scripts/generate_nixtla_reference.py

Output:
    tests/reference/arima_exog_reference.json
    tests/reference/mfles_exog_reference.json
    tests/reference/baseline_exog_reference.json
"""

import json
import numpy as np
from pathlib import Path

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, MFLES, Naive, SeasonalNaive
    HAS_STATSFORECAST = True
except ImportError:
    HAS_STATSFORECAST = False
    print("WARNING: statsforecast not installed. Install with: pip install statsforecast")

import pandas as pd


def generate_test_data_with_exog(n=100, seed=42):
    """Generate synthetic time series data with known exogenous effects."""
    np.random.seed(seed)

    # Create timestamps
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

    # Exogenous regressors
    x1 = np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly pattern
    x2 = np.linspace(0, 1, n)  # Linear trend regressor

    # True coefficients
    beta0 = 50.0  # Intercept
    beta1 = 5.0   # Coefficient for x1
    beta2 = 10.0  # Coefficient for x2

    # Base signal from exogenous
    exog_effect = beta0 + beta1 * x1 + beta2 * x2

    # Add AR(1) component
    ar_coef = 0.6
    noise = np.random.normal(0, 1, n)
    ar_component = np.zeros(n)
    for t in range(1, n):
        ar_component[t] = ar_coef * ar_component[t-1] + noise[t]

    # Combine
    y = exog_effect + ar_component

    return {
        "dates": dates.strftime("%Y-%m-%d").tolist(),
        "y": y.tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
        "true_coefficients": {
            "intercept": beta0,
            "x1": beta1,
            "x2": beta2
        },
        "ar_coef": ar_coef
    }


def run_statsforecast_arima_exog(data, horizon=10):
    """Run statsforecast AutoARIMA with exogenous regressors."""
    if not HAS_STATSFORECAST:
        return None

    n = len(data["y"])

    # Create dataframe for statsforecast
    df = pd.DataFrame({
        "unique_id": ["series1"] * n,
        "ds": pd.to_datetime(data["dates"]),
        "y": data["y"],
        "x1": data["x1"],
        "x2": data["x2"]
    })

    # Create future exogenous values
    future_dates = pd.date_range(
        start=df["ds"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )
    future_x1 = np.sin(2 * np.pi * (np.arange(n, n + horizon)) / 7).tolist()
    future_x2 = np.linspace(1, 1 + horizon/n, horizon).tolist()

    X_df = pd.DataFrame({
        "unique_id": ["series1"] * horizon,
        "ds": future_dates,
        "x1": future_x1,
        "x2": future_x2
    })

    # Fit model
    model = AutoARIMA(season_length=7)
    sf = StatsForecast(models=[model], freq="D")
    sf.fit(df)

    # Forecast with exogenous
    forecast = sf.predict(h=horizon, X_df=X_df)

    return {
        "model": "AutoARIMA",
        "horizon": horizon,
        "forecast": forecast["AutoARIMA"].tolist(),
        "future_x1": future_x1,
        "future_x2": future_x2,
        "future_dates": future_dates.strftime("%Y-%m-%d").tolist()
    }


def run_statsforecast_mfles_exog(data, horizon=10):
    """Run statsforecast MFLES with exogenous regressors."""
    if not HAS_STATSFORECAST:
        return None

    n = len(data["y"])

    # Create dataframe for statsforecast
    df = pd.DataFrame({
        "unique_id": ["series1"] * n,
        "ds": pd.to_datetime(data["dates"]),
        "y": data["y"],
        "x1": data["x1"],
        "x2": data["x2"]
    })

    # Create future exogenous values
    future_dates = pd.date_range(
        start=df["ds"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )
    future_x1 = np.sin(2 * np.pi * (np.arange(n, n + horizon)) / 7).tolist()
    future_x2 = np.linspace(1, 1 + horizon/n, horizon).tolist()

    X_df = pd.DataFrame({
        "unique_id": ["series1"] * horizon,
        "ds": future_dates,
        "x1": future_x1,
        "x2": future_x2
    })

    # Fit model
    model = MFLES(season_length=7)
    sf = StatsForecast(models=[model], freq="D")
    sf.fit(df)

    # Forecast with exogenous
    forecast = sf.predict(h=horizon, X_df=X_df)

    return {
        "model": "MFLES",
        "horizon": horizon,
        "forecast": forecast["MFLES"].tolist(),
        "future_x1": future_x1,
        "future_x2": future_x2,
        "future_dates": future_dates.strftime("%Y-%m-%d").tolist()
    }


def run_statsforecast_naive_exog(data, horizon=10):
    """Run statsforecast Naive with exogenous regressors."""
    if not HAS_STATSFORECAST:
        return None

    n = len(data["y"])

    # Create dataframe for statsforecast
    df = pd.DataFrame({
        "unique_id": ["series1"] * n,
        "ds": pd.to_datetime(data["dates"]),
        "y": data["y"],
        "x1": data["x1"],
        "x2": data["x2"]
    })

    # Create future exogenous values
    future_dates = pd.date_range(
        start=df["ds"].max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )
    future_x1 = np.sin(2 * np.pi * (np.arange(n, n + horizon)) / 7).tolist()
    future_x2 = np.linspace(1, 1 + horizon/n, horizon).tolist()

    X_df = pd.DataFrame({
        "unique_id": ["series1"] * horizon,
        "ds": future_dates,
        "x1": future_x1,
        "x2": future_x2
    })

    # Fit model
    model = Naive()
    sf = StatsForecast(models=[model], freq="D")
    sf.fit(df)

    # Forecast with exogenous
    forecast = sf.predict(h=horizon, X_df=X_df)

    return {
        "model": "Naive",
        "horizon": horizon,
        "forecast": forecast["Naive"].tolist(),
        "future_x1": future_x1,
        "future_x2": future_x2,
        "future_dates": future_dates.strftime("%Y-%m-%d").tolist()
    }


def main():
    output_dir = Path("tests/reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test data
    print("Generating test data...")
    data = generate_test_data_with_exog(n=100, seed=42)

    # Save test data
    with open(output_dir / "test_data_exog.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved test data to {output_dir / 'test_data_exog.json'}")

    if not HAS_STATSFORECAST:
        print("\nInstall statsforecast to generate reference outputs:")
        print("  pip install statsforecast")
        return

    # Run models and save results
    print("\nRunning AutoARIMA with exogenous...")
    arima_result = run_statsforecast_arima_exog(data)
    if arima_result:
        with open(output_dir / "arima_exog_reference.json", "w") as f:
            json.dump(arima_result, f, indent=2)
        print(f"Saved ARIMA reference to {output_dir / 'arima_exog_reference.json'}")

    print("Running MFLES with exogenous...")
    mfles_result = run_statsforecast_mfles_exog(data)
    if mfles_result:
        with open(output_dir / "mfles_exog_reference.json", "w") as f:
            json.dump(mfles_result, f, indent=2)
        print(f"Saved MFLES reference to {output_dir / 'mfles_exog_reference.json'}")

    print("Running Naive with exogenous...")
    naive_result = run_statsforecast_naive_exog(data)
    if naive_result:
        with open(output_dir / "naive_exog_reference.json", "w") as f:
            json.dump(naive_result, f, indent=2)
        print(f"Saved Naive reference to {output_dir / 'naive_exog_reference.json'}")

    print("\nDone! Reference files generated in tests/reference/")


if __name__ == "__main__":
    main()
