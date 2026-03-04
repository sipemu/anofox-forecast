#!/usr/bin/env python3
"""Diagnostic script for seasonality capture in AutoETS and AutoTheta.

Compares model selection and parameter values between statsforecast (NIXTLA)
and anofox-forecast (Rust) to identify sources of divergence.

Usage:
    python diagnose_seasonality.py          # needs statsforecast installed
    python diagnose_seasonality.py --rust-only  # compare Rust results only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RUST_RESULTS = Path(__file__).parent / "results" / "rust"
SF_RESULTS = Path(__file__).parent / "results" / "statsforecast"

SEASONAL_SERIES = [
    "seasonal",
    "trend_seasonal",
    "multiplicative_seasonal",
    "noisy_seasonal",
]

THETA_MODELS = ["Theta", "OptimizedTheta", "DynamicTheta", "DynamicOptimizedTheta", "AutoTheta"]
ETS_MODELS = ["AutoETS", "HoltWinters"]


def load_series(name: str) -> np.ndarray:
    """Load a time series from CSV."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["value"].values


def acf_at_lag(series: np.ndarray, lag: int) -> float:
    """Compute ACF at a specific lag."""
    n = len(series)
    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2) / n
    if c0 == 0:
        return 0.0
    ck = np.sum((series[:n - lag] - mean) * (series[lag:] - mean)) / n
    return ck / c0


def seasonal_test_stat(series: np.ndarray, period: int) -> tuple[float, float, float]:
    """Compute the ACF seasonal test statistic (statsforecast-compatible).

    Returns (test_statistic, z_90, z_95) so we can see which threshold passes.
    """
    from scipy.stats import norm

    acf_vals = [acf_at_lag(series, k) for k in range(period + 1)]
    r = acf_vals[1:]  # skip lag 0
    r_sq_sum = sum(x ** 2 for x in r[:-1])
    stat = np.sqrt((1.0 + 2.0 * r_sq_sum) / len(series))
    r_m = r[-1]  # ACF at lag m
    test_stat = abs(r_m) / stat
    z_90 = norm.ppf(0.90)
    z_95 = norm.ppf(0.95)
    return test_stat, z_90, z_95


def diagnose_rust_results():
    """Compare Rust forecasts against statsforecast for seasonal series."""
    rust_df = pd.read_csv(RUST_RESULTS / "point_forecasts.csv")
    sf_df = pd.read_csv(SF_RESULTS / "point_forecasts.csv")

    print("=" * 70)
    print("  Seasonality Diagnosis: Rust vs statsforecast")
    print("=" * 70)

    for series_name in SEASONAL_SERIES:
        print(f"\n{'─' * 70}")
        print(f"  Series: {series_name}")
        print(f"{'─' * 70}")

        # Load raw series for ACF test
        try:
            values = load_series(series_name)
            test_stat, z_90, z_95 = seasonal_test_stat(values, 12)
            print(f"  ACF seasonal test: stat={test_stat:.4f}  z_90={z_90:.4f}  z_95={z_95:.4f}")
            print(f"    passes z_90: {test_stat > z_90}    passes z_95: {test_stat > z_95}")
        except Exception as e:
            print(f"  ACF test error: {e}")

        # Compare forecasts for key models
        all_models = THETA_MODELS + ETS_MODELS
        for model in all_models:
            rust_row = rust_df[(rust_df["series_type"] == series_name) & (rust_df["model"] == model)]
            sf_row = sf_df[(sf_df["series_type"] == series_name) & (sf_df["model"] == model)]

            if rust_row.empty and sf_row.empty:
                continue

            rust_vals = rust_row["forecast"].values if not rust_row.empty else None
            sf_vals = sf_row["forecast"].values if not sf_row.empty else None

            print(f"\n  {model}:")
            if rust_vals is not None and sf_vals is not None and len(rust_vals) == len(sf_vals):
                mad = np.mean(np.abs(rust_vals - sf_vals))
                corr = np.corrcoef(rust_vals, sf_vals)[0, 1] if np.std(rust_vals) > 0 and np.std(sf_vals) > 0 else float("nan")
                print(f"    MAD: {mad:.4f}  Correlation: {corr:.4f}")
                print(f"    Rust first 3:  {rust_vals[:3]}")
                print(f"    SF   first 3:  {sf_vals[:3]}")
            elif rust_vals is not None:
                print(f"    Rust first 3:  {rust_vals[:3]}  (no statsforecast match)")
            elif sf_vals is not None:
                print(f"    SF   first 3:  {sf_vals[:3]}  (no Rust match)")


def diagnose_with_statsforecast():
    """Run statsforecast models and print detailed diagnostics."""
    try:
        from statsforecast.models import AutoETS, AutoTheta
    except ImportError:
        print("statsforecast not installed, skipping detailed diagnostics")
        return

    print("\n" + "=" * 70)
    print("  statsforecast Model Selection Details")
    print("=" * 70)

    for series_name in SEASONAL_SERIES:
        values = load_series(series_name)
        print(f"\n{'─' * 70}")
        print(f"  Series: {series_name}  (n={len(values)}, mean={np.mean(values):.2f}, std={np.std(values):.2f})")
        print(f"{'─' * 70}")

        # Format for statsforecast
        df = pd.DataFrame({
            "unique_id": [series_name] * len(values),
            "ds": pd.date_range("2020-01-01", periods=len(values), freq="MS"),
            "y": values,
        })

        # AutoETS
        try:
            model = AutoETS(season_length=12)
            model.fit(values)
            print(f"\n  AutoETS selected: {model.model_}")
        except Exception as e:
            print(f"\n  AutoETS error: {e}")

        # AutoTheta
        try:
            model = AutoTheta(season_length=12)
            model.fit(values)
            print(f"  AutoTheta selected: {model.model_}")
        except Exception as e:
            print(f"  AutoTheta error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose seasonality capture")
    parser.add_argument("--rust-only", action="store_true", help="Only compare Rust results")
    args = parser.parse_args()

    diagnose_rust_results()

    if not args.rust_only:
        diagnose_with_statsforecast()


if __name__ == "__main__":
    main()
