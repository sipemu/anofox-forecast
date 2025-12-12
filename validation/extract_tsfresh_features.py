#!/usr/bin/env python3
"""Extract tsfresh features from synthetic time series data.

This script loads the synthetic time series from CSV files and extracts
tsfresh features, saving the results for comparison with the Rust implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

# Configuration
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "tsfresh"

SERIES_TYPES = [
    "stationary",
    "trend",
    "seasonal",
    "trend_seasonal",
    "seasonal_negative",
    "multiplicative_seasonal",
    "intermittent",
    "high_frequency",
    "structural_break",
    "long_memory",
    "noisy_seasonal",
]

# Feature configuration matching Rust features module
# Map tsfresh feature names to their parameters
FEATURE_CONFIG = {
    # Basic features (basic.rs)
    "mean": None,
    "variance": None,
    "standard_deviation": None,
    "median": None,
    "minimum": None,
    "maximum": None,
    "length": None,
    "abs_energy": None,
    "absolute_maximum": None,
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    "root_mean_square": None,
    "sum_values": None,
    "mean_n_absolute_max": [{"number_of_maxima": n} for n in [1, 3, 5, 7]],

    # Distribution features (distribution.rs)
    "skewness": None,
    "kurtosis": None,
    "quantile": [{"q": q} for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
    "large_standard_deviation": [{"r": 0.25}],
    "variance_larger_than_standard_deviation": None,
    "variation_coefficient": None,
    "symmetry_looking": [{"r": 0.05}],
    "ratio_beyond_r_sigma": [{"r": r} for r in [1.0, 1.5, 2.0, 2.5, 3.0]],

    # Autocorrelation features (autocorrelation.rs)
    "autocorrelation": [{"lag": lag} for lag in [1, 2, 3, 5, 10]],
    "partial_autocorrelation": [{"lag": lag} for lag in [1, 2, 3, 5]],
    "agg_autocorrelation": [
        {"f_agg": "mean", "maxlag": 10},
        {"f_agg": "var", "maxlag": 10},
        {"f_agg": "std", "maxlag": 10},
        {"f_agg": "median", "maxlag": 10},
    ],
    "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in [1, 2, 3]],

    # Counting features (counting.rs)
    "count_above_mean": None,
    "count_below_mean": None,
    "number_peaks": [{"n": n} for n in [1, 3, 5]],
    "number_crossing_m": [{"m": 0}],
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "index_mass_quantile": [{"q": q} for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
    "value_count": [{"value": 0}],
    "range_count": [{"min": -1, "max": 1}],

    # Entropy features (entropy.rs)
    "sample_entropy": None,
    "approximate_entropy": [{"m": 2, "r": 0.2}],
    "permutation_entropy": [{"tau": 1, "dimension": 3}],
    "binned_entropy": [{"max_bins": 10}],

    # Complexity features (complexity.rs)
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    "c3": [{"lag": lag} for lag in [1, 2, 3]],
    "lempel_ziv_complexity": [{"bins": 10}],

    # Trend features (trend.rs)
    "linear_trend": [
        {"attr": "slope"},
        {"attr": "intercept"},
        {"attr": "rvalue"},
        {"attr": "stderr"},
        {"attr": "pvalue"},
    ],
    "agg_linear_trend": [
        {"attr": "slope", "chunk_len": 10, "f_agg": "mean"},
        {"attr": "slope", "chunk_len": 10, "f_agg": "var"},
        {"attr": "intercept", "chunk_len": 10, "f_agg": "mean"},
        {"attr": "rvalue", "chunk_len": 10, "f_agg": "mean"},
    ],
    "ar_coefficient": [{"k": 10, "coeff": i} for i in [0, 1, 2, 3]],
    "augmented_dickey_fuller": [{"attr": "teststat"}],

    # Change features (change.rs)
    "change_quantiles": [
        {"ql": 0.0, "qh": 1.0, "isabs": True, "f_agg": "mean"},
        {"ql": 0.0, "qh": 1.0, "isabs": False, "f_agg": "mean"},
        {"ql": 0.25, "qh": 0.75, "isabs": True, "f_agg": "var"},
    ],
    "energy_ratio_by_chunks": [
        {"num_segments": 10, "segment_focus": i} for i in range(5)
    ],
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "ratio_value_number_to_time_series_length": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
}


def load_series(series_type: str) -> pd.DataFrame:
    """Load a time series from CSV and format for tsfresh."""
    path = DATA_DIR / f"{series_type}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # tsfresh expects columns: id, time, value
    df = df.rename(columns={"timestamp": "time", "value": "value"})
    df["id"] = series_type
    df["time"] = range(len(df))  # Use integer index for time

    return df[["id", "time", "value"]]


def extract_features_for_series(df: pd.DataFrame, series_type: str) -> pd.DataFrame:
    """Extract tsfresh features for a single series."""
    try:
        features = extract_features(
            df,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=FEATURE_CONFIG,
            n_jobs=1,
            disable_progressbar=True,
        )
    except Exception as e:
        print(f"  Warning: tsfresh extraction failed: {e}")
        return pd.DataFrame()

    # Flatten feature names and add series type
    result = []
    for col in features.columns:
        value = features[col].iloc[0]
        result.append({
            "series_type": series_type,
            "feature_name": col,
            "value": value,
        })

    return pd.DataFrame(result)


def main():
    """Extract tsfresh features from all series types."""
    print("=== tsfresh Feature Extraction ===\n")

    # Check if data exists
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Run 'python generate_data.py' first to create the data files.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for series_type in SERIES_TYPES:
        print(f"Processing {series_type}...")

        csv_path = DATA_DIR / f"{series_type}.csv"
        if not csv_path.exists():
            print(f"  Warning: Data file not found: {csv_path}")
            continue

        df = load_series(series_type)
        print(f"  Loaded {len(df)} observations")

        features_df = extract_features_for_series(df, series_type)
        if features_df.empty:
            print(f"  Warning: No features extracted")
            continue

        all_results.append(features_df)
        print(f"  Extracted {len(features_df)} features")

    if not all_results:
        print("\nError: No features extracted from any series")
        return

    # Save results
    results = pd.concat(all_results, ignore_index=True)
    output_path = RESULTS_DIR / "features.csv"
    results.to_csv(output_path, index=False)

    print(f"\n=== Extraction Complete ===")
    print(f"Output: {output_path}")
    print(f"Total features: {len(results)}")
    print(f"Unique feature types: {results['feature_name'].nunique()}")


if __name__ == "__main__":
    main()
