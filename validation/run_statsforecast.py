#!/usr/bin/env python3
"""Run statsforecast models on synthetic time series data and M3 M3-reference fixture.

SYNTHETIC MODE (default / --synthetic):
    Loads the synthetic time series from CSV files and runs equivalent
    statsforecast models, saving the results for comparison with the Rust
    implementation.

M3 REFERENCE FIXTURE (--m3-reference):
    Runs AutoETS over M3 Yearly/Quarterly/Monthly using the same single-origin
    train/test split the Rust harness uses (train = all but last H steps, test =
    last H steps, no shuffle). Writes .planning/baselines/statsforecast_reference.json
    with a provenance block recording the pinned environment versions.

    IMPORTANT (D-06): This mode is a MANUAL MAINTAINER STEP. CI must never run it.
    Regenerate only when updating the pinned Python environment (uv.lock) or when
    the M3 preprocessing changes. After regeneration, verify that AutoETS M3-monthly
    MASE ≈ 0.93 (ACCUR-08) before committing the fixture.

    Run with:
        uv run python validation/run_statsforecast.py --m3-reference
"""

import json
import platform
import sys
from datetime import timezone, datetime
import importlib.metadata

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
    OptimizedTheta,
    DynamicTheta,
    DynamicOptimizedTheta,
    CrostonClassic,
    CrostonSBA,
    TSB,
    MSTL,
    ADIDA,
    IMAPA,
    SeasonalWindowAverage,
    HistoricAverage,
    WindowAverage,
    GARCH,
    AutoTheta,
    SeasonalExponentialSmoothing,
    SeasonalExponentialSmoothingOptimized,
    TBATS,
    AutoTBATS,
    MFLES,
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
    "seasonal_negative",
    "multiplicative_seasonal",
    "intermittent",
    "high_frequency",
    "structural_break",
    "long_memory",
    "noisy_seasonal",
    "exponential_trend",
    "damped_trend",
    "strong_seasonal",
    "quarterly_seasonal",
    "multiplicative_trend_seasonal",
    "heteroscedastic",
    "random_walk",
    "ar1",
    "outlier_series",
    "step_seasonal",
    "bimodal_seasonal",
    "asymmetric_seasonal",
    "seasonal_trend_break",
    "low_count",
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
        ("SARIMA_1_1_1_1_1_1_12", ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1), season_length=SEASONAL_PERIOD), True),
        ("AutoARIMA", AutoARIMA(season_length=SEASONAL_PERIOD), True),

        # ETS - has native intervals
        ("AutoETS", AutoETS(season_length=SEASONAL_PERIOD), True),

        # Theta variants - have native intervals
        ("Theta", Theta(season_length=SEASONAL_PERIOD), True),
        ("OptimizedTheta", OptimizedTheta(season_length=SEASONAL_PERIOD), True),
        ("DynamicTheta", DynamicTheta(season_length=SEASONAL_PERIOD), True),
        ("DynamicOptimizedTheta", DynamicOptimizedTheta(season_length=SEASONAL_PERIOD), True),
        ("AutoTheta", AutoTheta(season_length=SEASONAL_PERIOD), True),

        # Intermittent demand models - no native intervals
        ("Croston", CrostonClassic(), False),
        ("CrostonSBA", CrostonSBA(), False),
        ("TSB", TSB(alpha_d=0.1, alpha_p=0.1), False),
        ("ADIDA", ADIDA(), False),
        ("IMAPA", IMAPA(), False),

        # MSTL - decomposition-based forecasting (compare with MSTLForecaster)
        ("MSTLForecaster", MSTL(season_length=SEASONAL_PERIOD), True),

        # Seasonal Exponential Smoothing
        ("SeasonalES", SeasonalExponentialSmoothing(season_length=SEASONAL_PERIOD, alpha=0.1), False),

        # GARCH - volatility modeling (compare point forecasts)
        ("GARCH", GARCH(p=1, q=1), True),

        # TBATS - complex seasonality
        ("TBATS", TBATS(season_length=SEASONAL_PERIOD), True),
        ("AutoTBATS", AutoTBATS(season_length=SEASONAL_PERIOD), True),

        # MFLES - gradient boosted decomposition
        ("MFLES", MFLES(season_length=SEASONAL_PERIOD), True),

        # Window-based models
        ("SeasonalWindowAverage", SeasonalWindowAverage(season_length=SEASONAL_PERIOD, window_size=2), False),
        ("HistoricAverage", HistoricAverage(), False),
        ("WindowAverage", WindowAverage(window_size=12), False),
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


def _read_provenance() -> dict:
    """Read pinned environment versions at runtime (never hard-coded)."""
    def _ver(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "statsforecast_version": _ver("statsforecast"),
        "numpy_version": _ver("numpy"),
        "pandas_version": _ver("pandas"),
        "python_version": platform.python_version(),
        "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(),
        "m3_preprocessing": (
            "Single-origin expanding-window split: train = all values except last H steps, "
            "test = last H steps. Horizons: yearly=6, quarterly=8, monthly=18. "
            "TSF files read with Latin-1 byte decode. Series shorter than horizon+1 skipped. "
            "AutoETS season_length: yearly=1, quarterly=4, monthly=12. "
            "Reported MASE uses training-slice seasonal-naive denominator (same as Rust harness)."
        ),
    }


def _parse_tsf_m3(path: Path) -> tuple[str, int, list[tuple[str, list[float]]]]:
    """Parse a Monash .tsf file; return (frequency, horizon, [(id, values), ...]).

    Reads bytes as Latin-1 to handle accented category labels in M3 files.
    Values are guarded: non-parseable tokens are dropped (is_finite equivalent).
    """
    raw = path.read_bytes()
    content = "".join(chr(b) for b in raw)  # Latin-1 byte decode

    frequency = ""
    horizon = 0
    series = []
    in_data = False

    for line in content.splitlines():
        stripped = line.strip()
        if not in_data:
            if stripped.startswith("@frequency"):
                frequency = stripped[len("@frequency"):].strip()
            elif stripped.startswith("@horizon"):
                try:
                    horizon = int(stripped[len("@horizon"):].strip())
                except ValueError:
                    pass
            elif stripped.startswith("@data"):
                in_data = True
            continue

        parts = stripped.split(":", 2)
        if len(parts) < 3:
            continue
        series_id = parts[0]
        vals_str = parts[2]
        values = []
        for tok in vals_str.split(","):
            try:
                v = float(tok.strip())
                if not (v != v or v == float("inf") or v == float("-inf")):  # is_finite
                    values.append(v)
            except ValueError:
                pass
        if values:
            series.append((series_id, values))

    return frequency, horizon, series


def _mase_scale(train: list[float], period: int) -> float:
    """Seasonal-naive training-denominator (same as Rust mase_scale in loader.rs)."""
    if len(train) <= period:
        return 1.0
    diffs = [abs(train[i] - train[i - period]) for i in range(period, len(train))]
    mean_diff = sum(diffs) / len(diffs)
    return max(mean_diff, 1e-9)


def _compute_mase(train: list[float], actual: list[float], predicted: list[float], period: int) -> float:
    """MASE with training-slice denominator (competition-correct, matches Rust harness)."""
    denom = _mase_scale(train, period)
    h = min(len(actual), len(predicted))
    if h == 0:
        return float("nan")
    fmae = sum(abs(actual[i] - predicted[i]) for i in range(h)) / h
    return fmae / denom


def generate_m3_reference(data_dir: Path, output_path: Path) -> None:
    """Generate the M3 AutoETS reference fixture with provenance block (D-06).

    Reads M3 Y/Q/M .tsf files from data_dir, runs AutoETS (statsforecast) with
    the same single-origin train/test split the Rust harness uses (A3 alignment),
    computes MASE with training-slice denominator, and writes the fixture JSON.

    IMPORTANT (D-06): This is a MANUAL MAINTAINER STEP. CI must never invoke this
    function. Regenerate only on the pinned env (uv.lock) when necessary.
    """
    tsf_configs = [
        ("m3_yearly.tsf",    "yearly",    1,  6),
        ("m3_quarterly.tsf", "quarterly", 4,  8),
        ("m3_monthly.tsf",   "monthly",   12, 18),
    ]

    result: dict = {
        "provenance": _read_provenance(),
        "datasets": {"M3": {}},
    }

    for filename, freq_name, period, horizon in tsf_configs:
        tsf_path = data_dir / filename
        if not tsf_path.exists():
            print(f"  WARNING: {tsf_path} not found — skipping {freq_name}")
            continue

        print(f"  Processing M3-{freq_name} (period={period}, horizon={horizon})...")
        _freq, _h, all_series = _parse_tsf_m3(tsf_path)

        autoets_mases = []
        n_series_used = 0
        n_series_skipped = 0

        for series_id, values in all_series:
            n = len(values)
            if n <= horizon:
                n_series_skipped += 1
                continue

            # Single-origin expanding-window split: same as Rust CvFoldGenerator
            # with min_initial_window = n - horizon.
            # Train = values[0 .. n-horizon], Test = values[n-horizon .. n].
            train = values[: n - horizon]
            test = values[n - horizon:]

            if len(train) < 2 or len(test) == 0:
                n_series_skipped += 1
                continue

            # Build statsforecast-compatible DataFrame (single series).
            idx = pd.date_range("2000-01-01", periods=len(train), freq="MS")
            df = pd.DataFrame({"unique_id": series_id, "ds": idx, "y": train})

            try:
                sf = StatsForecast(
                    models=[AutoETS(season_length=period)],
                    freq="MS",
                    n_jobs=1,
                    verbose=False,
                )
                sf.fit(df)
                fc = sf.predict(h=horizon)
                # Column name is "AutoETS" in statsforecast 2.x
                pred_col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
                predicted = fc[pred_col].tolist()
                mase_val = _compute_mase(train, test, predicted, period)
                if mase_val == mase_val and mase_val > 0:  # is_finite and positive
                    autoets_mases.append(mase_val)
                    n_series_used += 1
                else:
                    n_series_skipped += 1
            except Exception as exc:
                print(f"    WARNING: AutoETS failed on {series_id}: {exc}")
                n_series_skipped += 1

        if autoets_mases:
            mean_mase = sum(autoets_mases) / len(autoets_mases)
        else:
            mean_mase = float("nan")

        print(f"    n_series_used={n_series_used} skipped={n_series_skipped} AutoETS_MASE={mean_mase:.4f}")

        result["datasets"]["M3"][freq_name] = {
            "n_series": n_series_used,
            "horizon": horizon,
            "seasonal_period": period,
            "autoets_mase": mean_mase,
        }

    # Verify monthly MASE is approximately 0.93 (ACCUR-08 anchor).
    monthly_mase = result["datasets"]["M3"].get("monthly", {}).get("autoets_mase", float("nan"))
    if monthly_mase != monthly_mase:
        print("\n  WARNING: monthly MASE is NaN — fixture may be invalid.")
    elif abs(monthly_mase - 0.93) > 0.10:
        print(
            f"\n  WARNING: monthly MASE={monthly_mase:.4f} deviates from expected ≈0.93 by "
            f"more than 0.10 — check preprocessing alignment (A3)."
        )
    else:
        print(f"\n  AutoETS M3-monthly MASE={monthly_mase:.4f} — within expected range of ≈0.93 ✓")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Fixture written to: {output_path}")


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
    if "--m3-reference" in sys.argv:
        # D-06: MANUAL MAINTAINER STEP — never run in CI.
        # Regenerate the M3 AutoETS reference fixture with pinned provenance.
        _data_dir = DATA_DIR
        _output = Path(__file__).parent.parent / ".planning" / "baselines" / "statsforecast_reference.json"
        print("=== M3 Reference Fixture Generation (D-06) ===\n")
        print(f"  Data dir : {_data_dir}")
        print(f"  Output   : {_output}\n")
        generate_m3_reference(_data_dir, _output)
        print("\n=== Done — review MASE output above before committing the fixture ===")
    else:
        main()
