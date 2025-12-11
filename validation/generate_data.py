#!/usr/bin/env python3
"""Generate synthetic time series data for validation testing.

Creates 6 types of time series:
1. Stationary - white noise around a mean
2. Trend - linear trend with noise
3. Seasonal - seasonal pattern (period=12) with noise (additive)
4. Trend + Seasonal - combined trend and seasonality (additive)
5. Seasonal with negatives - seasonal pattern that goes negative (tests fallback)
6. Multiplicative seasonal - seasonal amplitude scales with level (true multiplicative)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
SEED = 42
N_OBSERVATIONS = 100
SEASONAL_PERIOD = 12
DATA_DIR = Path(__file__).parent / "data"


def generate_timestamps(n: int, start: str = "2020-01-01") -> list[datetime]:
    """Generate monthly timestamps starting from the given date."""
    start_date = datetime.fromisoformat(start)
    return [start_date + timedelta(days=30 * i) for i in range(n)]


def generate_stationary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate stationary series: white noise around a mean."""
    mean = 50.0
    std = 5.0
    return mean + rng.normal(0, std, n)


def generate_trend(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate trend series: linear trend with noise."""
    intercept = 10.0
    slope = 0.5
    noise_std = 3.0
    t = np.arange(n)
    return intercept + slope * t + rng.normal(0, noise_std, n)


def generate_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate seasonal series: seasonal pattern (period=12) with noise."""
    mean = 50.0
    amplitude = 10.0
    noise_std = 2.0
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    return mean + seasonal + rng.normal(0, noise_std, n)


def generate_trend_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate trend + seasonal series: combined trend and seasonality."""
    intercept = 20.0
    slope = 0.3
    amplitude = 8.0
    noise_std = 2.0
    t = np.arange(n)
    trend = slope * t
    seasonal = amplitude * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    return intercept + trend + seasonal + rng.normal(0, noise_std, n)


def generate_seasonal_negative(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate seasonal series that goes negative.

    This series has a low mean and high amplitude, causing values to go negative.
    This should trigger fallback from multiplicative to additive decomposition.
    """
    mean = 5.0  # Low mean
    amplitude = 10.0  # High amplitude relative to mean -> goes negative
    noise_std = 1.0
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    return mean + seasonal + rng.normal(0, noise_std, n)


def generate_multiplicative_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate true multiplicative seasonal series.

    In multiplicative seasonality, the seasonal amplitude scales with the level:
    y_t = level_t * seasonal_t * noise_t

    This means peaks and troughs are proportionally larger when the level is higher.
    """
    # Base level with trend
    intercept = 50.0
    slope = 0.5
    t = np.arange(n)
    level = intercept + slope * t  # Level increases from 50 to ~100

    # Multiplicative seasonal factors (centered around 1.0)
    # Factor ranges from 0.7 to 1.3 (±30% seasonal variation)
    seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)

    # Multiplicative noise (small relative variation)
    noise_factor = 1.0 + rng.normal(0, 0.02, n)  # ±2% noise

    # y = level * seasonal * noise
    return level * seasonal_factor * noise_factor


def save_series(name: str, timestamps: list[datetime], values: np.ndarray) -> Path:
    """Save a time series to CSV."""
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })
    filepath = DATA_DIR / f"{name}.csv"
    df.to_csv(filepath, index=False)
    print(f"  Saved {name}.csv ({len(values)} observations)")
    return filepath


def main():
    """Generate all synthetic time series and save to CSV files."""
    print("Generating synthetic time series data...")
    print(f"  Seed: {SEED}")
    print(f"  Observations: {N_OBSERVATIONS}")
    print(f"  Seasonal period: {SEASONAL_PERIOD}")
    print()

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize random generator with seed for reproducibility
    rng = np.random.default_rng(SEED)

    # Generate timestamps
    timestamps = generate_timestamps(N_OBSERVATIONS)

    # Generate and save each series type
    series_generators = [
        ("stationary", generate_stationary),
        ("trend", generate_trend),
        ("seasonal", generate_seasonal),
        ("trend_seasonal", generate_trend_seasonal),
        ("seasonal_negative", generate_seasonal_negative),  # Tests fallback to additive
        ("multiplicative_seasonal", generate_multiplicative_seasonal),  # True multiplicative
    ]

    for name, generator in series_generators:
        values = generator(N_OBSERVATIONS, rng)
        save_series(name, timestamps, values)

    print()
    print(f"Data saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
