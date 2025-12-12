#!/usr/bin/env python3
"""Generate synthetic time series data for validation testing.

Creates 11 types of time series:
1. Stationary - white noise around a mean
2. Trend - linear trend with noise
3. Seasonal - seasonal pattern (period=12) with noise (additive)
4. Trend + Seasonal - combined trend and seasonality (additive)
5. Seasonal with negatives - seasonal pattern that goes negative (tests fallback)
6. Multiplicative seasonal - seasonal amplitude scales with level (true multiplicative)
7. Intermittent - sparse demand data with zeros (for intermittent demand models)
8. High frequency - hourly data with daily + weekly seasonality (for MSTL)
9. Structural break - series with level shift (tests robustness)
10. Long memory - ARFIMA-like series with slow decay
11. Noisy seasonal - high noise-to-signal ratio seasonal
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


def generate_intermittent(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate intermittent demand series.

    Intermittent demand has many zeros with sporadic non-zero demands.
    This is typical in spare parts, slow-moving inventory, etc.

    Parameters create ~30% demand occurrence rate with variable demand sizes.
    """
    # Probability of demand occurring at each time point
    demand_prob = 0.3

    # Generate demand occurrences (Bernoulli)
    has_demand = rng.random(n) < demand_prob

    # Generate demand sizes when demand occurs (Poisson-like with minimum of 1)
    mean_demand = 5.0
    demand_sizes = rng.poisson(mean_demand, n) + 1  # +1 ensures minimum of 1 when demand occurs

    # Combine: 0 when no demand, demand_size when demand occurs
    series = np.where(has_demand, demand_sizes, 0).astype(float)

    return series


def generate_high_frequency(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate high frequency series with multiple seasonalities.

    Simulates hourly data with:
    - Daily seasonality (period=24)
    - Weekly seasonality (period=168)
    - Slight upward trend
    - Random noise

    This tests models that handle multiple seasonal patterns (e.g., MSTL).
    Note: Uses the same n as other series for consistency in validation.
    """
    daily_period = 24
    weekly_period = 168

    t = np.arange(n)

    # Daily pattern (stronger)
    daily = 5.0 * np.sin(2 * np.pi * t / daily_period)

    # Weekly pattern (weaker)
    weekly = 3.0 * np.sin(2 * np.pi * t / weekly_period)

    # Slight trend
    trend = 0.01 * t

    # Noise
    noise = rng.normal(0, 1.5, n)

    return 50.0 + trend + daily + weekly + noise


def generate_structural_break(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate series with a structural break (level shift).

    The series has a sudden level shift at the midpoint, which tests
    model robustness and changepoint detection capabilities.
    """
    mean_before = 50.0
    mean_after = 70.0  # +20 level shift
    noise_std = 3.0

    break_point = n // 2

    values = np.zeros(n)
    values[:break_point] = mean_before + rng.normal(0, noise_std, break_point)
    values[break_point:] = mean_after + rng.normal(0, noise_std, n - break_point)

    return values


def generate_long_memory(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate long memory (ARFIMA-like) series.

    Uses fractional differencing approximation to create a series with
    slowly decaying autocorrelations. This tests ARIMA variants and
    models that assume short-memory processes.

    The series is generated using an AR(1) process with high persistence
    combined with slowly decaying weights.
    """
    d = 0.3  # Fractional differencing parameter (0 < d < 0.5 for stationarity)

    # Generate using truncated infinite MA representation
    # y_t = sum_{k=0}^{K} psi_k * epsilon_{t-k}
    # where psi_k = Gamma(k+d) / (Gamma(k+1) * Gamma(d))

    K = min(100, n)  # Truncation for MA weights
    psi = np.zeros(K)
    psi[0] = 1.0
    for k in range(1, K):
        psi[k] = psi[k-1] * (k - 1 + d) / k

    # Generate white noise
    epsilon = rng.normal(0, 1, n + K)

    # Convolve to get long memory process
    values = np.zeros(n)
    for t in range(n):
        values[t] = np.sum(psi * epsilon[t:t+K][::-1])

    # Scale and shift
    values = 50.0 + 5.0 * values

    return values


def generate_noisy_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate seasonal series with high noise-to-signal ratio.

    The noise standard deviation is larger than the seasonal amplitude,
    making the seasonal pattern harder to detect. This tests model
    selection and robustness to noise.
    """
    mean = 50.0
    amplitude = 5.0  # Seasonal amplitude
    noise_std = 8.0  # Noise > amplitude (high noise-to-signal)

    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / SEASONAL_PERIOD)

    return mean + seasonal + rng.normal(0, noise_std, n)


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
        ("intermittent", generate_intermittent),  # Sparse demand for intermittent models
        ("high_frequency", generate_high_frequency),  # Multiple seasonalities (MSTL)
        ("structural_break", generate_structural_break),  # Level shift test
        ("long_memory", generate_long_memory),  # ARFIMA-like slow decay
        ("noisy_seasonal", generate_noisy_seasonal),  # High noise-to-signal
    ]

    for name, generator in series_generators:
        values = generator(N_OBSERVATIONS, rng)
        save_series(name, timestamps, values)

    print()
    print(f"Data saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
