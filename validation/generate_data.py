#!/usr/bin/env python3
"""Generate synthetic time series data for validation testing.

Creates 25 types of time series:
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
12. Exponential trend - nonlinear exponential growth
13. Damped trend - trend that levels off over time
14. Strong seasonal - high amplitude, low noise seasonal
15. Quarterly seasonal - period=4 seasonal
16. Multiplicative trend seasonal - both trend and seasonal are multiplicative
17. Heteroscedastic - increasing variance over time
18. Random walk - pure random walk (unit root)
19. AR1 - autoregressive order 1 process
20. Outlier series - normal series with occasional large outliers
21. Step seasonal - non-sinusoidal (square wave) seasonal pattern
22. Bimodal seasonal - two peaks per seasonal cycle
23. Asymmetric seasonal - rapid rise, slow fall within each cycle
24. Seasonal trend break - seasonal pattern with a mid-series trend change
25. Low count - low-valued count data (small positive integers)
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


def generate_exponential_trend(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate exponential trend series.

    Nonlinear growth that accelerates over time.  Tests models that assume
    linear trends — they should struggle to capture the curvature.
    """
    t = np.arange(n)
    base = 10.0 * np.exp(0.02 * t)  # doubles roughly every 35 steps
    noise = rng.normal(0, 0.5, n) * (1.0 + 0.01 * t)  # slight heteroscedasticity
    return base + noise


def generate_damped_trend(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate damped trend series.

    A trend that levels off over time, matching ETS damped-trend models.
    y_t = level + b * phi^1 + b * phi^2 + ... + noise
    """
    level = 20.0
    b = 1.0
    phi = 0.9
    noise_std = 2.0

    values = np.zeros(n)
    cumulative_trend = 0.0
    for t in range(n):
        cumulative_trend += b * phi ** (t + 1)
        values[t] = level + cumulative_trend + rng.normal(0, noise_std)
    return values


def generate_strong_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate strong seasonal series with high signal-to-noise ratio.

    Large amplitude relative to noise — every model should capture this.
    Acts as a sanity check / baseline for seasonal detection.
    """
    mean = 100.0
    amplitude = 40.0
    noise_std = 2.0
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    return mean + seasonal + rng.normal(0, noise_std, n)


def generate_quarterly_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate quarterly seasonal series (period=4).

    Tests models with a short seasonal period — fewer seasonal states
    to estimate, but the pattern should be clearly captured.
    """
    period = 4
    mean = 60.0
    amplitude = 12.0
    noise_std = 3.0
    t = np.arange(n)
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    return mean + seasonal + rng.normal(0, noise_std, n)


def generate_multiplicative_trend_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate series with both multiplicative trend and multiplicative seasonality.

    y_t = level_t * seasonal_t * noise_t  where level has exponential growth.
    This is the hardest multiplicative case — both the trend and seasonal
    components interact multiplicatively.
    """
    t = np.arange(n)
    level = 30.0 * np.exp(0.01 * t)  # slow exponential growth
    seasonal_factor = 1.0 + 0.25 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    noise_factor = 1.0 + rng.normal(0, 0.03, n)
    return level * seasonal_factor * noise_factor


def generate_heteroscedastic(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate heteroscedastic series with increasing variance.

    The variance grows linearly with time.  Tests GARCH-type models
    and exposes models that assume constant variance.
    """
    mean = 50.0
    t = np.arange(n, dtype=float)
    variance = 1.0 + 0.2 * t  # variance grows from 1 to ~21
    noise = rng.normal(0, 1, n) * np.sqrt(variance)
    return mean + noise


def generate_random_walk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pure random walk (unit root process).

    y_t = y_{t-1} + epsilon_t.
    The Naive model should be optimal here.
    """
    increments = rng.normal(0, 1, n)
    return 50.0 + np.cumsum(increments)


def generate_ar1(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate AR(1) process with moderate persistence.

    y_t = phi * y_{t-1} + epsilon_t with phi=0.7.
    Tests ARIMA identification — should select AR(1) or close.
    """
    phi = 0.7
    noise_std = 2.0
    values = np.zeros(n)
    values[0] = 50.0
    for t in range(1, n):
        values[t] = 50.0 * (1 - phi) + phi * values[t - 1] + rng.normal(0, noise_std)
    return values


def generate_outlier_series(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate series with occasional large outliers.

    A smooth trend+seasonal base with ~5% of points replaced by
    outliers (3-5 sigma).  Tests model robustness to contamination.
    """
    t = np.arange(n)
    base = 50.0 + 0.3 * t + 8.0 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    noise = rng.normal(0, 2.0, n)
    values = base + noise

    # Insert outliers at ~5% of positions
    n_outliers = max(1, n // 20)
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    outlier_signs = rng.choice([-1, 1], size=n_outliers)
    outlier_magnitudes = rng.uniform(15, 30, size=n_outliers)
    values[outlier_idx] += outlier_signs * outlier_magnitudes

    return values


def generate_step_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate non-sinusoidal (square-wave) seasonal pattern.

    The seasonal component alternates between high and low levels within
    each period, rather than following a smooth sinusoid.  Tests whether
    models can capture non-smooth seasonal shapes.
    """
    mean = 50.0
    amplitude = 10.0
    noise_std = 2.0
    t = np.arange(n)
    # Square wave: high for first half of period, low for second half
    phase = (t % SEASONAL_PERIOD) / SEASONAL_PERIOD
    seasonal = np.where(phase < 0.5, amplitude, -amplitude)
    return mean + seasonal + rng.normal(0, noise_std, n)


def generate_bimodal_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate seasonal series with two peaks per cycle.

    Simulates e.g. retail with both summer and holiday peaks.
    Uses sum of two sine waves at different phases within the period.
    """
    mean = 60.0
    noise_std = 2.5
    t = np.arange(n)
    peak1 = 8.0 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)         # primary peak
    peak2 = 5.0 * np.sin(4 * np.pi * t / SEASONAL_PERIOD + 1.0)   # secondary peak (double freq)
    return mean + peak1 + peak2 + rng.normal(0, noise_std, n)


def generate_asymmetric_seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate asymmetric seasonal pattern: rapid rise, slow fall.

    Uses a sawtooth-like wave.  Real-world seasonal patterns are rarely
    perfectly symmetric, so this tests whether models capture asymmetry.
    """
    mean = 50.0
    amplitude = 12.0
    noise_std = 2.0
    t = np.arange(n)
    # Sawtooth via modular arithmetic — rapid linear rise, instant drop
    phase = (t % SEASONAL_PERIOD) / SEASONAL_PERIOD
    # Transform to rapid-rise/slow-fall using power function
    seasonal = amplitude * (1.0 - (1.0 - phase) ** 2) * 2.0 - amplitude
    return mean + seasonal + rng.normal(0, noise_std, n)


def generate_seasonal_trend_break(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate seasonal series with a mid-series trend change.

    First half has upward trend + seasonality, second half has
    downward trend + same seasonality.  Tests adaptive models.
    """
    t = np.arange(n)
    mid = n // 2
    amplitude = 10.0
    noise_std = 2.0
    seasonal = amplitude * np.sin(2 * np.pi * t / SEASONAL_PERIOD)

    trend = np.zeros(n)
    trend[:mid] = 0.5 * np.arange(mid)
    trend[mid:] = trend[mid - 1] - 0.5 * np.arange(n - mid)

    return 40.0 + trend + seasonal + rng.normal(0, noise_std, n)


def generate_low_count(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate low-valued count data (small positive integers).

    Simulates daily event counts (e.g. support tickets) with a slight
    weekly-like seasonal pattern.  Values are non-negative integers.
    """
    t = np.arange(n)
    rate = 5.0 + 2.0 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
    rate = np.maximum(rate, 0.5)  # keep rate positive
    values = rng.poisson(rate)
    return values.astype(float)


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
        ("seasonal_negative", generate_seasonal_negative),
        ("multiplicative_seasonal", generate_multiplicative_seasonal),
        ("intermittent", generate_intermittent),
        ("high_frequency", generate_high_frequency),
        ("structural_break", generate_structural_break),
        ("long_memory", generate_long_memory),
        ("noisy_seasonal", generate_noisy_seasonal),
        ("exponential_trend", generate_exponential_trend),
        ("damped_trend", generate_damped_trend),
        ("strong_seasonal", generate_strong_seasonal),
        ("quarterly_seasonal", generate_quarterly_seasonal),
        ("multiplicative_trend_seasonal", generate_multiplicative_trend_seasonal),
        ("heteroscedastic", generate_heteroscedastic),
        ("random_walk", generate_random_walk),
        ("ar1", generate_ar1),
        ("outlier_series", generate_outlier_series),
        ("step_seasonal", generate_step_seasonal),
        ("bimodal_seasonal", generate_bimodal_seasonal),
        ("asymmetric_seasonal", generate_asymmetric_seasonal),
        ("seasonal_trend_break", generate_seasonal_trend_break),
        ("low_count", generate_low_count),
    ]

    for name, generator in series_generators:
        values = generator(N_OBSERVATIONS, rng)
        save_series(name, timestamps, values)

    print()
    print(f"Data saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
