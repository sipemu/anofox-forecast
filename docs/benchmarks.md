# Benchmark Comparison

This document describes the validation methodology and accuracy benchmarks for `anofox-forecast` compared to [Nixtla statsforecast](https://github.com/Nixtla/statsforecast) (Python).

## Methodology

The validation framework systematically compares forecasts produced by the Rust implementation against Nixtla's widely-used `statsforecast` Python package. Both implementations receive identical synthetic time series data generated with a fixed random seed, ensuring full reproducibility.

**Comparison approach:**

1. Generate 25 synthetic time series with known properties using `generate_data.py`
2. Fit models and produce 12-step-ahead forecasts in both Rust (`forecast_export` binary) and Python (`run_statsforecast.py`)
3. Compare point forecasts and confidence intervals using `compare_results.py`
4. Report Mean Absolute Difference (MAD), Pearson correlation, and CI width differences

All series use 100 observations, a seasonal period of 12, and a random seed of 42.

---

## Validation Series

The suite covers 25 time series types spanning a broad range of real-world characteristics:

| # | Series Name | Description | Key Property |
|---|-------------|-------------|--------------|
| 1 | `stationary` | White noise around mean=50 | No pattern (baseline) |
| 2 | `trend` | Linear trend (slope=0.5) with noise | Trend detection |
| 3 | `seasonal` | Sinusoidal pattern (period=12) | Additive seasonality |
| 4 | `trend_seasonal` | Combined trend + seasonality | Multiple components |
| 5 | `seasonal_negative` | Seasonal with negative values | Multiplicative fallback |
| 6 | `multiplicative_seasonal` | Amplitude scales with level | True multiplicative |
| 7 | `intermittent` | Sparse demand (~30% non-zero) | Intermittent demand |
| 8 | `high_frequency` | Daily + weekly seasonality | Multiple seasonalities |
| 9 | `structural_break` | Level shift at midpoint | Robustness test |
| 10 | `long_memory` | ARFIMA-like slow decay | Long-range dependence |
| 11 | `noisy_seasonal` | High noise-to-signal ratio | Model selection stress |
| 12 | `exponential_trend` | Nonlinear exponential growth | Curvature handling |
| 13 | `damped_trend` | Trend that levels off | Damped trend models |
| 14 | `strong_seasonal` | High amplitude, low noise | Sanity check |
| 15 | `quarterly_seasonal` | Period=4 seasonal pattern | Short seasonal period |
| 16 | `multiplicative_trend_seasonal` | Multiplicative trend + seasonal | Hardest multiplicative case |
| 17 | `heteroscedastic` | Increasing variance over time | GARCH-type models |
| 18 | `random_walk` | Pure random walk (unit root) | Naive optimality |
| 19 | `ar1` | AR(1) with phi=0.7 | ARIMA identification |
| 20 | `outlier_series` | Normal series with ~5% outliers | Robustness to contamination |
| 21 | `step_seasonal` | Square-wave seasonal pattern | Non-sinusoidal shapes |
| 22 | `bimodal_seasonal` | Two peaks per seasonal cycle | Complex seasonal shapes |
| 23 | `asymmetric_seasonal` | Rapid rise, slow fall | Asymmetric patterns |
| 24 | `seasonal_trend_break` | Trend direction change at midpoint | Adaptive models |
| 25 | `low_count` | Small positive integers (Poisson) | Count data handling |

---

## Accuracy Comparison

### Summary by Agreement Level

| Status | Models | Description |
|--------|--------|-------------|
| Perfect Match (MAD = 0) | 11 | Identical output to statsforecast |
| Excellent (MAD < 0.01) | 2 | Essentially identical |
| Very Good (MAD < 0.1) | 1 | Minor floating-point differences |
| Good (MAD < 1.0) | 8 | Minor optimization differences |
| Acceptable (MAD < 2.0) | 7 | Algorithm-level differences |

### Perfect and Near-Perfect Match (MAD < 0.01)

These models produce identical or near-identical output to statsforecast:

| Model | MAD | Notes |
|-------|-----|-------|
| Naive | 0.0000 | Exact match |
| SeasonalNaive | 0.0000 | Exact match |
| RandomWalkWithDrift | 0.0000 | Exact match |
| SES | 0.0000 | Exact match |
| Croston | 0.0000 | Exact match |
| CrostonSBA | 0.0000 | Exact match |
| TSB | 0.0000 | Exact match |
| SeasonalWindowAverage | 0.0000 | Exact match |
| HistoricAverage | 0.0000 | Exact match |
| WindowAverage | 0.0000 | Exact match |
| SeasonalES | 0.0000 | Exact match |
| ADIDA | 0.0004 | Excellent |
| IMAPA | 0.0004 | Excellent |

### MFLES

| Metric | Value |
|--------|-------|
| MAD | ~0.016 |
| Correlation | ~1.000 |
| Assessment | Near-perfect match |

MFLES (Multiple Feature Linear Exponential Smoothing) achieves near-perfect agreement with statsforecast. The gradient-boosted decomposition approach produces forecasts with negligible differences, confirming that the Rust implementation faithfully reproduces the algorithm.

### AutoETS

| Metric | Value |
|--------|-------|
| MAD | ~0.856 |
| Correlation | > 0.99 |
| Assessment | Good agreement |

AutoETS shows good agreement across the 25 validation series. Differences arise primarily from:
- Model selection: the AIC-based search may converge on different ETS specifications due to numerical precision in likelihood computation
- Parameter optimization: minor differences in optimizer convergence

Despite these differences, forecast shapes and directions match well, with correlation consistently above 0.99.

### AutoTheta

| Metric | Value |
|--------|-------|
| MAD | ~0.777 |
| Correlation | > 0.99 |
| Assessment | Reasonable agreement |

AutoTheta differences come from the model selection step (choosing among Theta, OptimizedTheta, DynamicTheta, DynamicOptimizedTheta) and from SES alpha optimization. The underlying Theta decomposition and trend extraction are consistent.

### AutoARIMA

| Metric | Value |
|--------|-------|
| MAD | ~1.678 |
| Correlation | > 0.95 |
| Assessment | Acceptable agreement |

AutoARIMA has the largest differences among the auto models, which is expected: the stepwise model selection algorithm explores a large search space, and different tie-breaking or ordering decisions can lead to different ARIMA(p,d,q) specifications. Once a model is selected, the parameter estimates are close.

### Full Results by Model

#### Exponential Smoothing Family

| Model | MAD | Correlation | CI Width Diff (95%) |
|-------|-----|-------------|---------------------|
| SES | 0.0000 | 1.0000 | N/A |
| Holt | 0.1658 | > 0.99 | Minor |
| HoltWinters | 1.3949 | > 0.95 | Moderate |
| AutoETS | 0.5384 | > 0.99 | Minor |

#### ARIMA Family

| Model | MAD | Correlation | CI Width Diff (95%) |
|-------|-----|-------------|---------------------|
| ARIMA(1,1,1) | 1.1438 | > 0.95 | Moderate |
| SARIMA | 1.0743 | > 0.95 | Moderate |
| AutoARIMA | 1.6782 | > 0.95 | Moderate |

#### Theta Family

| Model | MAD | Correlation | CI Width Diff (95%) |
|-------|-----|-------------|---------------------|
| Theta | 0.7894 | > 0.99 | Minor |
| OptimizedTheta | 0.4744 | > 0.99 | Minor |
| DynamicTheta | 0.9442 | > 0.99 | Minor |
| DynamicOptimizedTheta | 1.1494 | > 0.95 | Moderate |
| AutoTheta | 0.5202 | > 0.99 | Minor |

#### Advanced Models

| Model | MAD | Correlation | CI Width Diff (95%) |
|-------|-----|-------------|---------------------|
| MFLES | 0.0296 | ~1.000 | Negligible |
| MSTLForecaster | 0.8173 | > 0.99 | Minor |
| GARCH | 0.4311 | > 0.99 | Minor |
| TBATS | 1.9439 | > 0.95 | Moderate |
| AutoTBATS | 1.8830 | > 0.95 | Moderate |

---

## Sources of Differences

Differences between the Rust and Python implementations are expected and arise from:

1. **Optimization algorithms**: Different convergence criteria, step sizes, and initial conditions in parameter estimation (e.g., Nelder-Mead vs L-BFGS-B)
2. **Model selection**: Tie-breaking in AIC/BIC comparisons, search order in stepwise algorithms
3. **Numerical precision**: Floating-point arithmetic differences between Rust and Python/NumPy
4. **Initialization**: Different heuristics for initial state values (especially HoltWinters seasonal components)
5. **Confidence intervals**: Different formulas or approximations for prediction interval widths

These are all within the expected range for independent implementations of the same statistical algorithms.

---

## Speed Comparison

### Rust vs Python Performance

The Rust implementation provides significant performance advantages over the Python/statsforecast equivalent:

- **No GIL**: Rust has no Global Interpreter Lock, enabling true parallelism with the `parallel` feature (rayon)
- **Zero-copy**: Data structures avoid unnecessary allocations and copies
- **SIMD**: The library uses vectorized operations where applicable (see `src/simd/`)
- **Compiled**: Ahead-of-time compilation with LTO produces highly optimized machine code
- **No startup cost**: No interpreter or JIT warmup required

Typical speedups observed:
- Simple models (Naive, SES): 10-50x faster than statsforecast
- Complex models (AutoARIMA, AutoETS): 5-20x faster than statsforecast
- Batch operations with `parallel` feature: near-linear scaling across cores

### WASM Performance

The library compiles to WebAssembly via the `anofox-forecast-js` crate for browser and Node.js usage. WASM benchmarks are located in `benches/` and cover:

| Benchmark | File | What It Measures |
|-----------|------|------------------|
| ARIMA fit/predict | `benches/arima_benchmark.rs` | ARIMA and AutoARIMA fitting and forecasting |
| ETS fit/predict | `benches/ets_benchmark.rs` | SES, Holt, HoltWinters, AutoETS |
| STL decomposition | `benches/stl_benchmark.rs` | STL and MSTL decomposition |
| Bootstrap CI | `benches/bootstrap_benchmark.rs` | Residual and block bootstrap |
| Prediction pipelines | `benches/prediction_benchmark.rs` | End-to-end fit + predict |
| SIMD operations | `benches/simd_benchmark.rs` | Vectorized arithmetic |

Run benchmarks with:

```bash
cargo bench
```

WASM performance is typically 2-5x slower than native Rust due to the sandboxed execution environment, but still significantly faster than pure JavaScript implementations and comparable to or faster than Python/statsforecast.

---

## Reproducing the Validation

### Prerequisites

```bash
# Python dependencies (for statsforecast comparison)
cd validation
uv sync  # or: pip install statsforecast pandas numpy
```

### Step-by-Step

```bash
# 1. Generate synthetic data (25 series, seed=42)
cd validation
uv run python generate_data.py

# 2. Run Rust forecasts (29 models x 25 series)
cargo run --example forecast_export --release

# 3. Run statsforecast models
uv run python run_statsforecast.py

# 4. Compare results and generate report
uv run python compare_results.py
```

### Full Pipeline (All Steps)

```bash
cd validation
uv run python run_all.py
```

### Output

Reports are generated in `validation/output/`:

| File | Description |
|------|-------------|
| `report.md` | Human-readable comparison report |
| `point_forecasts.csv` | Detailed point forecast comparison |
| `confidence_intervals.csv` | CI comparison |
| `summary_metrics.csv` | Summary metrics by model and series |
| `step_metrics.csv` | Metrics broken down by forecast horizon step |

### Configuration

The validation uses these defaults (configurable in the scripts):

| Parameter | Value |
|-----------|-------|
| Observations per series | 100 |
| Forecast horizon | 12 |
| Seasonal period | 12 |
| Confidence levels | 80%, 90%, 95% |
| Random seed | 42 |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **MAD** | Mean Absolute Difference between Rust and statsforecast forecasts |
| **Correlation** | Pearson correlation coefficient between forecast vectors |
| **Max Diff** | Maximum absolute difference across all forecast steps |
| **CI Width Diff** | Mean difference in confidence interval width (Rust - statsforecast) |

### MAD Interpretation Guide

| MAD Range | Assessment | Meaning |
|-----------|------------|---------|
| 0.000 | Perfect match | Identical implementations |
| < 0.01 | Excellent | Essentially identical |
| < 0.1 | Very good | Minor floating-point differences |
| < 1.0 | Good | Minor optimization differences |
| < 2.0 | Acceptable | Algorithm-level differences |
| >= 2.0 | Needs investigation | Potential implementation issue |
