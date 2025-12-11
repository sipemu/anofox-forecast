# anofox-forecast

[![CI](https://github.com/sipemu/anofox-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-forecast/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/anofox-forecast.svg)](https://crates.io/crates/anofox-forecast)
[![Documentation](https://docs.rs/anofox-forecast/badge.svg)](https://docs.rs/anofox-forecast)
[![codecov](https://codecov.io/gh/sipemu/anofox-forecast/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/anofox-forecast)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Time series forecasting library for Rust.

Provides 35+ forecasting models, 76+ statistical features, seasonality decomposition, changepoint detection, anomaly detection, and time series clustering. Rust port of [anofox-time](https://github.com/DataZooDE/anofox-time) (C++).

## Features

- **Forecasting Models (35+)**
  - ARIMA and AutoARIMA with automatic order selection
  - Exponential Smoothing: Simple (SES), Holt's Linear, Holt-Winters
  - ETS (Error-Trend-Seasonal) state-space framework with AutoETS
  - Baseline methods: Naive, Seasonal Naive, Random Walk with Drift, Simple Moving Average
  - Theta method for forecasting
  - Intermittent demand models: Croston, ADIDA, TSB
  - Ensemble methods with multiple combination strategies

- **Time Series Feature Extraction (76+ features)**
  - Basic statistics (mean, variance, quantiles, energy, etc.)
  - Distribution features (skewness, kurtosis, symmetry)
  - Autocorrelation and partial autocorrelation
  - Entropy features (approximate, sample, permutation, binned, Fourier)
  - Complexity measures (C3, CID, Lempel-Ziv)
  - Trend analysis and stationarity tests (ADF, KPSS)

- **Seasonality & Decomposition**
  - STL (Seasonal-Trend decomposition using LOESS)
  - MSTL (Multiple Seasonal-Trend decomposition) for complex seasonality

- **Changepoint Detection**
  - PELT algorithm with O(n) average complexity
  - Multiple cost functions: L1, L2, Normal, Poisson

- **Anomaly Detection**
  - Statistical methods (IQR, z-score)
  - Automatic threshold selection
  - Seasonality-aware detection

- **Time Series Clustering**
  - Dynamic Time Warping (DTW) distance
  - K-Means clustering with multiple distance metrics
  - Elbow method for optimal cluster selection

- **Data Transformations**
  - Scaling: standardization, min-max, robust scaling
  - Box-Cox transformation with automatic lambda selection
  - Window functions: rolling mean, std, min, max, median
  - Exponential weighted moving averages

- **Model Evaluation & Validation**
  - Accuracy metrics: MAE, MSE, RMSE, MAPE, and more
  - Time series cross-validation
  - Residual testing and diagnostics

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
anofox-forecast = "0.1"
```

## Quick Start

### Creating a Time Series

```rust
use anofox_forecast::prelude::*;
use chrono::{TimeZone, Utc};

// Create timestamps
let timestamps: Vec<_> = (0..100)
    .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i))
    .collect();

// Create values
let values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 10.0).collect();

// Build the time series
let ts = TimeSeries::builder()
    .timestamps(timestamps)
    .values(values)
    .build()?;
```

### ARIMA Forecasting

```rust
use anofox_forecast::prelude::*;
use anofox_forecast::models::arima::Arima;

// Create and fit an ARIMA(1,1,1) model
let mut model = Arima::new(1, 1, 1)?;
model.fit(&ts)?;

// Generate forecasts with 95% confidence intervals
let forecast = model.predict_with_intervals(12, 0.95)?;

println!("Point forecasts: {:?}", forecast.values());
println!("Lower bounds: {:?}", forecast.lower());
println!("Upper bounds: {:?}", forecast.upper());
```

### Holt-Winters Forecasting

```rust
use anofox_forecast::models::exponential::HoltWinters;

// Create Holt-Winters with additive seasonality (period = 12)
let mut model = HoltWinters::additive(12)?;
model.fit(&ts)?;

let forecast = model.predict(24)?;
```

### Feature Extraction

```rust
use anofox_forecast::features::{mean, variance, skewness, approximate_entropy};

let values = ts.values();

let m = mean(values);
let v = variance(values);
let s = skewness(values);
let ae = approximate_entropy(values, 2, 0.2)?;

println!("Mean: {}, Variance: {}, Skewness: {}, ApEn: {}", m, v, s, ae);
```

### STL Decomposition

```rust
use anofox_forecast::seasonality::Stl;

// Decompose with seasonal period of 12
let stl = Stl::new(12)?;
let decomposition = stl.decompose(&ts)?;

println!("Trend: {:?}", decomposition.trend());
println!("Seasonal: {:?}", decomposition.seasonal());
println!("Remainder: {:?}", decomposition.remainder());
```

### Changepoint Detection

```rust
use anofox_forecast::changepoint::{Pelt, CostFunction};

let pelt = Pelt::new(CostFunction::L2, 10.0)?;
let changepoints = pelt.detect(&ts)?;

println!("Changepoints at indices: {:?}", changepoints);
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `TimeSeries` | Main data structure for univariate/multivariate time series |
| `Forecast` | Prediction results with optional confidence intervals |
| `CalendarAnnotations` | Holiday and regressor management |
| `AccuracyMetrics` | Model evaluation metrics (MAE, MSE, RMSE, MAPE, etc.) |

### Forecasting Models

| Model | Description |
|-------|-------------|
| `Arima` | ARIMA(p,d,q) model |
| `AutoArima` | Automatic ARIMA order selection |
| `Ses` | Simple Exponential Smoothing |
| `Holt` | Holt's Linear Trend method |
| `HoltWinters` | Holt-Winters with seasonal components |
| `Ets` | ETS state-space model |
| `AutoEts` | Automatic ETS model selection |
| `Naive` | Naive forecasting |
| `SeasonalNaive` | Seasonal naive forecasting |
| `Theta` | Theta method |
| `Croston` | Croston's method for intermittent demand |

### Feature Categories

| Category | Examples |
|----------|----------|
| Basic | `mean`, `variance`, `minimum`, `maximum`, `quantile` |
| Distribution | `skewness`, `kurtosis`, `variation_coefficient` |
| Autocorrelation | `autocorrelation`, `partial_autocorrelation` |
| Entropy | `approximate_entropy`, `sample_entropy`, `permutation_entropy` |
| Complexity | `c3`, `cid_ce`, `lempel_ziv_complexity` |
| Trend | `linear_trend`, `adf_test`, `ar_coefficient` |

## Dependencies

- [chrono](https://crates.io/crates/chrono) - Date and time handling
- [faer](https://crates.io/crates/faer) - Linear algebra operations
- [statrs](https://crates.io/crates/statrs) - Statistical distributions and functions
- [thiserror](https://crates.io/crates/thiserror) - Error handling
- [rand](https://crates.io/crates/rand) - Random number generation

## License

MIT License - see [LICENSE](LICENSE) for details.
