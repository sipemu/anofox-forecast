# anofox-forecast

[![CI](https://github.com/sipemu/anofox-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/sipemu/anofox-forecast/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/anofox-forecast.svg)](https://crates.io/crates/anofox-forecast)
[![Documentation](https://docs.rs/anofox-forecast/badge.svg)](https://docs.rs/anofox-forecast)
[![codecov](https://codecov.io/gh/sipemu/anofox-forecast/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/anofox-forecast)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Technical Depth](https://img.shields.io/badge/technical%20depth-A-brightgreen.svg)](docs/API_REFERENCE.md)
[![Code Quality](https://img.shields.io/badge/code%20quality-production--ready-brightgreen.svg)](docs/API_REFERENCE.md)

> *Technical depth grading and code quality analysis powered by [pmat](https://github.com/paiml/paiml-mcp-agent-toolkit)*

Time series forecasting library for Rust.

Provides 35+ forecasting models, 76+ statistical features, seasonality decomposition, changepoint detection, anomaly detection, and bootstrap confidence intervals.

## Use Cases

**Need to run this on 10GB of data?** Use our [DuckDB extension](https://github.com/DataZooDE/anofox-forecast) for SQL-native forecasting at scale.

**Need to use this in a React Dashboard?** Use our [npm package](https://www.npmjs.com/package/@sipemu/anofox-forecast) for WebAssembly-powered forecasting in the browser.

```bash
npm install @sipemu/anofox-forecast
```

```javascript
import init, { TimeSeries, NaiveForecaster, AutoETSForecaster } from '@sipemu/anofox-forecast';

await init();

const ts = new TimeSeries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const model = new AutoETSForecaster();
model.fit(ts);
const forecast = model.predict(5);
console.log(forecast.values);  // [11, 12, 13, 14, 15] (approx)
```

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

- **Spectral Analysis**
  - Welch's periodogram for reduced variance spectral estimation
  - For comprehensive periodicity detection, see [fdars](https://crates.io/crates/fdars-core)

- **Changepoint Detection**
  - PELT algorithm with O(n) average complexity
  - Multiple cost functions: L1, L2, Normal, Poisson, LinearTrend, MeanVariance, Cusum

- **Anomaly Detection**
  - Statistical methods (IQR, z-score)
  - Automatic threshold selection
  - Seasonality-aware detection

- **Bootstrap Confidence Intervals**
  - Residual bootstrap and block bootstrap methods
  - Empirical confidence intervals for any model
  - Configurable sample size and reproducibility

- **Probabilistic Postprocessing**
  - Conformal Prediction: Distribution-free intervals with coverage guarantees
  - Historical Simulation: Non-parametric empirical error distribution
  - Normal Predictor: Gaussian error assumption baseline
  - IDR: Isotonic Distributional Regression (state-of-the-art calibration)
  - QRA: Quantile Regression Averaging for ensemble combining
  - Backtesting: Rolling/expanding window evaluation with horizon-aware calibration

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
anofox-forecast = "0.4"
```

For parallel AutoARIMA (4-8x speedup):
```toml
[dependencies]
anofox-forecast = { version = "0.3", features = ["parallel"] }
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

### Spectral Analysis

```rust
use anofox_forecast::detection::welch_periodogram;

// Welch's periodogram with overlapping windows
let psd = welch_periodogram(&values, 64, 0.5);

// Find dominant period
if let Some((period, power)) = psd.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
    println!("Dominant period: {}, power: {:.4}", period, power);
}
```

> For comprehensive periodicity detection (ACF, FFT, Autoperiod, CFD-Autoperiod, SAZED),
> see the [fdars](https://crates.io/crates/fdars-core) crate.

### Probabilistic Postprocessing

```rust
use anofox_forecast::postprocess::{PostProcessor, PointForecasts, BacktestConfig};

// Historical forecasts and actuals for calibration
let train_forecasts = PointForecasts::from_values(train_f);
let train_actuals = vec![/* ... */];

// Create a conformal predictor with 90% coverage
let processor = PostProcessor::conformal(0.90);

// Backtest with horizon-aware calibration
let config = BacktestConfig::new()
    .initial_window(100)
    .step(10)
    .horizon(7)
    .horizon_aware(true);

let results = processor.backtest(&train_forecasts, &train_actuals, config)?;
println!("Coverage: {:.1}%", results.coverage() * 100.0);

// Train calibrated model and predict
let trained = processor.train(&train_forecasts, &train_actuals)?;
let new_forecasts = PointForecasts::from_values(new_f);
let intervals = processor.predict_intervals(&trained, &new_forecasts)?;

println!("Lower: {:?}", intervals.lower());
println!("Upper: {:?}", intervals.upper());
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

### Postprocessing Types

| Type | Description |
|------|-------------|
| `PostProcessor` | Unified API for all postprocessing methods |
| `PointForecasts` | Wrapper for point forecast values |
| `QuantileForecasts` | Multi-quantile forecast container |
| `PredictionIntervals` | Lower/upper bound intervals |
| `BacktestConfig` | Configuration for rolling/expanding backtests |
| `BacktestResult` | Backtest metrics with per-horizon analysis |
| `ConformalPredictor` | Distribution-free prediction intervals |
| `HistoricalSimulator` | Empirical error distribution |
| `IDRPredictor` | Isotonic Distributional Regression |
| `QRAPredictor` | Quantile Regression Averaging |

## Dependencies

- [chrono](https://crates.io/crates/chrono) - Date and time handling
- [faer](https://crates.io/crates/faer) - Linear algebra operations
- [statrs](https://crates.io/crates/statrs) - Statistical distributions and functions
- [thiserror](https://crates.io/crates/thiserror) - Error handling
- [rand](https://crates.io/crates/rand) - Random number generation
- [rustfft](https://crates.io/crates/rustfft) - Fast Fourier Transform for spectral analysis

## Acknowledgments

The postprocessing module is a Rust port of [PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl). Feature extraction is inspired by [tsfresh](https://github.com/blue-yonder/tsfresh). Forecasting models are validated against [StatsForecast](https://github.com/Nixtla/statsforecast) by Nixtla. See [THIRDPARTY_NOTICE.md](THIRDPARTY_NOTICE.md) for full attribution and references to the research papers that inspired this implementation.

## License

MIT License - see [LICENSE](LICENSE) for details.
