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

Provides 35+ forecasting models, 76+ statistical features, automatic model selection, ensemble methods, seasonality decomposition, changepoint detection, anomaly detection, and model serialization.

## Use Cases

**Need to run this on 10GB of data?** Use our [DuckDB extension](https://github.com/DataZooDE/anofox-forecast) for SQL-native forecasting at scale.

**Need to use this in a React Dashboard?** Use our [npm package](https://www.npmjs.com/package/@sipemu/anofox-forecast) for WebAssembly-powered forecasting in the browser.

```bash
npm install @sipemu/anofox-forecast
```

```javascript
import init, { TimeSeries, AutoForecaster, AutoEnsembleForecaster } from '@sipemu/anofox-forecast';

await init();

const ts = new TimeSeries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const model = new AutoForecaster();
model.fit(ts);
const forecast = model.predict(5);
console.log(forecast.values);
```

## Features

- **Forecasting Models (35+)**
  - ARIMA, SARIMA, and AutoARIMA with automatic order selection
  - Exponential Smoothing: SES, Holt's Linear, Holt-Winters, SeasonalES
  - ETS (Error-Trend-Seasonal) state-space framework with AutoETS
  - Baseline methods: Naive, Seasonal Naive, Random Walk with Drift, SMA, Window Average
  - Theta family: Theta, Optimized Theta, Dynamic Theta, AutoTheta
  - Intermittent demand: Croston, ADIDA, TSB, IMAPA
  - TBATS/AutoTBATS for complex seasonality
  - MFLES (Multiple Frequency Locally Estimated Scatterplot)
  - MSTL-based forecasting with configurable trend/seasonal methods
  - GARCH for volatility modeling
  - Exogenous regressor support across model families

- **Automatic Model Selection**
  - `AutoForecast`: Unified selection across ARIMA, ETS, and Theta families (parallel with `parallel` feature)
  - `AutoEnsemble`: Automatic ensemble of top-K best models
  - Selection by in-sample MSE or cross-validation error
  - `fit_predict()` convenience method on all models

- **Batch Processing & Parallelism**
  - `fit_predict_many()`: Fit one model across many series (parallel with `parallel` feature)
  - `fit_registry()`: Fit all registered models on a series (parallel)
  - `compare_models()` / `compare_registry()`: Parallel model comparison
  - Bootstrap sampling uses `par_iter` when `parallel` is enabled

- **Ensemble Methods**
  - Mean, Median, Weighted MSE, and Custom weight combination strategies
  - Automatic ensemble construction from model registry

- **Model Comparison & Evaluation**
  - `compare_models()`: Side-by-side model evaluation with timing
  - `compare_registry()`: Compare all registered models at once
  - Accuracy metrics: MAE, MSE, RMSE, MAPE, sMAPE, MASE, and more
  - Time series cross-validation with configurable strategies
  - Streaming CV aggregation with early stopping (`cross_validate_early_stop()`)
  - Residual diagnostics: Ljung-Box, Durbin-Watson, Jarque-Bera, Box-Pierce
  - `diagnose_residuals()`: Unified residual diagnostic report

- **Time Series Feature Extraction (76+ features)**
  - Basic statistics (mean, variance, quantiles, energy, etc.)
  - Distribution features (skewness, kurtosis, symmetry)
  - Autocorrelation and partial autocorrelation
  - Entropy features (approximate, sample, permutation, binned, Fourier)
  - Complexity measures (C3, CID, Lempel-Ziv)
  - Trend analysis and stationarity tests (ADF, KPSS)
  - Automated feature selection (variance threshold, correlation filter, top-K)

- **Spectral Analysis**
  - Welch's periodogram for reduced variance spectral estimation
  - For comprehensive periodicity detection, see [fdars](https://crates.io/crates/fdars-core)

- **Seasonality & Decomposition**
  - STL (Seasonal-Trend decomposition using LOESS) with `StlBuilder` for ergonomic configuration
  - MSTL (Multiple Seasonal-Trend decomposition) for complex seasonality

- **Changepoint Detection**
  - PELT algorithm with O(n) average complexity
  - Builder API: `Pelt::new(CostFunction::L2).min_size(5).penalty(5.0).detect(&data)`
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

- **Model Serialization** (optional `serde` feature)
  - Save/load models to JSON with `to_json()`/`from_json()`
  - Binary serialization with `to_bincode()`/`from_bincode()` for compact storage
  - File persistence with `save_to_file()`/`load_from_file()`
  - Round-trip serialization for all major model families

- **Missing Value Imputation**
  - Policy-based: Drop, Fill, ForwardFill, BackwardFill, FillMean, FillMedian, Interpolate
  - Advanced: moving average imputation, seasonal median imputation
  - Convenience: forward-backward fill, regressor imputation
  - Metadata: missing mask, per-dimension missing counts

- **Data Transformations**
  - Scaling: standardization, min-max, robust scaling
  - Box-Cox transformation with automatic lambda selection
  - Window functions: rolling mean, std, min, max, median
  - Exponential weighted moving averages

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
anofox-forecast = "0.4"
```

### Optional Features

```toml
[dependencies]
# Parallel AutoARIMA (4-8x speedup via rayon, opt-in for embedding contexts like DuckDB)
anofox-forecast = { version = "0.4", features = ["parallel"] }

# Model serialization (save/load to JSON)
anofox-forecast = { version = "0.4", features = ["serde"] }

# Probabilistic postprocessing (conformal, IDR, QRA — enabled by default)
anofox-forecast = { version = "0.4", default-features = false }  # to disable
```

| Feature | Default | Description |
|---------|---------|-------------|
| `postprocess` | Yes | Conformal prediction, IDR, QRA, historical simulation |
| `parallel` | No | Rayon-based parallelism for AutoARIMA, AutoForecast, batch processing, bootstrap, and cross-validation (not available on WASM) |
| `serde` | No | JSON and bincode serialization/deserialization for models |

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

### Automatic Model Selection

```rust
use anofox_forecast::prelude::*;
use anofox_forecast::models::auto_forecast::AutoForecast;

// Automatically selects the best model across ARIMA, ETS, and Theta
let mut model = AutoForecast::new();
model.fit(&ts)?;

let forecast = model.predict(12)?;
println!("Best model: {}", model.name());
```

### ARIMA Forecasting

```rust
use anofox_forecast::prelude::*;
use anofox_forecast::models::arima::ARIMA;

// Create and fit an ARIMA(1,1,1) model
let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts)?;

// Generate forecasts with 95% confidence intervals
let forecast = model.predict_with_intervals(12, 0.95)?;

println!("Point forecasts: {:?}", forecast.primary());
println!("Lower bounds: {:?}", forecast.lower_series(0));
println!("Upper bounds: {:?}", forecast.upper_series(0));
```

### Holt-Winters Forecasting

```rust
use anofox_forecast::models::exponential::HoltWinters;

// Create Holt-Winters with additive seasonality (period = 12)
let mut model = HoltWinters::additive(12);
model.fit(&ts)?;

let forecast = model.predict(24)?;
```

### Model Comparison

```rust
use anofox_forecast::models::{BoxedForecaster, ModelRegistry};
use anofox_forecast::utils::comparison::{compare_registry, ComparisonConfig};

// Compare all registered models side-by-side
let config = ComparisonConfig::default();
let table = compare_registry(&ts, &config)?;
println!("{}", table);
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
| `Forecaster` | Trait implemented by all forecasting models |
| `AccuracyMetrics` | Model evaluation metrics (MAE, MSE, RMSE, MAPE, etc.) |

### Forecasting Models

| Family | Models |
|--------|--------|
| **Auto Selection** | `AutoForecast`, `AutoEnsemble` |
| **ARIMA** | `ARIMA`, `SARIMA`, `AutoARIMA` |
| **Exponential Smoothing** | `SES`, `Holt`, `HoltWinters`, `SeasonalES`, `ETS`, `AutoETS` |
| **Theta** | `Theta`, `OptimizedTheta`, `DynamicTheta`, `AutoTheta` |
| **Baseline** | `Naive`, `Mean`, `SeasonalNaive`, `RandomWalkWithDrift`, `SMA`, `WindowAverage`, `SeasonalWindowAverage` |
| **Intermittent** | `Croston`, `TSB`, `ADIDA`, `IMAPA` |
| **Complex Seasonality** | `TBATS`, `AutoTBATS`, `MFLES`, `MSTLForecaster` |
| **Volatility** | `GARCH` |
| **Ensemble** | `Ensemble` (Mean, Median, Weighted MSE, Custom) |

### Utilities

| Function / Type | Description |
|-----------------|-------------|
| `compare_models()` | Compare forecasters on the same data with timing |
| `compare_registry()` | Compare all registered models at once |
| `cross_validate()` | Time series cross-validation (parallel with `parallel` feature) |
| `cross_validate_early_stop()` | CV with convergence-based early stopping |
| `StreamingCVAggregator` | Online metric aggregation using Welford's algorithm |
| `fit_predict_many()` | Batch fit-predict across multiple series |
| `bootstrap_forecast()` | Bootstrap confidence intervals for any model |
| `diagnose_residuals()` | Unified residual diagnostics (Ljung-Box, DW, Jarque-Bera) |
| `select_features()` | Automated feature selection (variance, correlation, top-K) |
| `to_json()` / `from_json()` | Model serialization (requires `serde` feature) |
| `to_bincode()` / `from_bincode()` | Binary serialization (requires `serde` feature) |

### Feature Categories

| Category | Examples |
|----------|----------|
| Basic | `mean`, `variance`, `minimum`, `maximum`, `quantile` |
| Distribution | `skewness`, `kurtosis`, `variation_coefficient` |
| Autocorrelation | `autocorrelation`, `partial_autocorrelation` |
| Entropy | `approximate_entropy`, `sample_entropy`, `permutation_entropy` |
| Complexity | `c3`, `cid_ce`, `lempel_ziv_complexity` |
| Trend | `linear_trend`, `adf_test`, `ar_coefficient` |
| Selection | `select_features`, `rank_features` |

### Postprocessing Types

| Type | Description |
|------|-------------|
| `PostProcessor` | Unified API for all postprocessing methods |
| `ConformalPredictor` | Distribution-free prediction intervals |
| `HistoricalSimulator` | Empirical error distribution |
| `IDRPredictor` | Isotonic Distributional Regression |
| `QRAPredictor` | Quantile Regression Averaging |

## Guides

- [Model Selection Guide](docs/model_selection_guide.md) — Which model to use for your data

## Dependencies

- [chrono](https://crates.io/crates/chrono) - Date and time handling
- [trueno](https://crates.io/crates/trueno) - Linear algebra operations
- [statrs](https://crates.io/crates/statrs) - Statistical distributions and functions
- [thiserror](https://crates.io/crates/thiserror) - Error handling
- [rand](https://crates.io/crates/rand) - Random number generation
- [rustfft](https://crates.io/crates/rustfft) - Fast Fourier Transform for spectral analysis

## Acknowledgments

The postprocessing module is a Rust port of [PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl). Feature extraction is inspired by [tsfresh](https://github.com/blue-yonder/tsfresh). Forecasting models are validated against [StatsForecast](https://github.com/Nixtla/statsforecast) by Nixtla. See [THIRDPARTY_NOTICE.md](THIRDPARTY_NOTICE.md) for full attribution and references to the research papers that inspired this implementation.

## License

MIT License - see [LICENSE](LICENSE) for details.
