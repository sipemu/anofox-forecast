# Cookbook

Practical recipes for common time series forecasting tasks with `anofox-forecast`. Each recipe includes working Rust code that you can adapt to your own data.

## Table of Contents

- [Quick Start: Fit, Predict, Get Intervals](#quick-start-fit-predict-get-intervals)
- [Choosing the Right Model](#choosing-the-right-model)
- [Using AutoForecast for Automatic Model Selection](#using-autoforecast-for-automatic-model-selection)
- [Ensemble Forecasting with AutoEnsemble](#ensemble-forecasting-with-autoensemble)
- [Seasonal Data Handling](#seasonal-data-handling)
- [Cross-Validation for Model Selection](#cross-validation-for-model-selection)
- [Feature Extraction for Time Series Classification](#feature-extraction-for-time-series-classification)
- [Changepoint Detection Workflow](#changepoint-detection-workflow)
- [Residual Diagnostics](#residual-diagnostics)
- [Model Serialization](#model-serialization)
- [WASM Usage in Browser and Node.js](#wasm-usage-in-browser-and-nodejs)

---

## Quick Start: Fit, Predict, Get Intervals

The core workflow is the same across all models: create a `TimeSeries`, fit a model, and predict.

See also: `examples/quickstart.rs`

### ARIMA

```rust
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

// Create a time series
let timestamps: Vec<_> = (0..100)
    .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
    .collect();
let values: Vec<f64> = (0..100)
    .map(|i| 10.0 + 0.5 * i as f64 + 2.0 * (i as f64 * 0.3).sin())
    .collect();
let ts = TimeSeries::univariate(timestamps, values).unwrap();

// Fit an ARIMA(1,1,1) model
let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts).unwrap();

// Point forecast (10 steps ahead)
let forecast = model.predict(10).unwrap();
println!("Forecasts: {:?}", forecast.primary());

// Forecast with 95% confidence intervals
let forecast_ci = model.predict_with_intervals(10, 0.95).unwrap();
let lower = forecast_ci.lower_series(0).unwrap();
let upper = forecast_ci.upper_series(0).unwrap();

for i in 0..10 {
    println!(
        "h={}: {:.2} [{:.2}, {:.2}]",
        i + 1,
        forecast_ci.primary()[i],
        lower[i],
        upper[i]
    );
}
```

See also: `examples/forecasting/arima.rs`

### ETS (Exponential Smoothing)

```rust
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig};
use anofox_forecast::models::Forecaster;

// AutoETS selects the best ETS specification automatically
let config = AutoETSConfig::with_period(12);
let mut model = AutoETS::with_config(config);
model.fit(&ts).unwrap();

// See which model was selected
if let Some(spec) = model.selected_spec() {
    println!("Selected: ETS({:?}, {:?}, {:?})", spec.error, spec.trend, spec.seasonal);
}

let forecast = model.predict_with_intervals(12, 0.95).unwrap();
```

See also: `examples/forecasting/exponential.rs`

### Theta

```rust
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;

// Standard Theta method (won the M3 competition)
let mut model = Theta::new();
model.fit(&ts).unwrap();

println!("Alpha: {:.4}", model.alpha().unwrap());
println!("Slope: {:.4}", model.slope().unwrap());

let forecast = model.predict_with_intervals(10, 0.95).unwrap();
```

For seasonal data, use `Theta::seasonal(period)`:

```rust
let mut model = Theta::seasonal(12);
model.fit(&ts).unwrap();
let forecast = model.predict(12).unwrap();
```

See also: `examples/forecasting/theta.rs`

---

## Choosing the Right Model

Use this decision tree to select an appropriate model for your data:

```
Is the data intermittent (many zeros)?
├── Yes → Croston, TSB, ADIDA, or IMAPA
│         (Use Croston for classic intermittent demand,
│          TSB for obsolescence tracking)
└── No
    ├── Is there a clear seasonal pattern?
    │   ├── Yes
    │   │   ├── Single seasonality → HoltWinters, AutoETS(period), Theta::seasonal(period)
    │   │   └── Multiple seasonalities → MSTLForecaster, TBATS
    │   └── No
    │       ├── Is there a trend?
    │       │   ├── Yes, linear → Holt, ARIMA, Theta
    │       │   ├── Yes, damped → HoltLinearTrend::auto_damped(), ETS with damped trend
    │       │   └── Yes, exponential → Consider Box-Cox transform + model
    │       └── No trend
    │           ├── Stationary → SES, ARIMA(p,0,q), Naive
    │           └── Random walk → Naive, RandomWalkWithDrift
    │
    Is the data volatile with changing variance?
    ├── Yes → GARCH for volatility modeling
    └── No → Standard models above

    Not sure? → Use AutoForecast or AutoEnsemble
```

**Rules of thumb:**

- When in doubt, start with `AutoForecast` -- it tries AutoARIMA, AutoETS, and AutoTheta and picks the best one.
- For production systems, use `AutoEnsemble` to combine the top models for more robust predictions.
- Always validate with cross-validation before deploying (see [Cross-Validation](#cross-validation-for-model-selection)).

---

## Using AutoForecast for Automatic Model Selection

`AutoForecast` fits AutoARIMA, AutoETS, and AutoTheta, then selects the best model based on in-sample MSE or cross-validation error.

```rust
use anofox_forecast::models::auto_forecast::{AutoForecast, AutoForecastConfig, SelectionStrategy};
use anofox_forecast::models::Forecaster;

// Default: compares by in-sample MSE (fast)
let mut model = AutoForecast::new();
model.fit(&ts).unwrap();

println!("Selected model: {}", model.selected_model_name().unwrap());
println!("All candidate scores:");
for (name, score) in model.all_scores() {
    println!("  {}: {:.4}", name, score);
}

let forecast = model.predict_with_intervals(12, 0.95).unwrap();
```

### Seasonal AutoForecast

```rust
// Tell AutoForecast about the seasonal period
let mut model = AutoForecast::seasonal(12);
model.fit(&ts).unwrap();
```

### Cross-Validation Selection (More Robust)

```rust
let config = AutoForecastConfig::with_period(12)
    .with_selection(SelectionStrategy::CrossValidation);
let mut model = AutoForecast::with_config(config);
model.fit(&ts).unwrap();

// The selected model was validated across multiple time windows
println!("{}", model); // Display shows selected model and all scores
```

### Restricting Candidate Models

```rust
// Only consider ARIMA and ETS (skip Theta)
let config = AutoForecastConfig::default().without_theta();
let mut model = AutoForecast::with_config(config);
model.fit(&ts).unwrap();
```

---

## Ensemble Forecasting with AutoEnsemble

`AutoEnsemble` fits multiple model families, ranks them by in-sample MSE, and combines the top-K into a weighted ensemble.

```rust
use anofox_forecast::models::ensemble::{AutoEnsemble, AutoEnsembleConfig, CombinationMethod};
use anofox_forecast::models::Forecaster;

// Default: top-3 models, weighted by inverse MSE
let mut ensemble = AutoEnsemble::new();
ensemble.fit(&ts).unwrap();
let forecast = ensemble.predict(12).unwrap();
```

### Custom Configuration

```rust
let config = AutoEnsembleConfig::with_period(12)
    .with_top_k(5)                           // Use top 5 models
    .with_method(CombinationMethod::Median); // Median combination (robust to outlier forecasts)

let mut ensemble = AutoEnsemble::with_config(config);
ensemble.fit(&ts).unwrap();
let forecast = ensemble.predict_with_intervals(12, 0.95).unwrap();
```

### Manual Ensemble (Full Control)

For complete control over which models are combined:

```rust
use anofox_forecast::models::ensemble::{CombinationMethod, Ensemble};
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
use anofox_forecast::models::theta::Theta;

let models: Vec<Box<dyn Forecaster>> = vec![
    Box::new(Naive::new()),
    Box::new(SimpleExponentialSmoothing::auto()),
    Box::new(Theta::new()),
];

// Mean ensemble
let mut ensemble = Ensemble::new(models);
ensemble.fit(&ts).unwrap();

// Or weighted by inverse MSE
let models2: Vec<Box<dyn Forecaster>> = vec![
    Box::new(Naive::new()),
    Box::new(SimpleExponentialSmoothing::auto()),
    Box::new(Theta::new()),
];
let mut weighted = Ensemble::new(models2).with_method(CombinationMethod::WeightedMSE);
weighted.fit(&ts).unwrap();

println!("Weights: {:?}", weighted.weights());
let forecast = weighted.predict_with_intervals(10, 0.95).unwrap();
```

See also: `examples/forecasting/ensemble.rs`

---

## Seasonal Data Handling

### STL Decomposition

STL (Seasonal-Trend decomposition using LOESS) separates a series into trend, seasonal, and remainder components:

```rust
use anofox_forecast::seasonality::STL;

let period = 12;
let stl = STL::new(period);
let result = stl.decompose(&values).unwrap();

println!("Trend strength:    {:.4}", result.trend_strength());
println!("Seasonal strength: {:.4}", result.seasonal_strength());

// Access components
let trend = &result.trend;
let seasonal = &result.seasonal;
let remainder = &result.remainder;

// Verify: original = trend + seasonal + remainder
let error: f64 = values.iter()
    .zip(trend.iter().zip(seasonal.iter().zip(remainder.iter())))
    .map(|(y, (t, (s, r)))| (y - (t + s + r)).abs())
    .fold(0.0, f64::max);
assert!(error < 1e-10);
```

### Robust STL (Handles Outliers)

```rust
let stl_robust = STL::new(12).robust();
let result = stl_robust.decompose(&values_with_outliers).unwrap();
```

### Custom Smoothing Parameters

```rust
let stl = STL::new(12)
    .with_seasonal_smoothness(7)
    .with_trend_smoothness(21)
    .with_inner_iterations(3);
let result = stl.decompose(&values).unwrap();
```

### MSTL (Multiple Seasonal Periods)

For data with multiple seasonal patterns (e.g., hourly data with daily and weekly cycles):

```rust
use anofox_forecast::seasonality::MSTL;

let mstl = MSTL::new(vec![24, 168]); // Daily and weekly periods
if let Some(result) = mstl.decompose(&hourly_values) {
    println!("Trend strength: {:.4}", result.trend_strength());
    for (i, period) in result.seasonal_periods.iter().enumerate() {
        if let Some(strength) = result.seasonal_strength(i) {
            println!("Seasonal strength (period={}): {:.4}", period, strength);
        }
    }
}
```

### STL + Forecasting

Use `MSTLForecaster` to decompose and forecast each component separately:

```rust
use anofox_forecast::models::mstl_forecaster::MSTLForecaster;
use anofox_forecast::models::Forecaster;

let mut model = MSTLForecaster::new(12);
model.fit(&ts).unwrap();
let forecast = model.predict(12).unwrap();
```

See also: `examples/analysis/stl_decomposition.rs`

---

## Cross-Validation for Model Selection

Time series cross-validation respects temporal ordering: the training set always precedes the test set.

```rust
use anofox_forecast::models::baseline::{Naive, SeasonalNaive, SimpleMovingAverage};
use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig};

// Expanding window: training window grows with each fold
let config = CVConfig::expanding(30, 1)  // initial_window=30, horizon=1
    .with_step_size(1);

let result = cross_validate(&config, &ts, Naive::new).unwrap();
println!("Folds: {}", result.n_folds);
println!("Mean MAE:  {:.4} (+/- {:.4})", result.aggregated.mae, result.aggregated.mae_std);
println!("Mean RMSE: {:.4} (+/- {:.4})", result.aggregated.rmse, result.aggregated.rmse_std);
```

### Rolling Window

```rust
// Fixed-size window slides forward (better for detecting concept drift)
let config = CVConfig::rolling(30, 1)
    .with_step_size(1);

let result = cross_validate(&config, &ts, Naive::new).unwrap();
```

### Multi-Step Horizon

```rust
// Evaluate 5-step-ahead forecasts
let config = CVConfig::expanding(30, 5)
    .with_step_size(5);

let result = cross_validate(&config, &ts, Naive::new).unwrap();
```

### Comparing Models

```rust
let cv_config = CVConfig::expanding(30, 1)
    .with_step_size(2)
    .with_seasonal_period(12);

println!("{:<20} {:>10} {:>10}", "Model", "MAE", "RMSE");

if let Ok(r) = cross_validate(&cv_config, &ts, Naive::new) {
    println!("{:<20} {:>10.4} {:>10.4}", "Naive", r.aggregated.mae, r.aggregated.rmse);
}
if let Ok(r) = cross_validate(&cv_config, &ts, || SeasonalNaive::new(12)) {
    println!("{:<20} {:>10.4} {:>10.4}", "SeasonalNaive(12)", r.aggregated.mae, r.aggregated.rmse);
}
for window in [3, 5, 7] {
    if let Ok(r) = cross_validate(&cv_config, &ts, || SimpleMovingAverage::new(window)) {
        println!("{:<20} {:>10.4} {:>10.4}", format!("SMA({})", window), r.aggregated.mae, r.aggregated.rmse);
    }
}
```

### Per-Fold Analysis

```rust
let config = CVConfig::expanding(40, 1).with_step_size(5);
let result = cross_validate(&config, &ts, Naive::new).unwrap();

for (i, metrics) in result.fold_metrics.iter().enumerate() {
    println!("Fold {}: MAE={:.4}, RMSE={:.4}", i + 1, metrics.mae, metrics.rmse);
}
```

See also: `examples/validation/cross_validation.rs`

---

---

## Feature Extraction for Time Series Classification

Extract statistical features from time series for use in classification, clustering, or anomaly detection.

```rust
use anofox_forecast::features::{basic, change};

let series: Vec<f64> = (0..100)
    .map(|i| 10.0 + 0.5 * i as f64 + 3.0 * (i as f64 * 0.2).sin())
    .collect();

// Central tendency
let mean = basic::mean(&series);
let median = basic::median(&series);

// Dispersion
let variance = basic::variance(&series);
let std_dev = basic::standard_deviation(&series);

// Energy
let energy = basic::abs_energy(&series);

// Change characteristics
let mean_change = basic::mean_change(&series);
let mean_abs_change = basic::mean_abs_change(&series);
let abs_sum_changes = basic::absolute_sum_of_changes(&series);

// Second derivative (acceleration)
let second_deriv = basic::mean_second_derivative_central(&series);

// Top absolute values
let top5_mean = basic::mean_n_absolute_max(&series, 5);
```

### Entropy and Complexity Features

```rust
use anofox_forecast::features::{entropy, complexity};

// Entropy measures (higher = more complex/random)
let approx_entropy = entropy::approximate_entropy(&series, 2, 0.2);
let sample_entropy = entropy::sample_entropy(&series, 2, 0.2);
let perm_entropy = entropy::permutation_entropy(&series, 3, 1);

// Complexity measures
let c3 = complexity::c3(&series, 1);
let cid = complexity::cid_ce(&series, true);
```

### Autocorrelation Features

```rust
use anofox_forecast::features::autocorrelation;

// Autocorrelation at specific lags
let acf_1 = autocorrelation::autocorrelation(&series, 1);
let acf_12 = autocorrelation::autocorrelation(&series, 12);

// Partial autocorrelation
let pacf_1 = autocorrelation::partial_autocorrelation(&series, 1);
```

### Building a Feature Vector

```rust
// Combine features into a vector for ML models
let feature_vector = vec![
    basic::mean(&series),
    basic::variance(&series),
    basic::standard_deviation(&series),
    basic::mean_abs_change(&series),
    basic::mean_second_derivative_central(&series),
    basic::abs_energy(&series),
];

println!("Feature vector: {:?}", feature_vector);
```

See also: `examples/features/basic_features.rs`, `examples/features/entropy.rs`, `examples/features/complexity.rs`

---

## Changepoint Detection Workflow

Detect points where the statistical properties of a series change using the PELT algorithm.

```rust
use anofox_forecast::changepoint::{pelt_detect, CostFunction, PeltConfig};

// Series with two level shifts
let mut series: Vec<f64> = vec![10.0; 30];
series.extend(vec![50.0; 30]);
series.extend(vec![25.0; 30]);

// Detect changepoints with L2 cost (mean changes)
let config = PeltConfig::default().penalty(5.0);
let result = pelt_detect(&series, &config);

println!("Changepoints: {:?}", result.changepoints);
println!("Segments: {:?}", result.segments);
println!("Segment means: {:?}", result.segment_means(&series));
```

### Choosing the Penalty

The penalty controls sensitivity: higher values produce fewer changepoints.

```rust
// BIC penalty (good default, slightly conservative)
let n = series.len();
let config_bic = PeltConfig::with_bic_penalty(n);

// AIC penalty (less conservative, may overfit)
let config_aic = PeltConfig::with_aic_penalty();

// Manual penalty
let config_manual = PeltConfig::default().penalty(10.0);
```

### Different Cost Functions

```rust
// L2: detect mean changes (most common)
let config = PeltConfig::default()
    .cost_function(CostFunction::L2)
    .penalty(10.0);

// Normal: detect mean AND variance changes
let config = PeltConfig::default()
    .cost_function(CostFunction::Normal)
    .penalty(10.0);

// L1: more robust to outliers
let config = PeltConfig::default()
    .cost_function(CostFunction::L1)
    .penalty(10.0);
```

### Minimum Segment Length

Prevent detection of very short segments:

```rust
let config = PeltConfig::default()
    .penalty(5.0)
    .min_segment_length(10); // Segments must be at least 10 points
```

### Segment Analysis

```rust
let result = pelt_detect(&series, &config);

for (i, &(start, end)) in result.segments.iter().enumerate() {
    let segment = &series[start..end];
    let mean = segment.iter().sum::<f64>() / segment.len() as f64;
    println!("Segment {}: [{}, {}) mean={:.2}", i + 1, start, end, mean);
}
```

See also: `examples/analysis/changepoint.rs`

---

## Residual Diagnostics

After fitting a model, check whether residuals are white noise (no remaining patterns).

### Extract and Inspect Residuals

```rust
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;

let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts).unwrap();

if let Some(residuals) = model.residuals() {
    let valid: Vec<f64> = residuals.iter().filter(|r| !r.is_nan()).copied().collect();
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let variance = valid.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

    println!("Residuals: {} observations", valid.len());
    println!("Mean: {:.6} (should be near 0)", mean);
    println!("Std Dev: {:.6}", variance.sqrt());
}
```

### Ljung-Box Test (Autocorrelation in Residuals)

```rust
use anofox_forecast::validation::ljung_box;

if let Some(residuals) = model.residuals() {
    let valid: Vec<f64> = residuals.iter().filter(|r| !r.is_nan()).copied().collect();
    let result = ljung_box(&valid, Some(10), 0);

    println!("Ljung-Box Q: {:.4}", result.statistic);
    println!("p-value: {:.4}", result.p_value);

    if result.is_white_noise(0.05) {
        println!("PASS: Residuals are white noise (no autocorrelation)");
    } else {
        println!("FAIL: Significant autocorrelation detected");
        println!("Consider adding AR/MA terms or including seasonality");
    }
}
```

### Durbin-Watson Test (First-Order Autocorrelation)

```rust
use anofox_forecast::validation::durbin_watson;

if let Some(residuals) = model.residuals() {
    let valid: Vec<f64> = residuals.iter().filter(|r| !r.is_nan()).copied().collect();
    let dw = durbin_watson(&valid);

    println!("Durbin-Watson: {:.4}", dw.statistic);
    // DW near 0: strong positive autocorrelation
    // DW near 2: no autocorrelation (ideal)
    // DW near 4: strong negative autocorrelation
}
```

### Stationarity Tests

Test whether your series needs differencing before modeling:

```rust
use anofox_forecast::validation::{adf_test, kpss_test, test_stationarity};

let values = ts.primary_values();

// Combined ADF + KPSS (most robust)
let (adf, kpss, conclusion) = test_stationarity(values);
println!("ADF statistic: {:.4}, stationary={}", adf.statistic, adf.is_stationary);
println!("KPSS statistic: {:.4}, stationary={}", kpss.statistic, kpss.is_stationary);
println!("Conclusion: {}", conclusion);
```

### In-Sample Accuracy Metrics

```rust
use anofox_forecast::utils::calculate_metrics;

if let Some(fitted) = model.fitted_values() {
    let actual = ts.primary_values();

    // Filter out NaN values (models may have NaN for initial observations)
    let valid_start = fitted.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let actual_valid: Vec<f64> = actual[valid_start..].to_vec();
    let fitted_valid: Vec<f64> = fitted[valid_start..].to_vec();

    let metrics = calculate_metrics(&actual_valid, &fitted_valid, None).unwrap();
    println!("MAE:       {:.4}", metrics.mae);
    println!("RMSE:      {:.4}", metrics.rmse);
    println!("SMAPE:     {:.2}%", metrics.smape);
    println!("R-squared: {:.4}", metrics.r_squared);
}
```

### Using Display for Quick Summaries

The `AutoForecast` model implements `Display` for quick summaries:

```rust
use anofox_forecast::models::auto_forecast::AutoForecast;
use anofox_forecast::models::Forecaster;

let mut model = AutoForecast::new();
model.fit(&ts).unwrap();

// Print selected model and all candidate scores
println!("{}", model);
// Output:
// AutoForecast (selected: AutoETS)
// Candidate scores:
//   AutoETS: 2.3456
//   AutoTheta: 3.1234
//   AutoARIMA: 4.5678
```

See also: `examples/validation/diagnostics.rs`, `examples/validation/metrics.rs`

---

## Model Serialization

Serialize and deserialize models using the `serde` feature for persistence, caching, or transfer.

### Enable the Feature

```toml
[dependencies]
anofox-forecast = { version = "0.4", features = ["serde"] }
serde_json = "1.0"
```

### Serialize a Fitted Model

```rust
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;

let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts).unwrap();

// Serialize to JSON
let json = serde_json::to_string(&model).unwrap();

// Save to file
std::fs::write("model.json", &json).unwrap();
```

### Deserialize and Predict

```rust
// Load from file
let json = std::fs::read_to_string("model.json").unwrap();

// Deserialize
let loaded_model: ARIMA = serde_json::from_str(&json).unwrap();

// Predict without refitting
let forecast = loaded_model.predict(10).unwrap();
```

### Supported Models

The `serde` feature adds `Serialize` and `Deserialize` derives to core types including `ARIMA`, `AutoARIMA`, `ETS`, `Forecast`, and `AccuracyMetrics`. Check the type documentation for specific model support.

---

## WASM Usage in Browser and Node.js

The library provides WebAssembly bindings via the `anofox-forecast-js` crate, published as [`@sipemu/anofox-forecast`](https://www.npmjs.com/package/@sipemu/anofox-forecast) on npm.

### Installation

```bash
npm install @sipemu/anofox-forecast
```

### Browser Usage

```javascript
import init, { TimeSeries, NaiveForecaster, AutoETSForecaster } from '@sipemu/anofox-forecast';

// Initialize the WASM module
await init();

// Create a time series from values
const ts = new TimeSeries([10, 12, 14, 13, 15, 17, 16, 18, 20, 19]);

// Fit and predict with Naive
const naive = new NaiveForecaster();
naive.fit(ts);
const forecast = naive.predict(5);
console.log(forecast.values);  // [19, 19, 19, 19, 19]

// Fit and predict with AutoETS
const ets = new AutoETSForecaster();
ets.fit(ts);
const etsForecast = ets.predict(5);
console.log(etsForecast.values);
```

### Node.js Usage

```javascript
const { TimeSeries, AutoETSForecaster } = require('@sipemu/anofox-forecast');

const ts = new TimeSeries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const model = new AutoETSForecaster();
model.fit(ts);
const forecast = model.predict(5);
console.log(forecast.values);  // Approximately [11, 12, 13, 14, 15]
```

### Building from Source

To build the WASM package yourself:

```bash
# Install wasm-pack
cargo install wasm-pack

# Build the package
cd crates/anofox-forecast-js
wasm-pack build --target web --release

# The output is in pkg/
ls pkg/
```

### Performance Notes

- WASM runs in a sandboxed environment, typically 2-5x slower than native Rust
- Still significantly faster than pure JavaScript implementations
- The WASM binary is optimized with `-O3` and `wasm-opt`
- For processing many series, consider using the native Rust library via a server API instead

See also: the npm package README for the full JavaScript/TypeScript API reference.
