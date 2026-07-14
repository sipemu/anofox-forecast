---
name: forecast
description: How to create a TimeSeries, fit a forecasting model, and generate predictions with anofox-forecast
user_invocable: true
---

# Forecasting with anofox-forecast

## Quick Start

```rust
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

// 1. Build a TimeSeries
let timestamps: Vec<_> = (0..100)
    .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
    .collect();
let values: Vec<f64> = (0..100).map(|i| 50.0 + 0.5 * i as f64).collect();
let ts = TimeSeries::univariate(timestamps, values).unwrap();

// 2. Fit a model
let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts).unwrap();

// 3. Predict
let forecast = model.predict(12).unwrap();
for (i, val) in forecast.primary().iter().enumerate() {
    println!("h={}: {:.2}", i + 1, val);
}

// 4. Predict with confidence intervals
let fc = model.predict_with_intervals(12, 0.95).unwrap();
let lower = fc.lower_series(0).unwrap();
let upper = fc.upper_series(0).unwrap();
```

## Available Models

### Baseline
```rust
use anofox_forecast::models::baseline::{Naive, SeasonalNaive, HistoricAverage, WindowAverage};

Naive::new()                    // Repeat last value
SeasonalNaive::new(12)          // Repeat value from same season
HistoricAverage::new()          // Mean of all history
WindowAverage::new(5)           // Moving average
```

### Exponential Smoothing
```rust
use anofox_forecast::models::exponential::{
    SimpleExponentialSmoothing, HoltLinearTrend, HoltWinters, ETS, AutoETS,
    ETSSpec, ErrorType, TrendType, SeasonalType,
};

SimpleExponentialSmoothing::auto()
HoltLinearTrend::auto()
HoltLinearTrend::auto_damped()
HoltWinters::auto(12, SeasonalType::Additive)
ETS::new(ETSSpec::new(ErrorType::Additive, TrendType::Additive, SeasonalType::Additive), 12)
AutoETS::new()                  // Auto-select best ETS model
AutoETS::with_period(12)        // Auto-select with known period
AutoETS::non_seasonal()         // Restrict to non-seasonal models
```

### ARIMA
```rust
use anofox_forecast::models::arima::{ARIMA, SARIMA, AutoARIMA};

ARIMA::new(1, 1, 1)             // ARIMA(p, d, q)
ARIMA::ar(2)                    // AR(2) = ARIMA(2,0,0)
ARIMA::ma(1)                    // MA(1) = ARIMA(0,0,1)
SARIMA::new(1, 1, 1, 1, 1, 0, 12) // SARIMA(1,1,1)(1,1,0)[12]
AutoARIMA::new()                // Auto-select best ARIMA order
```

### Theta
```rust
use anofox_forecast::models::theta::{Theta, AutoTheta, OptimizedTheta, DynamicTheta};

Theta::new()
AutoTheta::new()
OptimizedTheta::new()
DynamicTheta::new(0.5)          // With elasticity parameter
```

### MSTL (Multi-Seasonal)
```rust
use anofox_forecast::models::mstl_forecaster::MSTLForecaster;

MSTLForecaster::new(vec![7])          // Single period
MSTLForecaster::new(vec![7, 365])     // Multiple periods
```

### Intermittent Demand
```rust
use anofox_forecast::models::intermittent::{Croston, TSB, ADIDA, IMAPA};

Croston::new()
TSB::new()
```

### Distributional (streaming, mixture output — `distributional` feature)

`LaplaceForecaster` is a streaming per-observation ensemble that emits a `Vec<GaussianMixture>` per forecast. See the dedicated **[laplace-distributional skill](laplace-distributional.md)** for the full API surface — quick pointer:

```rust
use anofox_forecast::models::laplace::{LaplaceForecaster, MultiScaleLaplace};
use anofox_forecast::models::{Forecaster, DistributionalForecaster};

// Baseline
let mut f = LaplaceForecaster::new().auto();
f.fit(&ts)?;
let dists = f.forecast_dist(12)?;                     // Vec<GaussianMixture>

// v0.15.4 best-tuned config on fev-27 (moves us competitive with Nixtla auto_ets)
let mut f = MultiScaleLaplace::skaters(H)
    .with_scoring_horizon()
    .with_scoring_window(14);
if period >= 2 { f = f.with_period(period); }
```

## Forecaster Trait (all models implement this)

```rust
trait Forecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()>;
    fn predict(&self, horizon: usize) -> Result<Forecast>;
    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast>;
    fn fitted_values(&self) -> Option<&[f64]>;
    fn residuals(&self) -> Option<&[f64]>;
    fn name(&self) -> &str;
    fn supports_exog(&self) -> bool;
    fn has_exog(&self) -> bool;
    fn exog_names(&self) -> Option<&[String]>;
    fn exog_coefficients(&self) -> Option<&OLSResult>;  // OLS pre-regression coefficients
    fn predict_with_exog(&self, horizon: usize, future: &HashMap<String, Vec<f64>>) -> Result<Forecast>;
}
```

## Model Comparison

```rust
use anofox_forecast::models::{ModelSpec, ModelRegistry};
use anofox_forecast::utils::comparison::{compare_models, ComparisonConfig};

let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
    ("Naive", Box::new(|| Box::new(Naive::new()))),
    ("ARIMA(1,1,1)", Box::new(|| Box::new(ARIMA::new(1, 1, 1)))),
    ("AutoETS", Box::new(|| Box::new(AutoETS::new()))),
];

let config = ComparisonConfig::new().with_horizon(12);
let results = compare_models(&factories, &ts, &config).unwrap();
```

## ModelSpec and ModelRegistry

```rust
use anofox_forecast::models::{ModelSpec, ModelRegistry};

// ModelSpec.name is String (accepts &str or String via Into<String>)
let spec = ModelSpec::new("Naive", || Box::new(Naive::new()), true);

// Dynamic names for multiple variants of the same model
let spec = ModelSpec::with_type("MFLES_additive", "MFLES", || ..., false);
let spec = ModelSpec::with_type("MFLES_multiplicative", "MFLES", || ..., false);

// with_period for seasonal models
let spec = ModelSpec::with_period("SeasonalNaive", |p| Box::new(SeasonalNaive::new(p)), 12, true);

// Registry queries
let mut reg = ModelRegistry::new();
reg.register(spec);
reg.names();              // Vec<&str>
reg.by_type("MFLES");    // Vec<&ModelSpec> — all specs with model_type == "MFLES"
reg.contains("Naive");    // bool
reg.remove("Naive");      // bool
```

