# API Reference

This document provides a comprehensive reference for all public APIs in the `anofox-forecast` crate. The crate provides 35+ forecasting models, 76+ time series features, and extensive utilities for time series analysis.

## Table of Contents

- [Core Types](#core-types)
  - [TimeSeries](#timeseries)
  - [TimeSeriesBuilder](#timeseriesbuilder)
  - [Forecast](#forecast)
  - [ForecastError](#forecasterror)
- [Forecaster Trait](#forecaster-trait)
- [Baseline Models](#baseline-models)
  - [Naive](#naive)
  - [SeasonalNaive](#seasonalnaive)
  - [RandomWalkWithDrift](#randomwalkwithdrift)
  - [HistoricAverage](#historicaverage)
  - [WindowAverage](#windowaverage)
  - [SeasonalWindowAverage](#seasonalwindowaverage)
- [Exponential Smoothing](#exponential-smoothing)
  - [SimpleExponentialSmoothing](#simpleexponentialsmoothing)
  - [Holt](#holt)
  - [HoltWinters](#holtwinters)
  - [ETS](#ets)
  - [AutoETS](#autoets)
  - [SeasonalES](#seasonales)
- [ARIMA Models](#arima-models)
  - [ARIMA](#arima)
  - [SARIMA](#sarima)
  - [AutoARIMA](#autoarima)
- [Theta Models](#theta-models)
  - [Theta](#theta)
  - [OptimizedTheta](#optimizedtheta)
  - [DynamicTheta](#dynamictheta)
  - [DynamicOptimizedTheta](#dynamicoptimizedtheta)
  - [AutoTheta](#autotheta)
- [Intermittent Demand Models](#intermittent-demand-models)
  - [Croston](#croston)
  - [TSB](#tsb)
  - [ADIDA](#adida)
  - [IMAPA](#imapa)
- [Advanced Models](#advanced-models)
  - [MFLES](#mfles)
  - [MSTLForecaster](#mstlforecaster)
  - [TBATS](#tbats)
  - [AutoTBATS](#autotbats)
  - [GARCH](#garch)
- [Ensemble](#ensemble)
- [Decomposition](#decomposition)
  - [STL](#stl)
  - [MSTL](#mstl)
- [Periodicity Detection](#periodicity-detection)
  - [PeriodicityDetector Trait](#periodicitydetector-trait)
  - [ACFPeriodicityDetector](#acfperiodicitydetector)
  - [FFTPeriodicityDetector](#fftperiodicitydetector)
  - [Autoperiod](#autoperiod)
  - [CFDAutoperiod](#cfdautoperiod)
  - [SAZED](#sazed)
  - [FFT Utilities](#fft-utilities)
- [Feature Extraction](#feature-extraction)
- [Transformations](#transformations)
- [Validation](#validation)
- [Changepoint Detection](#changepoint-detection)
- [Utilities](#utilities)

---

## Core Types

### TimeSeries

Container for time series data with timestamps and multivariate values.

```rust
pub struct TimeSeries {
    // Fields are private, use methods to access
}
```

#### Constructor Methods

| Method | Description |
|--------|-------------|
| `TimeSeries::new(timestamps, values, labels)` | Create with full configuration |
| `TimeSeries::univariate(values)` | Create simple single-dimension series |

#### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `len()` | `usize` | Number of observations |
| `dimensions()` | `usize` | Number of dimensions |
| `is_empty()` | `bool` | Check if series is empty |
| `is_multivariate()` | `bool` | Check if multi-dimensional |
| `timestamps()` | `&[DateTime<Utc>]` | Get timestamps |
| `values(dimension)` | `Result<&[f64]>` | Get values for dimension |
| `primary_values()` | `&[f64]` | Get first dimension values |
| `slice(start, end)` | `Result<TimeSeries>` | Extract subsequence |
| `has_missing_values()` | `bool` | Check for NaN/Inf |
| `interpolated(fill_edges)` | `TimeSeries` | Fill missing values |

[Back to top](#api-reference)

---

### TimeSeriesBuilder

Builder pattern for constructing TimeSeries.

```rust
let ts = TimeSeriesBuilder::new()
    .timestamps(timestamps)
    .values(values)
    .frequency(Duration::days(1))
    .build()?;
```

| Method | Description |
|--------|-------------|
| `new()` | Create new builder |
| `timestamps(Vec<DateTime<Utc>>)` | Set timestamps |
| `values(Vec<f64>)` | Set univariate values |
| `multivariate_values(Vec<Vec<f64>>, ValueLayout)` | Set multivariate values |
| `labels(Vec<String>)` | Set dimension labels |
| `frequency(Duration)` | Set time frequency |
| `build()` | Build the TimeSeries |

[Back to top](#api-reference)

---

### Forecast

Prediction output containing point forecasts and optional confidence intervals.

```rust
pub struct Forecast {
    // Fields are private, use methods to access
}
```

#### Constructor Methods

| Method | Description |
|--------|-------------|
| `Forecast::new()` | Create empty forecast |
| `Forecast::from_values(values)` | Create from point forecasts |
| `Forecast::from_values_with_intervals(values, lower, upper)` | Create with intervals |

#### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `horizon()` | `usize` | Number of forecast steps |
| `primary()` | `&[f64]` | Get point forecasts |
| `lower()` | `Option<&[Vec<f64>]>` | Get lower bounds |
| `upper()` | `Option<&[Vec<f64>]>` | Get upper bounds |
| `has_lower()` | `bool` | Check if lower bounds exist |
| `has_upper()` | `bool` | Check if upper bounds exist |

[Back to top](#api-reference)

---

### ForecastError

Error types for forecasting operations.

```rust
pub enum ForecastError {
    EmptyData,
    InsufficientData { needed: usize, got: usize },
    InvalidParameter(String),
    DimensionMismatch { expected: usize, got: usize },
    FitRequired,
    MissingValues,
    ComputationError(String),
    // ... other variants
}
```

| Variant | Description |
|---------|-------------|
| `EmptyData` | Input data is empty |
| `InsufficientData` | Not enough data points |
| `InvalidParameter` | Invalid parameter value |
| `DimensionMismatch` | Dimension mismatch in data |
| `FitRequired` | Model not fitted before prediction |
| `MissingValues` | Missing values detected |
| `ComputationError` | Numerical computation error |

[Back to top](#api-reference)

---

## Forecaster Trait

Main trait interface that all forecasting models implement.

```rust
pub trait Forecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()>;
    fn predict(&self, horizon: usize) -> Result<Forecast>;
    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast>;
    fn fitted_values(&self) -> Option<&[f64]>;
    fn residuals(&self) -> Option<&[f64]>;
    fn name(&self) -> &str;
    fn is_fitted(&self) -> bool;
}
```

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `fit` | `series: &TimeSeries` | `Result<()>` | Fit model to data |
| `predict` | `horizon: usize` | `Result<Forecast>` | Generate point forecasts |
| `predict_with_intervals` | `horizon: usize, level: f64` | `Result<Forecast>` | Forecasts with confidence intervals |
| `fitted_values` | - | `Option<&[f64]>` | Get in-sample fitted values |
| `residuals` | - | `Option<&[f64]>` | Get residuals (actual - fitted) |
| `name` | - | `&str` | Model name |
| `is_fitted` | - | `bool` | Check if model is fitted |

[Back to top](#api-reference)

---

## Baseline Models

### Naive

Repeats the last observed value for all forecast horizons.

```rust
pub struct Naive;

impl Naive {
    pub fn new() -> Self;
}
```

**Example:**
```rust
let mut model = Naive::new();
model.fit(&ts)?;
let forecast = model.predict(12)?;
```

[Back to top](#api-reference)

---

### SeasonalNaive

Repeats values from the same season in the previous cycle.

```rust
pub struct SeasonalNaive {
    period: usize,
}

impl SeasonalNaive {
    pub fn new(period: usize) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `period` | `usize` | Seasonal period (e.g., 12 for monthly data) |

[Back to top](#api-reference)

---

### RandomWalkWithDrift

Random walk model with trend drift component.

```rust
pub struct RandomWalkWithDrift;

impl RandomWalkWithDrift {
    pub fn new() -> Self;
}
```

The drift is estimated as the average change between consecutive observations.

[Back to top](#api-reference)

---

### HistoricAverage

Forecasts the mean of all historical observations.

```rust
pub struct HistoricAverage;

impl HistoricAverage {
    pub fn new() -> Self;
}
```

[Back to top](#api-reference)

---

### WindowAverage

Moving window average forecaster.

```rust
pub struct WindowAverage {
    window: usize,
}

impl WindowAverage {
    pub fn new(window: usize) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `window` | `usize` | Number of observations to average |

[Back to top](#api-reference)

---

### SeasonalWindowAverage

Seasonal window-based averaging.

```rust
pub struct SeasonalWindowAverage {
    period: usize,
    window: usize,
}

impl SeasonalWindowAverage {
    pub fn new(period: usize, window: usize) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `period` | `usize` | Seasonal period |
| `window` | `usize` | Number of seasonal cycles to average |

[Back to top](#api-reference)

---

## Exponential Smoothing

### SimpleExponentialSmoothing

Simple exponential smoothing for non-seasonal, non-trending data.

```rust
pub struct SimpleExponentialSmoothing {
    alpha: Option<f64>,
}

impl SimpleExponentialSmoothing {
    pub fn new(alpha: f64) -> Self;
    pub fn auto() -> Self;
    pub fn alpha(&self) -> Option<f64>;
    pub fn level(&self) -> Option<f64>;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | `f64` | Smoothing parameter (0 < alpha < 1) |

| Method | Returns | Description |
|--------|---------|-------------|
| `auto()` | `Self` | Create with auto-optimized alpha |
| `alpha()` | `Option<f64>` | Get fitted alpha value |
| `level()` | `Option<f64>` | Get final level |

[Back to top](#api-reference)

---

### Holt

Holt's linear trend method (double exponential smoothing).

```rust
pub struct Holt {
    alpha: Option<f64>,
    beta: Option<f64>,
}

impl Holt {
    pub fn new(alpha: f64, beta: f64) -> Self;
    pub fn auto() -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | `f64` | Level smoothing (0 < alpha < 1) |
| `beta` | `f64` | Trend smoothing (0 < beta < 1) |

[Back to top](#api-reference)

---

### HoltWinters

Holt-Winters seasonal exponential smoothing.

```rust
pub struct HoltWinters {
    period: usize,
    seasonal_type: SeasonalType,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
}

impl HoltWinters {
    pub fn new(period: usize, seasonal_type: SeasonalType) -> Self;
    pub fn with_params(period: usize, seasonal_type: SeasonalType,
                       alpha: f64, beta: f64, gamma: f64) -> Self;
    pub fn auto(period: usize, seasonal_type: SeasonalType) -> Self;
}

pub enum SeasonalType {
    Additive,
    Multiplicative,
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `period` | `usize` | Seasonal period |
| `seasonal_type` | `SeasonalType` | Additive or Multiplicative |
| `alpha` | `f64` | Level smoothing |
| `beta` | `f64` | Trend smoothing |
| `gamma` | `f64` | Seasonal smoothing |

[Back to top](#api-reference)

---

### ETS

Error-Trend-Seasonal state-space model supporting 30 model combinations.

```rust
pub struct ETS {
    spec: ETSSpec,
}

impl ETS {
    pub fn new(spec: ETSSpec) -> Self;
    pub fn auto() -> Self;
}

pub struct ETSSpec {
    pub error: ErrorType,
    pub trend: TrendType,
    pub seasonal: SeasonalTypeETS,
    pub period: Option<usize>,
    pub damped: bool,
}

pub enum ErrorType { Additive, Multiplicative }
pub enum TrendType { None, Additive, Multiplicative }
pub enum SeasonalTypeETS { None, Additive, Multiplicative }
```

**Predefined Specs:**

| Method | Model | Description |
|--------|-------|-------------|
| `ETSSpec::ann()` | ETS(A,N,N) | Simple exponential smoothing |
| `ETSSpec::aan()` | ETS(A,A,N) | Holt's linear trend |
| `ETSSpec::aaa(period)` | ETS(A,A,A) | Additive Holt-Winters |
| `ETSSpec::mam(period)` | ETS(M,A,M) | Multiplicative Holt-Winters |

[Back to top](#api-reference)

---

### AutoETS

Automatic ETS model selection using information criteria.

```rust
pub struct AutoETS {
    config: AutoETSConfig,
}

impl AutoETS {
    pub fn new() -> Self;
    pub fn with_config(config: AutoETSConfig) -> Self;
    pub fn with_period(period: usize) -> Self;
}

pub struct AutoETSConfig {
    pub criterion: SelectionCriterion,
    pub seasonal_period: Option<usize>,
    pub allow_multiplicative: bool,
}

pub enum SelectionCriterion { AIC, BIC, AICc }
```

[Back to top](#api-reference)

---

### SeasonalES

Multiplicative seasonal exponential smoothing.

```rust
pub struct SeasonalES {
    period: usize,
}

impl SeasonalES {
    pub fn new(period: usize) -> Self;
}
```

[Back to top](#api-reference)

---

## ARIMA Models

### ARIMA

Non-seasonal ARIMA(p,d,q) model.

```rust
pub struct ARIMA {
    spec: ARIMASpec,
}

impl ARIMA {
    pub fn new(p: usize, d: usize, q: usize) -> Self;
    pub fn spec(&self) -> &ARIMASpec;
    pub fn ar_coefficients(&self) -> &[f64];
    pub fn ma_coefficients(&self) -> &[f64];
    pub fn intercept(&self) -> f64;
    pub fn aic(&self) -> Option<f64>;
    pub fn bic(&self) -> Option<f64>;
}

pub struct ARIMASpec {
    pub p: usize,  // AR order
    pub d: usize,  // Differencing order
    pub q: usize,  // MA order
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | `usize` | Autoregressive order |
| `d` | `usize` | Differencing order |
| `q` | `usize` | Moving average order |

[Back to top](#api-reference)

---

### SARIMA

Seasonal ARIMA(p,d,q)(P,D,Q)\[s\] model.

```rust
pub struct SARIMA {
    spec: SARIMASpec,
}

impl SARIMA {
    pub fn new(p: usize, d: usize, q: usize,
               cap_p: usize, cap_d: usize, cap_q: usize, s: usize) -> Self;
}

pub struct SARIMASpec {
    pub p: usize, pub d: usize, pub q: usize,       // Non-seasonal
    pub cap_p: usize, pub cap_d: usize, pub cap_q: usize,  // Seasonal
    pub s: usize,  // Seasonal period
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `p, d, q` | `usize` | Non-seasonal orders |
| `P, D, Q` | `usize` | Seasonal orders |
| `s` | `usize` | Seasonal period |

[Back to top](#api-reference)

---

### AutoARIMA

Automatic ARIMA order selection.

```rust
pub struct AutoARIMA {
    config: AutoARIMAConfig,
}

impl AutoARIMA {
    pub fn new() -> Self;
    pub fn with_config(config: AutoARIMAConfig) -> Self;
}

pub struct AutoARIMAConfig {
    pub max_p: usize,
    pub max_d: usize,
    pub max_q: usize,
    pub seasonal_period: Option<usize>,
    pub criterion: SelectionCriterion,
    pub stepwise: bool,
    pub true_stepwise: bool,  // Neighbor-based hill climbing
}

impl AutoARIMAConfig {
    pub fn with_true_stepwise(self, enabled: bool) -> Self;
    pub fn exhaustive(self) -> Self;
}
```

| Parameter | Description |
|-----------|-------------|
| `stepwise` | Use stepwise search (faster, fewer models) |
| `true_stepwise` | Use neighbor-based hill climbing (60-70% fewer evaluations) |

**Parallel Execution:**

Enable with `--features parallel` for 4-8x speedup on multi-core systems:
```toml
[dependencies]
anofox-forecast = { version = "0.3", features = ["parallel"] }
```

[Back to top](#api-reference)

---

## Theta Models

### Theta

Standard Theta Model (STM) for forecasting.

```rust
pub struct Theta {
    theta: f64,
    seasonal_period: usize,
    decomposition: DecompositionType,
}

impl Theta {
    pub fn new() -> Self;
    pub fn with_theta(theta: f64) -> Self;
    pub fn seasonal(period: usize) -> Self;
    pub fn seasonal_with_type(period: usize, decomposition: DecompositionType) -> Self;
}

pub enum DecompositionType {
    Additive,
    Multiplicative,
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `theta` | `f64` | Theta parameter (default: 2.0) |
| `period` | `usize` | Seasonal period (0 for non-seasonal) |
| `decomposition` | `DecompositionType` | Seasonal decomposition type |

[Back to top](#api-reference)

---

### OptimizedTheta

Theta model with optimized alpha and theta parameters.

```rust
pub struct OptimizedTheta;

impl OptimizedTheta {
    pub fn new() -> Self;
}
```

Parameters are optimized using Nelder-Mead minimization of MSE.

[Back to top](#api-reference)

---

### DynamicTheta

Theta with dynamic linear coefficient updates.

```rust
pub struct DynamicTheta {
    period: Option<usize>,
}

impl DynamicTheta {
    pub fn new(period: Option<usize>) -> Self;
}
```

[Back to top](#api-reference)

---

### DynamicOptimizedTheta

Combines dynamic updates with parameter optimization.

```rust
pub struct DynamicOptimizedTheta {
    period: Option<usize>,
}

impl DynamicOptimizedTheta {
    pub fn new(period: Option<usize>) -> Self;
}
```

[Back to top](#api-reference)

---

### AutoTheta

Automatic Theta model selection.

```rust
pub struct AutoTheta;

impl AutoTheta {
    pub fn new() -> Self;
}
```

Selects the best Theta variant based on cross-validation performance.

[Back to top](#api-reference)

---

## Intermittent Demand Models

### Croston

Croston's method for intermittent demand forecasting.

```rust
pub struct Croston {
    variant: CrostonVariant,
    alpha: f64,
}

impl Croston {
    pub fn new() -> Self;           // Classic variant
    pub fn classic() -> Self;
    pub fn sba() -> Self;           // Syntetos-Babai adjusted
    pub fn with_alpha(alpha: f64) -> Self;
}

pub enum CrostonVariant {
    Classic,
    SBA,  // Syntetos-Babai Approximation
}
```

| Variant | Description |
|---------|-------------|
| `Classic` | Original Croston method |
| `SBA` | Bias-corrected Syntetos-Babai variant |

[Back to top](#api-reference)

---

### TSB

Teunter-Syntetos-Babai method for intermittent demand.

```rust
pub struct TSB {
    alpha: f64,
    beta: f64,
}

impl TSB {
    pub fn new() -> Self;
    pub fn with_params(alpha: f64, beta: f64) -> Self;
}
```

[Back to top](#api-reference)

---

### ADIDA

Aggregate-Disaggregate Intermittent Demand Approach.

```rust
pub struct ADIDA {
    aggregation_level: usize,
}

impl ADIDA {
    pub fn new() -> Self;
    pub fn with_aggregation(level: usize) -> Self;
}
```

[Back to top](#api-reference)

---

### IMAPA

Intermittent Multiple Aggregation Prediction Algorithm.

```rust
pub struct IMAPA;

impl IMAPA {
    pub fn new() -> Self;
}
```

Combines forecasts from multiple aggregation levels.

[Back to top](#api-reference)

---

## Advanced Models

### MFLES

Multiple Fourier Linear Exponential Smoothing - gradient boosted decomposition.

```rust
pub struct MFLES {
    seasonal_periods: Vec<usize>,
}

impl MFLES {
    pub fn new(seasonal_periods: Vec<usize>) -> Self;
    pub fn with_max_rounds(rounds: usize) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `seasonal_periods` | `Vec<usize>` | Seasonal periods to model |
| `max_rounds` | `usize` | Maximum boosting iterations |

[Back to top](#api-reference)

---

### MSTLForecaster

MSTL decomposition-based forecaster for multiple seasonalities.

```rust
pub struct MSTLForecaster {
    seasonal_periods: Vec<usize>,
}

impl MSTLForecaster {
    pub fn new(seasonal_periods: Vec<usize>) -> Self;
}
```

[Back to top](#api-reference)

---

### TBATS

Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, Seasonal components.

```rust
pub struct TBATS {
    seasonal_periods: Vec<usize>,
}

impl TBATS {
    pub fn new(seasonal_periods: Vec<usize>) -> Self;
}
```

Handles complex seasonal patterns with trigonometric representation.

[Back to top](#api-reference)

---

### AutoTBATS

Automatic TBATS configuration selection.

```rust
pub struct AutoTBATS;

impl AutoTBATS {
    pub fn new(seasonal_periods: Vec<usize>) -> Self;
}
```

[Back to top](#api-reference)

---

### GARCH

Generalized Autoregressive Conditional Heteroskedasticity for volatility modeling.

```rust
pub struct GARCH {
    p: usize,
    q: usize,
}

impl GARCH {
    pub fn new(p: usize, q: usize) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | `usize` | GARCH order (variance lags) |
| `q` | `usize` | ARCH order (squared residual lags) |

[Back to top](#api-reference)

---

## Ensemble

Combines multiple forecasting models.

```rust
pub struct Ensemble {
    models: Vec<Box<dyn Forecaster>>,
    method: CombinationMethod,
}

impl Ensemble {
    pub fn new(models: Vec<Box<dyn Forecaster>>) -> Self;
    pub fn with_method(method: CombinationMethod) -> Self;
    pub fn with_weights(weights: Vec<f64>) -> Self;
}

pub enum CombinationMethod {
    Mean,
    Median,
    WeightedMSE,
    Custom,
}
```

| Method | Description |
|--------|-------------|
| `Mean` | Simple average of forecasts |
| `Median` | Median of forecasts |
| `WeightedMSE` | Inverse MSE weighting |
| `Custom` | User-provided weights |

[Back to top](#api-reference)

---

## Decomposition

### STL

Seasonal-Trend decomposition using LOESS.

```rust
pub struct STL {
    period: usize,
}

impl STL {
    pub fn new(period: usize) -> Self;
    pub fn decompose(&self, series: &[f64]) -> Result<STLResult>;
}

pub struct STLResult {
    pub seasonal: Vec<f64>,
    pub trend: Vec<f64>,
    pub remainder: Vec<f64>,
}
```

[Back to top](#api-reference)

---

### MSTL

Multiple Seasonal-Trend decomposition using LOESS.

```rust
pub struct MSTL {
    periods: Vec<usize>,
}

impl MSTL {
    pub fn new(periods: Vec<usize>) -> Self;
    pub fn decompose(&self, series: &[f64]) -> Result<MSTLResult>;
}

pub struct MSTLResult {
    pub seasonal_components: Vec<Vec<f64>>,
    pub trend: Vec<f64>,
    pub remainder: Vec<f64>,
}
```

[Back to top](#api-reference)

---

## Periodicity Detection

Automatic detection of periodic patterns in time series data using multiple algorithms.

### PeriodicityDetector Trait

Common interface for all periodicity detection algorithms.

```rust
pub trait PeriodicityDetector {
    fn detect(&self, series: &[f64]) -> PeriodicityResult;
    fn name(&self) -> &'static str;
}
```

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `detect` | `series: &[f64]` | `PeriodicityResult` | Detect periods in the series |
| `name` | - | `&'static str` | Algorithm name |

### Result Types

```rust
pub struct PeriodicityResult {
    pub primary_period: Option<usize>,
    pub periods: Vec<DetectedPeriod>,
    pub method: String,
}

pub struct DetectedPeriod {
    pub period: usize,
    pub score: f64,
    pub source: PeriodSource,
}

pub enum PeriodSource {
    Frequency,  // FFT/periodogram
    Time,       // ACF
    Hybrid,     // Validated by both domains
    Ensemble,   // Consensus from multiple methods
}
```

| Method | Returns | Description |
|--------|---------|-------------|
| `confidence()` | `f64` | Confidence score (0.0-1.0) for primary period |

### Convenience Functions

```rust
/// Detect period using Autoperiod (recommended default)
pub fn detect_period(series: &[f64]) -> PeriodicityResult;

/// Detect period with custom range
pub fn detect_period_range(series: &[f64], min_period: usize, max_period: usize) -> PeriodicityResult;

/// Detect period using SAZED ensemble
pub fn detect_period_ensemble(series: &[f64]) -> PeriodicityResult;
```

[Back to top](#api-reference)

---

### ACFPeriodicityDetector

Time-domain detector using autocorrelation function peaks.

```rust
pub struct ACFPeriodicityDetector {
    min_period: usize,
    max_period: usize,
    correlation_threshold: f64,
}

impl ACFPeriodicityDetector {
    pub fn new(min_period: usize, max_period: usize, correlation_threshold: f64) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_period` | `usize` | Minimum period to consider |
| `max_period` | `usize` | Maximum period to consider |
| `correlation_threshold` | `f64` | Minimum ACF value for peaks (e.g., 0.3) |

**Example:**
```rust
let detector = ACFPeriodicityDetector::new(2, 365, 0.3);
let result = detector.detect(&values);
```

[Back to top](#api-reference)

---

### FFTPeriodicityDetector

Frequency-domain detector using periodogram peaks.

```rust
pub struct FFTPeriodicityDetector {
    min_period: usize,
    max_period: usize,
    power_threshold: f64,
}

impl FFTPeriodicityDetector {
    pub fn new(min_period: usize, max_period: usize, power_threshold: f64) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_period` | `usize` | Minimum period to consider |
| `max_period` | `usize` | Maximum period to consider |
| `power_threshold` | `f64` | Multiplier for noise floor threshold (e.g., 3.0) |

**Example:**
```rust
let detector = FFTPeriodicityDetector::new(2, 365, 3.0);
let result = detector.detect(&values);
```

[Back to top](#api-reference)

---

### Autoperiod

Hybrid FFT+ACF detector (Vlachos et al. 2005). Recommended default method.

```rust
pub struct Autoperiod {
    min_period: usize,
    max_period: usize,
    power_threshold: f64,
    acf_threshold: f64,
}

impl Autoperiod {
    pub fn new(min_period: usize, max_period: usize,
               power_threshold: f64, acf_threshold: f64) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_period` | `usize` | Minimum period to consider |
| `max_period` | `usize` | Maximum period to consider |
| `power_threshold` | `f64` | FFT power threshold (e.g., 3.0) |
| `acf_threshold` | `f64` | ACF validation threshold (e.g., 0.2) |

**Algorithm:**
1. **GetPeriodHints**: Find peaks in periodogram above threshold
2. **ACFFiltering**: Validate hints lie on ACF local maxima
3. **Gradient ascent**: Refine period using ACF slope

**Example:**
```rust
let detector = Autoperiod::new(2, 365, 3.0, 0.2);
let result = detector.detect(&values);
```

[Back to top](#api-reference)

---

### CFDAutoperiod

Cluster-Filter-Detect Autoperiod (Puech et al. 2020). Noise-resistant with detrending and clustering.

```rust
pub struct CFDAutoperiod {
    min_period: usize,
    max_period: usize,
    cluster_tolerance: f64,
}

impl CFDAutoperiod {
    pub fn new(min_period: usize, max_period: usize, cluster_tolerance: f64) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_period` | `usize` | Minimum period to consider |
| `max_period` | `usize` | Maximum period to consider |
| `cluster_tolerance` | `f64` | Clustering tolerance for period grouping (e.g., 2.0) |

**Algorithm:**
1. **Detrend**: Remove trend using differencing
2. **FFT hints**: Get period candidates from periodogram
3. **Cluster**: Group similar periods using density-based clustering
4. **ACF validation**: Filter with autocorrelation

**Example:**
```rust
let detector = CFDAutoperiod::new(2, 365, 2.0);
let result = detector.detect(&values);
```

[Back to top](#api-reference)

---

### SAZED

Spectral + Autocorrelation + Zero-crossing + Ensemble + Density detector (Toller et al. 2019). Parameter-free ensemble method.

```rust
pub struct SAZED {
    min_period: usize,
    max_period: usize,
    vote_tolerance: usize,
}

impl SAZED {
    pub fn new(min_period: usize, max_period: usize) -> Self;
    pub fn with_tolerance(min_period: usize, max_period: usize, tolerance: usize) -> Self;
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_period` | `usize` | Minimum period to consider |
| `max_period` | `usize` | Maximum period to consider |
| `vote_tolerance` | `usize` | Period tolerance for voting (default: 1) |

**Components:**
1. **Spectral**: FFT-based period from periodogram
2. **ACF peaks**: Strongest autocorrelation peak
3. **ACF average**: Mean of all significant ACF peaks
4. **Zero-distance**: Period from ACF zero-crossing patterns
5. **Voting**: Consensus from all methods

**Example:**
```rust
let detector = SAZED::new(2, 365);
let result = detector.detect(&values);

// Results include periods from multiple sources
for p in &result.periods {
    println!("{}: period={}, score={:.3}", p.source, p.period, p.score);
}
```

[Back to top](#api-reference)

---

### FFT Utilities

Low-level FFT functions for custom periodicity analysis.

```rust
/// Compute FFT of real-valued signal (returns positive frequencies only)
pub fn fft_real(signal: &[f64]) -> Vec<Complex64>;

/// Compute periodogram (power spectral density)
/// Returns (period, power) pairs sorted by period
pub fn periodogram(signal: &[f64]) -> Vec<(usize, f64)>;

/// Find significant peaks in periodogram
pub fn periodogram_peaks(
    signal: &[f64],
    threshold: f64,      // Multiplier for noise floor
    min_period: usize,
    max_period: usize,
) -> Vec<(usize, f64)>;

/// Welch's periodogram for reduced variance
pub fn welch_periodogram(
    signal: &[f64],
    window_size: usize,  // Segment size (power of 2 recommended)
    overlap: f64,        // Overlap ratio (0.0-0.9, typically 0.5)
) -> Vec<(usize, f64)>;

/// Convert period to frequency index
pub fn period_to_frequency_index(period: usize, n: usize) -> usize;

/// Convert frequency index to period
pub fn frequency_index_to_period(freq_index: usize, n: usize) -> usize;
```

**Example:**
```rust
use anofox_forecast::detection::fft::{periodogram, periodogram_peaks};

// Get full periodogram
let psd = periodogram(&values);

// Find significant peaks (3x above noise floor)
let peaks = periodogram_peaks(&values, 3.0, 2, 100);
for (period, power) in peaks {
    println!("Period: {}, Power: {:.4}", period, power);
}
```

[Back to top](#api-reference)

---

## Feature Extraction

The crate provides 76+ time series features organized by category.

### Basic Features

```rust
pub fn mean(series: &[f64]) -> f64;
pub fn median(series: &[f64]) -> f64;
pub fn variance(series: &[f64]) -> f64;
pub fn standard_deviation(series: &[f64]) -> f64;
pub fn minimum(series: &[f64]) -> f64;
pub fn maximum(series: &[f64]) -> f64;
pub fn sum_values(series: &[f64]) -> f64;
pub fn abs_energy(series: &[f64]) -> f64;
pub fn mean_abs_change(series: &[f64]) -> f64;
pub fn mean_change(series: &[f64]) -> f64;
```

### Distribution Features

```rust
pub fn skewness(series: &[f64]) -> f64;
pub fn kurtosis(series: &[f64]) -> f64;
pub fn quantile(series: &[f64], q: f64) -> f64;
pub fn variation_coefficient(series: &[f64]) -> f64;
```

### Autocorrelation Features

```rust
pub fn autocorrelation(series: &[f64], lag: usize) -> f64;
pub fn partial_autocorrelation(series: &[f64], lag: usize) -> f64;
```

### Entropy Features

```rust
pub fn approximate_entropy(series: &[f64], m: usize, r: f64) -> f64;
pub fn sample_entropy(series: &[f64], m: usize, r: f64) -> f64;
pub fn permutation_entropy(series: &[f64], order: usize) -> f64;
```

### Complexity Features

```rust
pub fn cid_ce(series: &[f64], normalize: bool) -> f64;
pub fn lempel_ziv_complexity(series: &[f64], threshold: Option<f64>) -> f64;
```

### Trend Features

```rust
pub fn linear_trend(series: &[f64]) -> LinearTrendResult;
pub fn ar_coefficient(series: &[f64], k: usize) -> f64;
```

[Back to top](#api-reference)

---

## Transformations

### Box-Cox

```rust
pub fn boxcox(series: &[f64], lambda: f64) -> Result<BoxCoxResult>;
pub fn boxcox_auto(series: &[f64]) -> Result<BoxCoxResult>;
pub fn inv_boxcox(transformed: &[f64], lambda: f64) -> Result<Vec<f64>>;
pub fn boxcox_lambda(series: &[f64]) -> Result<f64>;

pub struct BoxCoxResult {
    pub transformed: Vec<f64>,
    pub lambda: f64,
}
```

### Scaling

```rust
pub fn standardize(series: &[f64]) -> ScaleResult;
pub fn normalize(series: &[f64]) -> ScaleResult;
pub fn robust_scale(series: &[f64]) -> ScaleResult;

pub struct ScaleResult {
    pub scaled: Vec<f64>,
    pub mean: f64,
    pub std: f64,
}
```

### Window Functions

```rust
pub fn rolling_mean(series: &[f64], window: usize, center: bool) -> Vec<f64>;
pub fn rolling_std(series: &[f64], window: usize, center: bool) -> Vec<f64>;
pub fn rolling_min(series: &[f64], window: usize, center: bool) -> Vec<f64>;
pub fn rolling_max(series: &[f64], window: usize, center: bool) -> Vec<f64>;
pub fn expanding_mean(series: &[f64]) -> Vec<f64>;
pub fn ewm_mean(series: &[f64], span: f64) -> Vec<f64>;
```

[Back to top](#api-reference)

---

## Validation

### Residual Tests

```rust
pub fn ljung_box(residuals: &[f64], nlags: Option<usize>, seasonal: usize) -> LjungBoxResult;
pub fn box_pierce(residuals: &[f64], nlags: Option<usize>, seasonal: usize) -> LjungBoxResult;
pub fn durbin_watson(residuals: &[f64]) -> DurbinWatsonResult;

pub struct LjungBoxResult {
    pub statistic: f64,
    pub p_value: f64,
    pub degrees_of_freedom: usize,
}

pub struct DurbinWatsonResult {
    pub statistic: f64,  // Values near 2 indicate no autocorrelation
}
```

### Stationarity Tests

```rust
pub fn adf_test(series: &[f64], nlags: Option<usize>) -> StationarityResult;
pub fn kpss_test(series: &[f64], nlags: Option<usize>) -> StationarityResult;

pub struct StationarityResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_stationary: bool,
}
```

[Back to top](#api-reference)

---

## Changepoint Detection

PELT (Pruned Exact Linear Time) algorithm for changepoint detection.

```rust
pub fn pelt_detect(series: &[f64], config: &PeltConfig) -> PeltResult;

pub struct PeltConfig {
    pub penalty: f64,
    pub min_segment_length: usize,
    pub cost_fn: CostFunction,
}

impl PeltConfig {
    pub fn default() -> Self;
    pub fn penalty(penalty: f64) -> Self;
    pub fn with_bic_penalty(n: usize) -> Self;
}

pub enum CostFunction {
    L2,      // Squared cost
    L1,      // Absolute cost
    Normal,  // Normal likelihood
    Poisson, // Poisson likelihood
}

pub struct PeltResult {
    pub changepoints: Vec<usize>,
    pub n_changepoints: usize,
}
```

[Back to top](#api-reference)

---

## Utilities

### Accuracy Metrics

```rust
pub fn calculate_metrics(actual: &[f64], predicted: &[f64],
                         seasonal_period: Option<usize>) -> Result<AccuracyMetrics>;

pub struct AccuracyMetrics {
    pub mae: f64,      // Mean Absolute Error
    pub mse: f64,      // Mean Squared Error
    pub rmse: f64,     // Root Mean Squared Error
    pub mape: f64,     // Mean Absolute Percentage Error
    pub smape: f64,    // Symmetric MAPE
    pub mase: f64,     // Mean Absolute Scaled Error
    pub r_squared: f64,
}
```

### Cross-Validation

```rust
pub fn cross_validate<F>(config: &CVConfig, series: &TimeSeries,
                         model_factory: F) -> Result<CVResults>;

pub struct CVConfig {
    pub horizon: usize,
    pub initial_window: usize,
    pub step_size: usize,
    pub strategy: CVStrategy,
}

pub enum CVStrategy {
    Rolling,    // Fixed window slides forward
    Expanding,  // Window grows over time
}

pub struct CVResults {
    pub n_folds: usize,
    pub aggregated: AggregatedMetrics,
    pub fold_metrics: Vec<AccuracyMetrics>,
}
```

### Optimization

```rust
pub fn nelder_mead(objective: fn(&[f64]) -> f64, initial: Vec<f64>,
                   config: &NelderMeadConfig) -> Result<NelderMeadResult>;

pub struct NelderMeadConfig {
    pub max_iter: usize,
    pub tolerance: f64,
}

pub struct NelderMeadResult {
    pub parameters: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
}
```

[Back to top](#api-reference)

---

### Bootstrap Intervals

Bootstrap methods for empirical confidence intervals.

```rust
pub struct BootstrapConfig {
    pub n_samples: usize,      // Number of bootstrap samples (default: 1000)
    pub block_size: Option<usize>,  // Block size for block bootstrap
    pub seed: Option<u64>,     // Random seed for reproducibility
}

impl BootstrapConfig {
    pub fn new(n_samples: usize) -> Self;
    pub fn with_block_size(self, block_size: usize) -> Self;
    pub fn with_seed(self, seed: u64) -> Self;
}

pub struct BootstrapResult {
    pub lower: Vec<f64>,       // Lower bounds per horizon step
    pub upper: Vec<f64>,       // Upper bounds per horizon step
    pub level: f64,            // Confidence level used
    pub n_samples: usize,      // Number of samples used
}

/// Generate bootstrap confidence intervals
pub fn bootstrap_intervals<M: Forecaster + Clone>(
    model: &M,
    series: &TimeSeries,
    horizon: usize,
    level: f64,
    config: &BootstrapConfig,
) -> Result<BootstrapResult>;

/// Generate forecast with bootstrap intervals
pub fn bootstrap_forecast<M: Forecaster + Clone>(
    model: &M,
    series: &TimeSeries,
    horizon: usize,
    level: f64,
    config: &BootstrapConfig,
) -> Result<Forecast>;
```

| Method | Description |
|--------|-------------|
| Residual Bootstrap | Resamples fitted residuals with replacement |
| Block Bootstrap | Preserves autocorrelation structure |

**Example:**
```rust
use anofox_forecast::utils::bootstrap::{bootstrap_forecast, BootstrapConfig};

let config = BootstrapConfig::new(500).with_seed(42);
let forecast = bootstrap_forecast(&model, &ts, 12, 0.95, &config)?;
```

[Back to top](#api-reference)

---

## Prelude

Convenience re-exports for common usage:

```rust
pub use crate::core::{Forecast, TimeSeries};
pub use crate::error::{ForecastError, Result};
pub use crate::models::Forecaster;
pub use crate::utils::{calculate_metrics, AccuracyMetrics};
```

**Usage:**
```rust
use anofox_forecast::prelude::*;
```

[Back to top](#api-reference)
