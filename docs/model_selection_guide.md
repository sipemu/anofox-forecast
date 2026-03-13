# Model Selection Guide

Practical guidance for choosing the right forecasting model in `anofox-forecast`.

## Quick Reference Table

| Model | Best For | Trend | Seasonality | Speed | Use When... |
|-------|----------|-------|-------------|-------|-------------|
| **Naive** | Flat data | No | No | Instant | Baseline benchmark; data is a random walk |
| **HistoricAverage** | Stationary data | No | No | Instant | Mean-reverting series; another baseline |
| **WindowAverage** | Recent-level data | No | No | Instant | You want a smoothed recent average |
| **RandomWalkWithDrift** | Linear trend | Yes | No | Instant | Steady upward/downward drift |
| **SeasonalNaive** | Stable seasonality | No | Yes | Instant | Repeating pattern with little noise |
| **SeasonalWindowAverage** | Noisy seasonality | No | Yes | Instant | Seasonal pattern that benefits from averaging |
| **SES** | Level data | No | No | Fast | No trend or seasonality; adaptive smoothing |
| **HoltLinearTrend** | Trending data | Yes | No | Fast | Linear trend, no seasonality |
| **HoltLinearTrend (damped)** | Flattening trend | Yes | No | Fast | Trend that decays over the forecast horizon |
| **HoltWinters** | Trend + season | Yes | Yes | Fast | Both trend and seasonality present |
| **SeasonalES** | Per-slot smoothing | No | Yes | Fast | Independent seasonal slots (e.g., day-of-week) |
| **ETS** | General purpose | Configurable | Configurable | Fast | You know the error/trend/seasonal structure |
| **AutoETS** | General purpose | Auto | Auto | Medium | Let the library choose the best ETS spec |
| **ARIMA** | Autocorrelated data | Configurable | No | Medium | Known (p,d,q) order; correlated residuals |
| **SARIMA** | Seasonal + autocorrelation | Configurable | Yes | Medium | Known seasonal ARIMA orders |
| **AutoARIMA** | General purpose | Auto | Auto | Slow | Automatic ARIMA order selection |
| **Theta (STM)** | Competition-grade | Yes | Optional | Fast | M3-winning method; strong baseline |
| **OptimizedTheta (OTM)** | Optimized Theta | Yes | Optional | Medium | When STM defaults are not ideal |
| **DynamicTheta (DSTM/DOTM)** | Changing trends | Yes | Optional | Medium | Trend changes over time |
| **AutoTheta** | General purpose | Auto | Auto | Medium | Let the library pick the best Theta variant |
| **Croston** | Intermittent demand | No | No | Fast | Sparse demand with many zeros |
| **Croston SBA** | Intermittent demand | No | No | Fast | Bias-corrected Croston |
| **TSB** | Obsolescence risk | No | No | Fast | Intermittent demand that may go to zero permanently |
| **ADIDA** | Intermittent demand | No | No | Fast | Aggregate-disaggregate approach to sparse data |
| **IMAPA** | Intermittent demand | No | No | Medium | Most robust intermittent method (multi-aggregation) |
| **TBATS** | Complex seasonality | Yes | Multiple | Slow | Multiple seasonal periods; non-integer seasonality |
| **AutoTBATS** | Complex seasonality | Auto | Multiple | Very Slow | Automatic TBATS configuration selection |
| **MFLES** | Robust decomposition | Yes | Yes | Medium | Gradient-boosted decomposition; robust to outliers |
| **MSTLForecaster** | Multiple seasonality | Yes | Multiple | Medium | MSTL decomposition + trend forecasting |
| **GARCH** | Volatility | No | No | Medium | Financial data; variance changes over time |
| **AutoForecast** | Unknown data | Auto | Auto | Slow | No idea which model to use; compares families |
| **AutoEnsemble** | Best accuracy | Auto | Auto | Very Slow | Maximum accuracy; combines top models |
| **Ensemble** | Custom combination | Varies | Varies | Varies | You want to combine specific models |


## Decision Flowchart

```
Start: What does your data look like?
 |
 +-- Is demand intermittent (many zeros)?
 |    |
 |    +-- YES --> Is demand potentially obsolescent (trending to zero)?
 |    |            |
 |    |            +-- YES --> TSB
 |    |            +-- NO  --> Do you want maximum robustness?
 |    |                         |
 |    |                         +-- YES --> IMAPA
 |    |                         +-- NO  --> Croston SBA
 |    |
 |    +-- NO --> Continue below
 |
 +-- Does the data have seasonality?
 |    |
 |    +-- YES --> Multiple seasonal periods (e.g., hourly + daily)?
 |    |            |
 |    |            +-- YES --> Is the seasonality complex or non-integer?
 |    |            |            |
 |    |            |            +-- YES --> TBATS or AutoTBATS
 |    |            |            +-- NO  --> MSTLForecaster or MFLES
 |    |            |
 |    |            +-- NO (single period) --> Does it have a trend?
 |    |                                       |
 |    |                                       +-- YES --> AutoETS, HoltWinters, or AutoForecast
 |    |                                       +-- NO  --> SeasonalNaive, SeasonalES, or AutoETS
 |    |
 |    +-- NO --> Does the data have a trend?
 |               |
 |               +-- YES --> Is the trend linear and persistent?
 |               |            |
 |               |            +-- YES --> HoltLinearTrend or RandomWalkWithDrift
 |               |            +-- NO (decaying/changing) --> HoltLinearTrend(damped) or AutoETS
 |               |
 |               +-- NO --> Is there autocorrelation in residuals?
 |                           |
 |                           +-- YES --> ARIMA or AutoARIMA
 |                           +-- NO  --> SES or Naive
 |
 +-- Not sure / want best accuracy? --> AutoForecast or AutoEnsemble
```


## Model Families

### 1. Baseline Models

Simple methods that serve as benchmarks. Always compare your chosen model against these.

**Naive** -- Repeats the last observed value. Use as a lower bound for any forecasting task.

**HistoricAverage** -- Predicts the mean of all historical data. Good for stationary, mean-reverting series.

**WindowAverage** -- Predicts the mean of the last N observations. Smooths out recent noise.

**RandomWalkWithDrift** -- Last value plus a constant drift (average historical change). The go-to baseline for trending data.

**SeasonalNaive** -- Repeats the last complete seasonal cycle. The standard baseline for seasonal data.

**SeasonalWindowAverage** -- Averages same-season values across multiple cycles. Smooths noisy seasonal patterns compared to SeasonalNaive.

```rust
use anofox_forecast::models::baseline::{
    Naive, HistoricAverage, WindowAverage, RandomWalkWithDrift,
    SeasonalNaive, SeasonalWindowAverage,
};
use anofox_forecast::models::Forecaster;

// Flat baseline
let mut model = Naive::new();
model.fit(&ts).unwrap();

// Trending baseline
let mut model = RandomWalkWithDrift::new();
model.fit(&ts).unwrap();

// Seasonal baseline (period=12 for monthly data)
let mut model = SeasonalNaive::new(12);
model.fit(&ts).unwrap();

// Smoothed seasonal baseline (average last 3 years)
let mut model = SeasonalWindowAverage::new(12, 3);
model.fit(&ts).unwrap();

let forecast = model.predict(12).unwrap();
```

### 2. Exponential Smoothing (ETS Family)

Adaptive methods that weight recent observations more heavily.

**SES (SimpleExponentialSmoothing)** -- Single smoothing parameter (alpha). Use for data with no trend or seasonality. A fixed alpha gives you control; `SES::optimized()` finds the best alpha automatically.

**HoltLinearTrend** -- Adds a trend component (alpha + beta). Use `.damped()` when the trend should flatten over the forecast horizon -- this is often the safest default for trending data.

**HoltWinters** -- Full triple exponential smoothing (alpha + beta + gamma). Choose additive seasonality when seasonal swings are constant in absolute terms; choose multiplicative when they scale with the level.

**SeasonalES** -- Applies SES independently to each seasonal slot. Different from Holt-Winters: it does not model trend, but captures per-slot dynamics well.

**ETS** -- The full state-space framework with 30 possible model specifications (Error, Trend, Seasonal). Use when you know the structure: `ETSSpec::ann()` for SES, `ETSSpec::aaa()` for additive Holt-Winters, `ETSSpec::mam()` for multiplicative, etc.

**AutoETS** -- Automatically selects the best ETS specification using information criteria (AICc by default). This is the recommended starting point for most forecasting tasks.

```rust
use anofox_forecast::models::exponential::{
    SimpleExponentialSmoothing, HoltLinearTrend, HoltWinters, SeasonalType,
    ETS, ETSSpec, AutoETS,
};

// Simple exponential smoothing with optimized alpha
let mut model = SimpleExponentialSmoothing::optimized();
model.fit(&ts).unwrap();

// Damped trend (safe default for trending data)
let mut model = HoltLinearTrend::auto_damped();
model.fit(&ts).unwrap();

// Holt-Winters with multiplicative seasonality
let mut model = HoltWinters::auto(12, SeasonalType::Multiplicative);
model.fit(&ts).unwrap();

// Specific ETS model: ETS(A,Ad,M) = additive error, damped trend, multiplicative seasonal
let mut model = ETS::new(ETSSpec::new(
    anofox_forecast::models::exponential::ErrorType::Additive,
    anofox_forecast::models::exponential::TrendType::AdditiveDamped,
    anofox_forecast::models::exponential::ETSSeasonalType::Multiplicative,
), 12);
model.fit(&ts).unwrap();

// Automatic ETS selection (recommended starting point)
let mut model = AutoETS::with_period(12);
model.fit(&ts).unwrap();

let forecast = model.predict_with_intervals(12, 0.95).unwrap();
```

**Parameter guidance:**
- Alpha (level): Higher values (0.5-0.9) react quickly to changes; lower values (0.01-0.2) produce smoother forecasts.
- Beta (trend): Keep lower than alpha. Values above 0.3 rarely help.
- Gamma (seasonal): Typically 0.01-0.3. Higher values allow seasonal patterns to evolve quickly.
- Phi (damping): Range 0.8-1.0. Values near 0.98 provide mild damping. Use `auto_damped()` to optimize.

### 3. ARIMA Family

Best when the data has significant autocorrelation structure that ETS does not capture well.

**ARIMA** -- Specify (p, d, q) orders manually. Use d=1 for trending data (first differencing makes it stationary). AR terms (p) capture momentum; MA terms (q) capture shocks.

**SARIMA** -- Extends ARIMA with seasonal (P, D, Q)[s] components. Use when both non-seasonal and seasonal autocorrelation are present.

**AutoARIMA** -- Searches over (p, d, q) and optionally (P, D, Q)[s] orders to find the best fit by AIC. Supports stepwise search (fast) and exhaustive search (thorough). Use `with_true_stepwise()` for faster hill-climbing search.

```rust
use anofox_forecast::models::arima::{ARIMA, ARIMASpec, SARIMA, SARIMASpec, AutoARIMA};

// Manual ARIMA(1,1,1)
let mut model = ARIMA::new(ARIMASpec::new(1, 1, 1));
model.fit(&ts).unwrap();

// Seasonal ARIMA(1,1,1)(1,1,1)[12]
let mut model = SARIMA::new(SARIMASpec::new(1, 1, 1, 1, 1, 1, 12));
model.fit(&ts).unwrap();

// Automatic selection (non-seasonal)
let mut model = AutoARIMA::new();
model.fit(&ts).unwrap();

// Automatic selection (seasonal, with stepwise search)
let mut model = AutoARIMA::seasonal(12);
model.fit(&ts).unwrap();

let forecast = model.predict(12).unwrap();
```

**When ARIMA beats ETS:**
- Data has strong autocorrelation (e.g., AR(2) structure).
- Residuals from ETS show significant autocorrelation patterns.
- The data has been differenced and the differenced series is well-behaved.
- You need exogenous regressors (ARIMAX). Both ARIMA and SARIMA support exogenous variables.

### 4. Theta Family

The Theta method won the M3 forecasting competition. It decomposes the series into "theta lines" and combines SES with a linear trend.

**Theta (STM)** -- Standard Theta Model with fixed alpha=0.1 and theta=2.0. Fast and surprisingly effective.

**OptimizedTheta (OTM)** -- Optimizes both alpha and theta parameters. Better than STM when defaults are suboptimal.

**DynamicTheta (DSTM/DOTM)** -- Updates linear coefficients dynamically, adapting to changing trends. DOTM (dynamic + optimized) was a component of the M4 competition winner.

**AutoTheta** -- Fits all four variants (STM, OTM, DSTM, DOTM) and selects the best by in-sample MSE.

All Theta models support seasonal decomposition (multiplicative by default, matching statsforecast).

```rust
use anofox_forecast::models::theta::{
    Theta, OptimizedTheta, DynamicTheta, DynamicOptimizedTheta, AutoTheta,
};

// Standard Theta (fast, good default)
let mut model = Theta::new();
model.fit(&ts).unwrap();

// Seasonal Theta (multiplicative decomposition by default)
let mut model = Theta::seasonal(12);
model.fit(&ts).unwrap();

// Optimized Theta
let mut model = OptimizedTheta::new();
model.fit(&ts).unwrap();

// Auto Theta (recommended -- picks best variant)
let mut model = AutoTheta::new();
model.fit(&ts).unwrap();

// Seasonal Auto Theta
let mut model = AutoTheta::seasonal(12);
model.fit(&ts).unwrap();

let forecast = model.predict(12).unwrap();
```

**Strengths:** Excellent out-of-the-box performance; robust to noise; fast.
**Limitations:** Assumes a simple decomposition structure; may struggle with complex non-linear patterns.

### 5. Intermittent Demand Models

For time series where demand is sporadic -- many zero values interspersed with occasional non-zero demand. Standard models break down on this pattern.

**Croston (Classic)** -- Separately smooths demand sizes and inter-arrival intervals. The original intermittent demand method.

**Croston SBA** -- Syntetos-Boylan Approximation. Applies bias correction to classic Croston. Generally preferred over classic Croston.

**TSB (Teunter-Syntetos-Babai)** -- Models demand probability and demand size separately. Handles obsolescence: if demand probability trends to zero, forecasts naturally decline. Use when items may become obsolete.

**ADIDA** -- Aggregate-Disaggregate Intermittent Demand Approach. Aggregates the series to reduce intermittency, applies SES, then disaggregates. Simple and effective.

**IMAPA** -- Tests multiple aggregation levels and averages forecasts. Most robust intermittent demand method.

```rust
use anofox_forecast::models::intermittent::{Croston, CrostonVariant, TSB, ADIDA, IMAPA};

// Croston SBA (bias-corrected, recommended default)
let mut model = Croston::new().sba();
model.fit(&ts).unwrap();

// TSB for potential obsolescence
let mut model = TSB::new();
model.fit(&ts).unwrap();

// ADIDA with automatic aggregation
let mut model = ADIDA::new();
model.fit(&ts).unwrap();

// IMAPA (most robust)
let mut model = IMAPA::new();
model.fit(&ts).unwrap();

let forecast = model.predict(12).unwrap();
```

**How to detect intermittent demand:** If more than ~30% of observations are zero and non-zero demands are separated by irregular intervals, use an intermittent demand model.

### 6. TBATS

For time series with complex seasonal patterns that simpler models cannot handle.

**TBATS** handles:
- Multiple seasonal periods simultaneously (e.g., hourly data with daily + weekly patterns).
- Non-integer seasonality (e.g., 365.25 days/year).
- Box-Cox variance stabilization.
- ARMA errors for residual autocorrelation.

Uses Fourier (trigonometric) terms to represent seasonality, which is much more parsimonious than explicit seasonal states.

**AutoTBATS** automatically selects the best TBATS configuration by comparing models with/without Box-Cox, with/without trend, with/without damping, and different Fourier harmonic counts.

```rust
use anofox_forecast::models::tbats::{TBATS, AutoTBATS};

// Single seasonality (e.g., daily pattern in hourly data)
let mut model = TBATS::new(vec![24]);
model.fit(&ts).unwrap();

// Multiple seasonalities (e.g., daily + weekly in hourly data)
let mut model = TBATS::new(vec![24, 168]);
model.fit(&ts).unwrap();

// Automatic configuration selection
let mut model = AutoTBATS::new(vec![24, 168]);
model.fit(&ts).unwrap();

let forecast = model.predict(48).unwrap();
```

### 7. MFLES and MSTLForecaster

Decomposition-based approaches for complex data.

**MFLES (Median Fourier Linear Exponential Smoothing)** -- Gradient-boosted decomposition that iteratively fits median, Fourier (seasonal), linear trend, and exponential smoothing components. Robust to outliers. Supports exogenous regressors.

**MSTLForecaster** -- Uses MSTL (Multiple Seasonal-Trend decomposition using LOESS) to decompose the series, then forecasts each component separately. Supports multiple seasonal periods and configurable trend/seasonal forecast methods.

```rust
use anofox_forecast::models::MFLES;
use anofox_forecast::models::MSTLForecaster;

// MFLES with monthly seasonality
let mut model = MFLES::new(vec![12]);
model.fit(&ts).unwrap();

// MSTLForecaster with multiple seasonalities
let mut model = MSTLForecaster::new(vec![24, 168]);
model.fit(&ts).unwrap();

let forecast = model.predict(24).unwrap();
```

### 8. GARCH

For financial time series where volatility (variance) changes over time.

**GARCH(p,q)** models conditional variance. It does not forecast the level of the series -- it forecasts how volatile the series will be. Use `forecast_variance()` for analytical variance forecasts.

```rust
use anofox_forecast::models::GARCH;

let mut model = GARCH::new(1, 1);  // GARCH(1,1)
model.fit(&ts).unwrap();

// Forecast future variance
let variance_forecast = model.forecast_variance(10).unwrap();
```


## Automatic Selection

When you are unsure which model to use, the library offers three levels of automatic selection.

### AutoETS -- Best for: Single-family selection with ETS

Searches all valid ETS(Error, Trend, Seasonal) combinations and selects by AICc. Fast and reliable for most standard time series.

```rust
use anofox_forecast::models::exponential::AutoETS;

let mut model = AutoETS::with_period(12);
model.fit(&ts).unwrap();
```

### AutoARIMA -- Best for: Single-family selection with ARIMA

Searches ARIMA orders using stepwise or exhaustive search. Preferred when autocorrelation is the dominant feature.

```rust
use anofox_forecast::models::arima::AutoARIMA;

let mut model = AutoARIMA::seasonal(12);
model.fit(&ts).unwrap();
```

### AutoTheta -- Best for: Single-family selection with Theta

Compares STM, OTM, DSTM, and DOTM variants. Fast and effective, especially for competition-style forecasting.

```rust
use anofox_forecast::models::theta::AutoTheta;

let mut model = AutoTheta::seasonal(12);
model.fit(&ts).unwrap();
```

### AutoForecast -- Best for: Cross-family comparison

Fits AutoARIMA, AutoETS, and AutoTheta, then selects the single best model by in-sample MSE or cross-validation error. Use this when you have no prior knowledge about your data.

```rust
use anofox_forecast::models::auto_forecast::{AutoForecast, AutoForecastConfig, SelectionStrategy};

// Default: compares all three families by in-sample MSE
let mut model = AutoForecast::new();
model.fit(&ts).unwrap();
println!("Selected: {}", model.selected_model_name().unwrap());

// Seasonal, with cross-validation (more robust, slower)
let config = AutoForecastConfig::with_period(12)
    .with_selection(SelectionStrategy::CrossValidation);
let mut model = AutoForecast::with_config(config);
model.fit(&ts).unwrap();
```

### When to use which Auto model

| Scenario | Recommended |
|----------|-------------|
| Quick exploration, general data | `AutoForecast` |
| You suspect ETS will work well | `AutoETS` |
| Data has strong autocorrelation | `AutoARIMA` |
| Competition-style forecasting | `AutoTheta` |
| Maximum accuracy, cost is not an issue | `AutoEnsemble` |
| Cross-validation-based selection | `AutoForecast` with `SelectionStrategy::CrossValidation` |


## Ensemble Methods

Combining multiple models often produces more accurate and robust forecasts than any single model.

### Manual Ensemble

Combine any set of models with a chosen combination method.

```rust
use anofox_forecast::models::ensemble::{Ensemble, CombinationMethod};
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::theta::AutoTheta;

let ensemble = Ensemble::new(vec![
    Box::new(AutoETS::with_period(12)),
    Box::new(AutoARIMA::seasonal(12)),
    Box::new(AutoTheta::seasonal(12)),
]).with_method(CombinationMethod::WeightedMSE);
```

**Combination methods:**
- `Mean` -- Simple average. Most robust when you have no reason to prefer one model.
- `Median` -- Robust to outlier models. Good when one model might fail badly.
- `WeightedMSE` -- Weights models by inverse in-sample MSE. Gives more weight to better-fitting models.
- `Custom` -- Provide your own weights via `.with_weights(vec![...])`.

### AutoEnsemble

Automatically fits AutoARIMA, AutoETS, and AutoTheta, ranks by in-sample MSE, and combines the top-K models.

```rust
use anofox_forecast::models::ensemble::{AutoEnsemble, AutoEnsembleConfig};

// Default: top 3 models, weighted by inverse MSE
let mut model = AutoEnsemble::new();
model.fit(&ts).unwrap();

// Seasonal, top 2 models
let config = AutoEnsembleConfig::with_period(12).with_top_k(2);
let mut model = AutoEnsemble::with_config(config);
model.fit(&ts).unwrap();

let forecast = model.predict(12).unwrap();
```

**When ensembles help most:**
- You do not know which model family is best for your data.
- Forecast accuracy matters more than speed or interpretability.
- Models disagree on the forecast (high model uncertainty).

**When ensembles may not help:**
- One model clearly dominates (adding worse models hurts the average).
- Speed is critical (ensembles fit multiple models).
- You need interpretable model parameters.


## Common Patterns

### Pattern: Stationary data (no trend, no seasonality)
**Try:** Naive, SES, HistoricAverage
**Auto:** AutoETS (will likely select ETS(A,N,N) or ETS(M,N,N))

### Pattern: Linear trend, no seasonality
**Try:** RandomWalkWithDrift, HoltLinearTrend, HoltLinearTrend(damped)
**Auto:** AutoETS, AutoForecast

### Pattern: Monthly data with yearly seasonality
**Try:** SeasonalNaive, HoltWinters(period=12), AutoETS(period=12)
**Auto:** AutoForecast::seasonal(12)

### Pattern: Hourly data with daily + weekly seasonality
**Try:** TBATS(vec![24, 168]), MSTLForecaster(vec![24, 168]), MFLES(vec![24])
**Auto:** AutoTBATS(vec![24, 168])

### Pattern: Sparse/intermittent demand (many zeros)
**Try:** Croston SBA, TSB, IMAPA
**Do not use:** ETS, ARIMA, Theta (these assume continuous demand)

### Pattern: Financial returns / volatility
**Try:** GARCH(1,1) for variance forecasting
**Combine with:** ARIMA for level forecasting + GARCH for variance

### Pattern: Data with exogenous regressors
**Try:** ARIMA/SARIMA (supports ARIMAX), ETS (via AutoETS), Theta, MFLES, Naive
**Note:** Fit with a `TimeSeries` that has regressors attached, then use `predict_with_exog()`.

### Pattern: Short series (< 30 observations)
**Try:** Naive, SES, Theta (STM), RandomWalkWithDrift
**Avoid:** Complex models (TBATS, SARIMA with many parameters) -- they need more data to estimate reliably.

### Pattern: Unknown data, want best result
**Try:** AutoForecast or AutoEnsemble
**Validate with:** Cross-validation (`cross_validate` from `utils`)
