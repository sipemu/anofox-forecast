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

Provides 40+ forecasting models, 76+ statistical features, automatic model selection, ensemble methods, seasonality decomposition, changepoint detection, anomaly detection, hierarchical reconciliation, and model serialization.

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

### Forecasting

- **Forecasting Models (40+)**
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
  - VAR (Vector Autoregression) for multivariate forecasting with Granger causality
  - Kalman filter / state-space models (local level, local linear trend, custom)
  - Exogenous regressor support across model families

- **Automatic Model Selection**
  - `AutoForecast`: Unified selection across ARIMA, ETS, and Theta families (parallel with `parallel` feature)
  - `AutoEnsemble`: Automatic ensemble of top-K best models
  - Selection by in-sample MSE or cross-validation error
  - Builder API: `AutoForecast::builder().seasonal_period(12).include_arima(true).build()`
  - `fit_predict()` convenience method on all models

- **Ensemble Methods**
  - Mean, Median, Weighted MSE, InverseAIC, Stacking, HorizonAdaptive combination strategies
  - Widest-envelope interval combination for ensemble prediction intervals
  - Automatic ensemble construction from model registry
  - `ensemble_best_k()`: Auto-select top-k models by holdout performance

- **Hierarchical Forecasting**
  - `HierarchyTree`: Define parent-children structure for grouped series
  - Bottom-up, top-down, MiddleOut, MinTrace OLS, and MinTrace Shrink (Ledoit-Wolf) reconciliation
  - Ensures coherent forecasts across hierarchical levels

### Analysis & Decomposition

- **Seasonality & Decomposition**
  - `SeasonalComponent` / `TrendComponent` traits — composable, dual-purpose (standalone + feature extraction)
  - STL (Seasonal-Trend decomposition using LOESS) with `StlBuilder` for ergonomic configuration
  - MSTL (Multiple Seasonal-Trend decomposition) for complex seasonality
  - Prophet-style Fourier seasonality (`FourierSeasonality`) with flexible harmonic modeling
  - Dummy (one-hot) seasonality (`DummySeasonality`) — captures arbitrary seasonal shapes without smoothness assumptions
  - Seasonal differencing (`SeasonalDifference`) — standalone transform with strength/variance-reduction features
  - Hodrick-Prescott filter (`HodrickPrescottFilter`) — smooth trend extraction with cycle decomposition
  - Piecewise linear trend (`PiecewiseLinearTrend`) — PELT-based changepoint detection + per-segment regression
  - Polynomial trend (`PolynomialTrend`) — degree 1-3, Vandermonde + Cholesky solve
  - Exponential trend (`ExponentialTrend`) — log-linear regression for growth/decay
  - Logistic trend (`LogisticTrend`) — S-curve fitting with auto or fixed capacity
  - Theil-Sen trend (`TheilSenTrend`) — robust median-of-pairwise-slopes estimator
  - `AutoTrend` — automatic selection of best trend component via AICc/BIC
  - `AutoSeasonal` — automatic selection of best seasonal component via AICc/BIC
  - `Recency` — fit on recent data only (last N, last X%, full, or Auto via PELT changepoint detection) for trend-aware forecasting
  - `TimeSeries::seasonal_strength()` / `trend_strength()` — quick strength assessment
  - Convenience: `deseasonalize()`, `detrend()`, `seasonal_adjust()`, `recompose()`

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

- **Changepoint Detection**
  - PELT algorithm with O(n) average complexity
  - Builder API: `Pelt::new(CostFunction::L2).min_size(5).penalty(5.0).detect(&data)`
  - Multiple cost functions: L1, L2, Normal, Poisson, LinearTrend, MeanVariance, Cusum

- **Anomaly Detection & Outlier Handling**
  - Statistical methods (IQR, z-score, modified z-score)
  - Automatic threshold selection
  - `TimeSeries::with_outliers_replaced()` — automatic outlier replacement with local median

### Evaluation & Uncertainty

- **Model Comparison & Evaluation**
  - `compare_models()`: Side-by-side model evaluation with timing
  - `compare_registry()`: Compare all registered models at once
  - `fit_all_and_compare()`: Fit all registry models, rank by holdout accuracy
  - `cross_validate_all()`: CV all registry models at once with aggregated metrics
  - Accuracy metrics: MAE, MSE, RMSE, MAPE, sMAPE, MASE, WAPE, MDA, Theil's U, MSIS, coverage, skill scores
  - `ForecastMetrics::compute()`: All 10 core metrics in a single call
  - Time series cross-validation with configurable strategies and embargo
  - `rolling_forecast()`: Walk-forward evaluation with rolling/expanding windows
  - Streaming CV aggregation with early stopping (`cross_validate_early_stop()`)
  - `ModelDiagnostics`: Ljung-Box, Jarque-Bera, Breusch-Pagan residual diagnostics
  - `IntermittentDiagnostics`: Syntetos-Boylan demand classification (Smooth/Erratic/Intermittent/Lumpy)
  - `AidAnalyzer`: Automatic Identification of Demand — distribution fitting, demand type classification, and per-observation anomaly detection (stockouts, lifecycle events, outliers)

- **Probabilistic Postprocessing**
  - Conformal Prediction: Distribution-free intervals with coverage guarantees
  - Historical Simulation: Non-parametric empirical error distribution
  - Normal Predictor: Gaussian error assumption baseline
  - IDR: Isotonic Distributional Regression (state-of-the-art calibration)
  - QRA: Quantile Regression Averaging for ensemble combining
  - Backtesting: Rolling/expanding window evaluation with horizon-aware calibration

- **Bootstrap Confidence Intervals**
  - Residual bootstrap and block bootstrap methods
  - Empirical confidence intervals for any model
  - Configurable sample size and reproducibility

- **Forecast Constraints**
  - `NonNegative`, `LowerBound`, `UpperBound`, `Bounds`, `IntegerRound`, `Custom`
  - Convenience methods: `forecast.non_negative()`, `.clamp(lo, hi)`, `.round_to_integer()`
  - Constraints apply to point forecasts and prediction intervals

- **Forecast Explainability**
  - `Explainable` trait with `ForecastExplanation` (level, trend, seasonal, residual, named components)
  - Implemented for ETS, Theta, and MSTL models
  - Components sum to forecast values for verification

### Data Processing & Pipeline

- **Orchestration / Agent Forecasting** (`orchestration` module)
  - `DataProfile`: Automated data profiling — stationarity (ADF), trend direction, seasonality, ACF, quality score, changepoint detection
  - `PipelineBuilder`: Declarative pipeline — profile → preprocess → changepoint → trend → model selection → cross-validation → ensemble → postprocess → constraints
  - `Pipeline::from_config()`: Replay a pipeline from saved `PipelineConfig`
  - `PreprocessMode`: Automatic preprocessing — Box-Cox for skewed data, outlier replacement for low-quality data
  - `MetricStrategy`: Data-aware multi-metric model selection — Auto, Single, or weighted Composite of MAE/MSE/RMSE/SMAPE/WAPE/MDA
  - `EnsembleMode`: Auto (MCS-based), Fixed (specify combination method), or None (single best)
  - `TrendIntegration`: Decompose (detrend → forecast residuals → recompose) or Regressor (trend as exogenous feature `"__trend"`)
  - `ChangepointMode`: Auto (PELT-based) or FitFrom(index) — adapt training to regime changes with safety checks
  - `PipelineReport`: Multi-section structured report from pipeline results (summary, profile, forecast, decision log, etc.)
  - `PipelineStore` trait: Abstract storage backend with `Value` IR — `InMemoryStore` included, swap in DuckDB/SQLite
  - Structured tool functions: `profile_data`, `select_models`, `run_pipeline`, `explain_result` — MCP-ready with typed I/O
  - `DecisionLog`: Structured audit trail with categories, outcomes, and timing
  - `FallbackChain`: Ordered model failover with automatic recovery
  - `HorizonAnalysis`: Per-step-ahead error decomposition (RMSE, MAE, bias per horizon)
  - `SelectionConfidence`: Diebold-Mariano pairwise test for forecast accuracy comparison
  - `ModelConfidenceSet`: Bootstrap-based set of statistically best models (Hansen et al. 2011)
  - `QualityFloor`: Superior Predictive Ability test — does any model beat the benchmark? (Hansen 2005)
  - `ExecutionMetadata` / `ExecutionTimer`: Fit/predict timing and convergence tracking

- **Batch Processing & Parallelism**
  - `fit_predict_many()`: Fit one model across many series (parallel with `parallel` feature)
  - `fit_registry()`: Fit all registered models on a series (parallel)
  - `compare_models()` / `compare_registry()`: Parallel model comparison
  - Cross-validation folds run in parallel when `parallel` feature is enabled
  - Bootstrap sampling uses `par_iter` when `parallel` is enabled

- **Data Transformations**
  - Scaling: standardization, min-max, robust scaling
  - Box-Cox transformation with automatic lambda selection
  - Window functions: rolling mean, std, min, max, median
  - Exponential weighted moving averages

- **Missing Value Imputation**
  - Policy-based: Drop, Fill, ForwardFill, BackwardFill, FillMean, FillMedian, Interpolate
  - Advanced: moving average imputation, seasonal median imputation
  - Convenience: forward-backward fill, regressor imputation
  - Metadata: missing mask, per-dimension missing counts

- **TimeSeries Temporal Aggregation**
  - `aggregate(period, method)` — Sum, Mean, Median, First, Last, Min, Max
  - `downsample(factor)` — decimation with timestamp preservation
  - `upsample(factor, method)` — Linear, ForwardFill, BackwardFill, Zero interpolation
  - `sliding_window_aggregate(window, step, method)` — configurable sliding windows

### Persistence & Interoperability

- **Model Serialization** (optional `serde` feature)
  - Save/load models to JSON with `to_json()`/`from_json()`
  - Binary serialization with `to_bincode()`/`from_bincode()` for compact storage
  - File persistence with `save_to_file()`/`load_from_file()`
  - Round-trip serialization for all major model families

- **Model Warm-Starting**
  - `ETS::with_initial_states()` — start from pre-fitted level/trend/seasonal states
  - `SES::with_alpha()` — use pre-fitted smoothing parameter
  - `ARIMA::with_coefficients()` — use pre-fitted AR/MA coefficients
  - `Theta::with_theta_value()` — use specified theta parameter
  - `Forecaster::fitted_params()` — extract fitted parameters for transfer

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
| **Multivariate** | `VAR` (Vector Autoregression) |
| **State-Space** | `KalmanFilter`, `StateSpaceModel` (local level, local linear trend) |
| **Ensemble** | `Ensemble` (Mean, Median, Weighted MSE, InverseAIC, Stacking, HorizonAdaptive) |
| **Hierarchical** | `HierarchyTree` (BottomUp, TopDown, MiddleOut, MinTraceOls, MinTraceShrink) |

### Utilities

| Function / Type | Description |
|-----------------|-------------|
| `compare_models()` | Compare forecasters on the same data with timing |
| `compare_registry()` | Compare all registered models at once |
| `cross_validate()` | Time series cross-validation (parallel with `parallel` feature) |
| `cross_validate_early_stop()` | CV with convergence-based early stopping |
| `rolling_forecast()` | Walk-forward evaluation with rolling/expanding windows |
| `StreamingCVAggregator` | Online metric aggregation using Welford's algorithm |
| `fit_predict_many()` | Batch fit-predict across multiple series |
| `bootstrap_forecast()` | Bootstrap confidence intervals for any model |
| `diagnose_residuals()` | Unified residual diagnostics (Ljung-Box, DW, Jarque-Bera) |
| `ModelDiagnostics` | Comprehensive diagnostics: Ljung-Box, Jarque-Bera, Breusch-Pagan |
| `IntermittentDiagnostics` | Syntetos-Boylan demand classification with model recommendations |
| `AidAnalyzer` | Automatic Identification of Demand: distribution fitting, anomaly detection |
| `PipelineBuilder` | Declarative forecasting pipeline with profiling, preprocessing, multi-metric selection, ensemble, fallback |
| `DataProfile` | Automated data profiling (stationarity, trend, seasonality, quality) |
| `PreprocessMode` | Auto/Manual preprocessing (Box-Cox, outlier replacement) |
| `MetricStrategy` | Data-aware multi-metric model selection (Auto/Single/Composite) |
| `EnsembleMode` | Auto (MCS-based) / Fixed / None ensemble construction |
| `PipelineReport` | Multi-section structured report from pipeline results |
| `PipelineStore` | Abstract storage trait with `Value` IR — backend-agnostic persistence |
| `SelectionConfidence` | Diebold-Mariano pairwise forecast comparison |
| `ModelConfidenceSet` | Bootstrap model confidence set (Hansen et al. 2011) |
| `QualityFloor` | Superior Predictive Ability test vs benchmark |
| `HorizonAnalysis` | Per-step-ahead error decomposition |
| `bias()` / `periods_in_stock()` | Signed bias and inventory-focused PIS metric |
| `ForecastMetrics::compute()` | All 10 metrics in one call (MAE through Theil's U) |
| `fit_all_and_compare()` | Fit all registry models, rank by holdout accuracy |
| `cross_validate_all()` | CV all registry models with aggregated metrics |
| `ensemble_best_k()` | Auto-select top-k models into an ensemble |
| `SeasonalComponent` / `TrendComponent` | Composable traits for seasonal/trend components (standalone + features) |
| `DummySeasonality` | One-hot seasonal encoding — arbitrary seasonal shapes |
| `SeasonalDifference` | Standalone seasonal differencing with strength/variance features |
| `HodrickPrescottFilter` | Smooth trend extraction with cycle decomposition |
| `PiecewiseLinearTrend` | PELT-based piecewise linear trend with per-segment regression |
| `PolynomialTrend` | Polynomial trend (degree 1-3) with Cholesky solve |
| `ExponentialTrend` | Log-linear exponential growth/decay trend |
| `LogisticTrend` | Logistic S-curve trend with auto/fixed capacity |
| `TheilSenTrend` | Robust Theil-Sen median-slope trend estimator |
| `AutoTrend` | Automatic best-trend selection via AICc/BIC/holdout |
| `AutoSeasonal` | Automatic best-seasonal selection via AICc/BIC |
| `Recency` | Fit on recent data only (Window, Fraction, Full, Auto via PELT) |
| `TrendIntegration` | Decompose (detrend/recompose) or Regressor (trend as exog feature) |
| `ChangepointMode` | Auto (PELT) or FitFrom — regime-aware training with safety checks |
| `deseasonalize()` / `seasonal_adjust()` | Remove seasonal component from data or TimeSeries |
| `select_features()` | Automated feature selection (variance, correlation, top-K) |
| `to_json()` / `from_json()` | Serialization for models, `Forecast`, and `TimeSeries` (requires `serde` feature) |
| `to_bincode()` / `from_bincode()` | Binary serialization (requires `serde` feature) |

### Feature Categories

| Category | Examples |
|----------|----------|
| Basic | `mean`, `variance`, `minimum`, `maximum`, `quantile` |
| Distribution | `skewness`, `kurtosis`, `variation_coefficient` |
| Autocorrelation | `autocorrelation`, `partial_autocorrelation` |
| Entropy | `approximate_entropy`, `sample_entropy`, `permutation_entropy` |
| Complexity | `c3`, `cid_ce`, `lempel_ziv_complexity` |
| Trend | `linear_trend`, `adf_test`, `ar_coefficient`, `hp_trend_strength`, `piecewise_n_segments` |
| Seasonality | `dummy_seasonal_strength`, `seasonal_diff_strength`, `seasonal_diff_variance_reduction` |
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
- [anofox-statistics](https://crates.io/crates/anofox-statistics) - Statistical hypothesis tests (DM, MCS, SPA)
- [statrs](https://crates.io/crates/statrs) - Statistical distributions and functions
- [thiserror](https://crates.io/crates/thiserror) - Error handling
- [rand](https://crates.io/crates/rand) - Random number generation
- [rustfft](https://crates.io/crates/rustfft) - Fast Fourier Transform for spectral analysis

## Acknowledgments

The postprocessing module is a Rust port of [PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl). Feature extraction is inspired by [tsfresh](https://github.com/blue-yonder/tsfresh). Forecasting models are validated against [StatsForecast](https://github.com/Nixtla/statsforecast) by Nixtla. See [THIRDPARTY_NOTICE.md](THIRDPARTY_NOTICE.md) for full attribution and references to the research papers that inspired this implementation.

## License

MIT License - see [LICENSE](LICENSE) for details.
