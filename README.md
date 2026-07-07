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

Provides 50+ forecasting models, 76+ statistical features, automatic model selection, ensemble methods, seasonality decomposition, changepoint detection, anomaly detection, hierarchical reconciliation, forecastability analysis, and model serialization.

## Use Cases

**Want to try it out?** Use the [anofox app](https://muon-stat.com/apps/anofox-app/) for interactive forecasting in the browser.

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

- **Forecasting Models (50+)**
  - ARIMA, SARIMA, and AutoARIMA with automatic order selection
  - Exponential Smoothing: SES, Holt's Linear, Holt-Winters, SeasonalES
  - ETS (Error-Trend-Seasonal) state-space framework with AutoETS and `ModelPool` (Reduced/Complete/DampedTrendOnly/MatchErrorSeasonal)
  - Baseline methods: Naive, Seasonal Naive, Random Walk with Drift, SMA, Window Average
  - Theta family: Theta, Optimized Theta, Dynamic Theta, AutoTheta
  - Intermittent demand: Croston, ADIDA, TSB, IMAPA
  - TBATS/AutoTBATS for complex seasonality
  - MFLES (Multiple Frequency Locally Estimated Scatterplot) with cached Cholesky Fourier OLS
  - MSTL-based forecasting with configurable trend/seasonal methods and pre-regression exogenous support
  - GARCH for volatility modeling
  - VAR (Vector Autoregression) for multivariate forecasting with Granger causality
  - Kalman filter / state-space models (local level, local linear trend, custom)
  - Exogenous regressor support across model families with OLS coefficient extraction (`exog_coefficients()`)
  - `FeatureGenerator`: deterministic regressor generation (Fourier harmonics, day-of-week, month-of-year, quarter, holiday indicators, cyclical sin/cos encoding, binary calendar indicators)
  - `RegressionForecaster`: `anofox-regression` backends as `Forecaster` — 11 regression backends (OLS, Ridge, ElasticNet, Quantile, WLS, RLS, Tweedie, Poisson, BLS, NNLS, Dynamic), configurable trend/seasonal/structural features, recursive multi-step prediction, auto-lag selection (BIC/AIC), differencing and seasonal differencing, **rolling-window features** (mean/std/var/min/max/median/sum/EWM) via the `RecursiveFeature` trait — recomputed at every horizon step using the rolling history buffer, with built-in `lag >= 1` leakage guard

- **Automatic Model Selection**
  - `AutoForecast`: Unified selection across ARIMA, ETS, and Theta families (parallel with `parallel` feature)
  - `AutoEnsemble`: Automatic ensemble of top-K best models
  - Selection by cross-validation error
  - Builder API: `AutoForecast::builder().seasonal_period(12).include_arima(true).build()`
  - `fit_predict()` convenience method on all models

- **Batch / Global Forecasting** — process many series with shared computation
  - `GlobalETS`: shared smoothing params across N series (75-96x faster for seasonal ETS)
  - `GlobalAutoETS`: automatic model selection across N series (28-32x faster)
  - `GlobalCroston`: shared α across N intermittent demand series (3-6x faster)
  - `GlobalTheta`: shared α for Standard Theta Method
  - `batch::auto_ets()`, `batch::ets()`, `batch::mfles()`: parallel batch convenience functions
  - `STL::decompose_batch()`: batch decomposition with parallel support
  - Validated on M5 dataset (30,490 series): identical accuracy, 2x speedup with Reduced pool

- **Ensemble Methods**
  - Mean, Median, Weighted MSE, InverseAIC, Stacking, HorizonAdaptive combination strategies
  - Widest-envelope interval combination for ensemble prediction intervals
  - Automatic ensemble construction from model registry
  - `ensemble_best_k()`: Auto-select top-k models by holdout performance

- **Hierarchical Forecasting**
  - `HierarchyTree`: Define parent-children structure for grouped series
  - Bottom-up, top-down, MiddleOut, MinTrace OLS, and MinTrace Shrink (Ledoit-Wolf) reconciliation
  - Scalable MinTrace: `MinTraceVariance` and `MinTraceStruct` with sparse summing matrix — safe for 100k+ series (no N×N covariance)
  - Ensures coherent forecasts across hierarchical levels

### Analysis & Decomposition

- **Seasonality & Decomposition**
  - `SeasonalComponent` / `TrendComponent` traits — composable, dual-purpose (standalone + feature extraction)
  - STL (Seasonal-Trend decomposition using LOESS) with `StlBuilder` — optimized with running-sum MA and precomputed tricube kernel (2-2.5x faster)
  - MSTL (Multiple Seasonal-Trend decomposition) for complex seasonality, with pre-regression exogenous regressor support
  - Prophet-style Fourier seasonality (`FourierSeasonality`) with flexible harmonic modeling
  - Dummy (one-hot) seasonality (`DummySeasonality`) — captures arbitrary seasonal shapes without smoothness assumptions
  - Seasonal differencing (`SeasonalDifference`) — standalone transform with strength/variance-reduction features
  - Hodrick-Prescott filter (`HodrickPrescottFilter`) — smooth trend extraction with cycle decomposition
  - Christiano-Fitzgerald band-pass filter (`cf_filter`) — asymmetric, preserves full series length
  - Baxter-King band-pass filter (`bk_filter`) — symmetric, loses 2k edge observations
  - Hamilton filter (`hamilton_filter`) — regression-based trend-cycle decomposition (avoids HP endpoint bias)
  - Fractional differencing (`fractional_difference`) — memory-preserving stationarity (Lopez de Prado 2018)
  - `find_min_fractional_d()` — automatic minimum d for ADF stationarity
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

- **Changepoint Detection** — full [`ruptures`](https://github.com/deepcharles/ruptures) parity
  - 6 detection algorithms: `PeltDetector` (PELT pruning), `DynpDetector` (exact O(K·n²) dynamic programming), `BinsegDetector` (greedy binary segmentation), `BottomUpDetector` (agglomerative), `WindowDetector` (sliding-window), `KernelCpdDetector` (kernel — Linear / Rbf / Cosine)
  - 14 cost functions: `CostL1`, `CostL2`, `CostNormal`, `CostLinear` (multivariate OLS), `CostRank`, `CostMahalanobis`, `CostAR`, `CostRbf`, `CostCosine`, `CostCLinear` (= ruptures' 10) plus `CostPoisson`, `CostMeanVariance`, `CostCusum`, `CostLinearTrend` (extras)
  - 3 evaluation metrics: `precision_recall`, `hausdorff`, `randindex`
  - Multivariate-aware `Signal` carrier (`(n, d)` row-major); `From<&[f64]>` for univariate ergonomics
  - Trait-based composition: any `Cost` works with any `Detector`. Three predict modes: `predict_pen(p)`, `predict_n_bkps(K)`, `predict_eps(ε)`
  - Validated against `ruptures==1.1.9` via `scripts/generate_ruptures_fixtures.py` + `tests/changepoint_ruptures_parity.rs` — 5/5 fixtures match exactly on deterministic detectors
  - Legacy free-function API (`pelt_detect`, `Pelt::new`, `CostFunction` enum, `auto_detect()`, `crops()`) retained unchanged for backward compat

- **Sequential Monitoring of Forecast Errors** (`monitor::` module)
  - Online changepoint detection on a stream of forecast residuals — flags the moment a fitted model becomes inaccurate
  - Four CUSUM detectors: `PageCusum` (default), `PageCusum1`, `Cusum`, `Cusum1` (two-sided and one-sided variants)
  - Detects mean shifts (raw stream), variance shifts (squared stream), or both in parallel
  - Constant-size online state: `SequentialDetector::update(&new_errors)` is bit-equivalent to a fresh fit on the full series — production-safe streaming
  - Baked-in 228-entry critical-value table (4 detectors × 19 γ × 3 α) with reproducible Monte-Carlo regeneration
  - `Forecaster` trait integration: `monitor_forecaster()` (in-sample residuals, cheap) and `monitor_forecaster_cv()` (rolling-origin CV residuals, calibrated)
  - Rust port of [`changepoint.forecast`](https://github.com/grundy95/changepoint.forecast) by Thomas Grundy (Lancaster), based on [Fremdt (2014)](https://doi.org/10.1080/02331888.2014.921899)

- **Forecastability Analysis** (`forecastability::` module, feature-gated: `forecastability`)
  - Pre-modeling triage: determine whether a series has exploitable predictive structure *before* running expensive model search
  - **kNN Mutual Information** (Kraskov KSG1, 2004) with 2D KD-tree for O(n log n) bulk queries
  - **AMI curve**: `ami_curve(series, max_lag)` — MI at each horizon lag, revealing how far predictive signal reaches
  - **pAMI curve**: partial AMI via linear residualization — isolates direct dependence at each lag
  - **GCMI**: Gaussian Copula MI (Ince 2017) — captures only linear dependence; comparing with AMI reveals nonlinear structure
  - **Transfer Entropy**: `transfer_entropy_curve(source, target, max_lag)` — directional information flow between two series
  - **Distance correlation** (Szekely/Rizzo 2007) — detects both linear and nonlinear dependence, unlike Pearson
  - **Phase-randomized surrogates** with significance bands for any lag-curve metric
  - **Forecastability fingerprint**: `ForecastabilityFingerprint::compute()` → `information_mass`, `information_horizon`, `information_structure`, `nonlinear_share`, `signal_to_noise`, `directness_ratio`
  - **Largest Lyapunov exponent** (Rosenstein 1993) — detect chaos via delay-embedding divergence
  - **10-scorer registry**: Mi, Pearson, Spearman, Kendall, Distance, TransferEntropy, Gcmi, PermutationEntropy, SpectralEntropy, SpectralPredictability
  - Lag correlation curves: `pearson_curve`, `spearman_curve`, `kendall_curve` (O(n log n) Kendall via merge-sort)
  - Inspired by [`dependence-forecastability`](https://github.com/AdamKrysztopa/dependence-forecastability) by Adam Krysztopa (MIT)

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
  - Accuracy metrics: MAE, MSE, RMSE, MAPE, sMAPE, MASE, WAPE, MDA, Theil's U, RMSSE, WRMSSE, MSIS, coverage, skill scores
  - `ForecastMetrics::compute()`: All 10 core metrics in a single call
  - Time series cross-validation: backward-anchored folds, n_folds-driven, expanding/rolling windows, gap/purge/embargo
  - `rolling_forecast()`: Walk-forward evaluation with rolling/expanding windows
  - Streaming CV aggregation with early stopping (`cross_validate_early_stop()`)
  - `ModelDiagnostics`: Ljung-Box, Jarque-Bera, Breusch-Pagan residual diagnostics
  - `IntermittentDiagnostics`: Syntetos-Boylan demand classification (Smooth/Erratic/Intermittent/Lumpy)
  - `AidAnalyzer`: Automatic Identification of Demand — distribution fitting, demand type classification, and per-observation anomaly detection (stockouts, lifecycle events, outliers)

- **Probabilistic Postprocessing**
  - Conformal Prediction: Distribution-free intervals with coverage guarantees
  - Per-horizon-step conformal: separate interval widths per forecast step (tighter at h=1, wider at h=12)
  - Binned Conformal Prediction: Heteroscedastic intervals — bins residuals by predicted magnitude for wider intervals where uncertainty is larger
  - **CQR** (Conformalized Quantile Regression): wraps any quantile-regression base learner with symmetric bound adjustment — tighter than absolute-residual conformal under heteroscedasticity (Romano, Patterson & Candès, NeurIPS 2019)
  - **EnbPI** (Ensemble Bootstrap Prediction Interval): leave-one-out residuals from a bagged ensemble (no retraining); online window for drift (Xu & Xie, ICML 2021)
  - **ACI** (Adaptive Conformal Inference): stateful streaming wrapper with α-adaptation — long-run coverage converges to target under arbitrary distribution drift (Gibbs & Candès, NeurIPS 2021)
  - Bootstrap Prediction Intervals: Model-agnostic residual resampling with cumulative error paths (IID and block bootstrap)
  - Historical Simulation: Non-parametric empirical error distribution
  - Normal Predictor: Gaussian error assumption baseline
  - IDR: Isotonic Distributional Regression (state-of-the-art calibration)
  - QRA: Quantile Regression Averaging for ensemble combining
  - Multi-quantile forecasts: `predict_quantiles()` on Bootstrap and Conformal predictors (e.g., 10th/25th/50th/75th/90th percentiles)
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

- **Parallelism**
  - `compare_models()` / `compare_registry()`: Parallel model comparison
  - Cross-validation folds run in parallel when `parallel` feature is enabled
  - Bootstrap sampling uses `par_iter` when `parallel` is enabled

- **Data Transformations**
  - `Pipeline`: composable transform chains around any `Forecaster` — `Pipeline::builder().transform(BoxCoxTransform::auto()).transform(DifferenceTransform::new(1)).model(Box::new(Naive::new())).build()`
  - `Transform` trait: `DifferenceTransform`, `SeasonalDifferenceTransform`, `BoxCoxTransform`, `YeoJohnsonTransform` (Box-Cox extension that handles zeros and negatives), `ScaleTransform`, `LogTransform`
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
anofox-forecast = "0.12.0-alpha.22"
```

### Optional Features

```toml
[dependencies]
# Forecastability analysis (MI, GCMI, distance correlation, fingerprint)
anofox-forecast = { version = "0.8", features = ["forecastability"] }

# Forecastability + parallel (60× faster with rayon)
anofox-forecast = { version = "0.8", features = ["forecastability", "parallel"] }

# Parallel AutoARIMA (4-8x speedup via rayon, opt-in for embedding contexts like DuckDB)
anofox-forecast = { version = "0.8", features = ["parallel"] }

# Model serialization (save/load to JSON)
anofox-forecast = { version = "0.8", features = ["serde"] }

# Probabilistic postprocessing (conformal, IDR, QRA — enabled by default)
anofox-forecast = { version = "0.8", default-features = false }  # to disable
```

| Feature | Default | Description |
|---------|---------|-------------|
| `postprocess` | No | Conformal prediction, IDR, QRA, historical simulation |
| `forecastability` | No | kNN MI, GCMI, distance correlation, phase surrogates, fingerprint, Lyapunov exponent, 10-scorer registry |
| `parallel` | No | Rayon-based parallelism for AutoARIMA, AutoForecast, bootstrap, cross-validation, and forecastability (not available on WASM) |
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

### Rolling Features in Regression Models

```rust
use anofox_forecast::models::regression::{
    RegressionFeatures, RegressionForecaster, RollingStatKind,
};

// OLS with lag-1, a rolling mean of the last 7 values, and a rolling std.
// Every rolling feature is recomputed at each horizon step using the
// previous predictions — correct recursive multi-step semantics.
let mut model = RegressionForecaster::ols(
    RegressionFeatures::new()
        .no_trend()
        .lags(1)
        .with_rolling_mean(7)?                        // last 7, lag=1
        .with_rolling_std(14)?                        // last 14, lag=1
        .with_ewm_mean(20, 0.3)?                      // EWM window=20, α=0.3
        .no_exog(),
);
model.fit(&ts)?;
let forecast = model.predict(12)?;

// All `RollingStatKind` variants: Mean, Std, Var, Min, Max, Median, Sum,
// EwmMean { alpha }, EwmStd { alpha }. Custom lag via `.with_rolling_lagged(w, lag, kind)`.
// `lag == 0` is rejected at build time to prevent target leakage.
```

### Transform Pipeline

```rust
use anofox_forecast::transform::pipeline::{Pipeline, PipelineBuilder};
use anofox_forecast::transform::transforms::{BoxCoxTransform, DifferenceTransform};
use anofox_forecast::models::baseline::Naive;

// Chain transforms around any model — Pipeline itself implements Forecaster
let mut pipeline = Pipeline::builder()
    .transform(BoxCoxTransform::auto())
    .transform(DifferenceTransform::new(1))
    .model(Box::new(Naive::new()))
    .build();

pipeline.fit(&ts)?;
let forecast = pipeline.predict(12)?;
```

### Batch Forecasting (Many Series)

```rust
use anofox_forecast::models::exponential::{GlobalAutoETS, GlobalETS, ETSSpec, ModelPool};

// 1000 series, each a Vec<f64> — all same length
let all_series: Vec<Vec<f64>> = load_my_data();

// GlobalAutoETS: select best model per series, shared optimization (28-32x faster)
let mut model = GlobalAutoETS::new(12, ModelPool::Reduced);
model.fit(&all_series).unwrap();
let forecasts = model.predict(12); // Vec<Vec<f64>>, one per series

// GlobalETS: fit a known spec across all series (75-96x faster)
let mut model = GlobalETS::new(ETSSpec::ana(), 12);
model.fit(&all_series).unwrap();
let forecasts = model.predict(12);
```

### Exogenous Regressors

```rust
use anofox_forecast::features::FeatureGenerator;

// Generate deterministic regressors from timestamps
let gen = FeatureGenerator::new()
    .fourier(7, 2)       // Weekly Fourier terms
    .day_of_week()        // Day-of-week indicators
    .holiday("promo", promo_dates);

gen.add_to(&mut ts);     // Attach features to TimeSeries

let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts)?;

// Inspect OLS pre-regression coefficients
if let Some(ols) = model.exog_coefficients() {
    println!("Intercept: {:.4}", ols.intercept);
    for (name, coef) in ols.regressor_names.iter().zip(&ols.coefficients) {
        println!("  {}: {:.4}", name, coef);
    }
}
```

### Changepoint Detection

Full [`ruptures`](https://github.com/deepcharles/ruptures) parity — trait-based API where any `Cost` composes with any `Detector`, plus three predict modes:

```rust
use anofox_forecast::changepoint::{
    CostL2, CostRbf, Detector, DynpDetector, PeltDetector, Signal,
};

// Penalty mode (PELT, exact O(n) average via pruning)
let signal = Signal::univariate(&data);
let mut pelt = PeltDetector::new(CostL2::new()).min_size(5);
pelt.fit(&signal)?;
let r = pelt.predict_pen(3.0)?;
println!("{} CPs at {:?}", r.n_changepoints(), r.changepoints());

// Fixed number of changepoints (Dynp, exact dynamic programming)
let mut dynp = DynpDetector::new(CostL2::new()).min_size(5);
dynp.fit(&signal)?;
let r = dynp.predict_n_bkps(3)?;

// Kernel cost composes with any detector (Pelt, Dynp, Binseg, …)
let mut pelt_rbf = PeltDetector::new(CostRbf::auto());
pelt_rbf.fit(&signal)?;
let r = pelt_rbf.predict_pen(1.0)?;
```

Validated against `ruptures==1.1.9`: `scripts/generate_ruptures_fixtures.py` produces JSON fixtures from the canonical library, `tests/changepoint_ruptures_parity.rs` asserts identical breakpoints + total cost.

The legacy free-function API stays for backward compat:

```rust
use anofox_forecast::changepoint::{Pelt, CostFunction};

let result = Pelt::new(CostFunction::L2)
    .min_size(5)
    .auto_detect(&data);                          // CROPS + elbow detection
```

### Sequential Monitoring of Forecast Errors

Online detection of when a fitted model has become inaccurate. Port of the R
package [`changepoint.forecast`](https://github.com/grundy95/changepoint.forecast)
by Thomas Grundy, based on
[Fremdt (2014)](https://doi.org/10.1080/02331888.2014.921899).

```rust
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::monitor::{
    monitor_forecaster, Detector, ForecastErrorType, SequentialConfig, SequentialDetector,
};

// Option A: monitor a fitted forecaster's residuals directly
let mut model = Naive::new();
model.fit(&ts)?;

let cfg = SequentialConfig::new(200)        // training window length m
    .detector(Detector::PageCusum)          // recommended default
    .error_type(ForecastErrorType::Both);   // monitor mean AND variance
let detector = monitor_forecaster(&model, cfg)?;

if let Some(tau) = detector.first_detection() {
    println!("Model drifted at observation {}", tau);
}

// Option B: bring your own residual stream and update it online
let cfg = SequentialConfig::new(100).detector(Detector::PageCusum);
let mut detector = SequentialDetector::fit(&residuals, cfg)?;

// Each time a new actual arrives, compute the new error and stream it in.
// State is constant-size; this is bit-equivalent to a fresh fit on the full
// concatenated series.
detector.update(&[new_error])?;
if detector.has_detected() {
    // refit the forecasting model
}
```

### Forecastability Analysis

Pre-modeling triage: determine whether a series has exploitable predictive
structure before running expensive model search. Inspired by
[`dependence-forecastability`](https://github.com/AdamKrysztopa/dependence-forecastability)
by Adam Krysztopa.

```rust
use anofox_forecast::forecastability::{
    ForecastabilityFingerprint, ami_curve, gcmi_curve, score, Scorer,
};

// Quick fingerprint — answers "should I model this series?"
let fp = ForecastabilityFingerprint::compute(
    &values,
    20,       // max_lag: probe 20 lags
    100,      // n_surrogates: 100 phase surrogates
    0.05,     // alpha: 5% significance
    Some(42), // seed for reproducibility
);

if fp.information_mass < 1e-6 {
    println!("No signal — use Naive or skip");
} else if fp.nonlinear_share < 0.2 {
    println!("Linear signal → ARIMA/ETS");
} else {
    println!("Nonlinear signal → MFLES/tree-based regression");
}
println!("Signal reaches {} lags deep", fp.information_horizon);
println!("Informative lags: {:?}", fp.informative_horizons);

// Individual measures
let ami = ami_curve(&values, 10);     // MI at each horizon lag
let gcmi = gcmi_curve(&values, 10);   // linear-only MI for comparison
let mi_score = score(&values, Scorer::Mi);              // kNN MI at lag 1
let dcor = score(&values, Scorer::Distance);            // distance correlation
let pe = score(&values, Scorer::PermutationEntropy);    // regularity
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

## Demand Forecasting (α feature)

Behind the default `postprocess` + opt-in `distributional` features, the crate ships a distributional-forecasting shell inspired by [skaters](https://github.com/microprediction/skaters) and integrated with the `anofox-regression` AID (Automatic Identification of Demand) classifier. Three zero-config selectors are available; **which one wins depends on your panel type**.

### Choosing a selector

Full cross-panel benchmarks are in `examples/skaters_m5_full_auto.rs`, `skaters_m4_daily_benchmark.rs`, and `skaters_m3_monthly_benchmark.rs`. Summary:

| panel | domain | best selector | vs. AutoETS median MAE |
|-------|--------|---------------|------------------------|
| **M5 full 30k** | retail counts (all intermittent) | `LaplaceForecaster::new().auto_aid()` | **+0.8 %**, 42× faster than AutoETS |
| **M5 top-1000** | retail non-intermittent | `Laplace + AR2 + S7 + FD + OU` | +2.9 % |
| **M4 daily** | economic continuous | `LaplaceForecaster::new().auto()` | +7.5 % |
| **M3 monthly** | macroeconomic | `LaplaceForecaster::new().auto()` | +6.2 % |

### Rules of thumb

- **Retail SKU / demand data (counts, intermittency)** → use `LaplaceForecaster::new().auto_aid()` or `SmartForecaster::new()`. The AID classifier picks a matching distribution family (Poisson, Negative-Binomial, LogNormal, Gamma, RectifiedNormal); Croston-style intermittent leaves are enabled automatically.

  ```rust
  use anofox_forecast::models::{Forecaster, SmartForecaster};
  let mut f = SmartForecaster::new();
  f.fit(&series)?;
  let forecast = f.predict(28)?;   // 28-day horizon, non-negative
  ```

- **Economic / financial / continuous non-demand series** → use `LaplaceForecaster::new().auto()` (plain, no AID). On M3 monthly `auto_aid` **regresses ~7 % median MAE vs plain auto** because AID picks distribution families whose Gaussian moment-match doesn't fit smooth continuous data. AutoTheta remains a stronger point-forecast baseline for these panels.

- **Not sure which** → benchmark both on a held-out window of your own data. Each fit+predict is a few milliseconds.

> ⚠ `SmartForecaster` is specifically **demand-focused**. It commits to a single Laplace distribution-family configuration based on AID's classification. On non-demand panels (M3 monthly, M4 daily) it regresses vs. plain `auto()`. Do not use it as a general-purpose selector.

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `TimeSeries` | Main data structure for univariate/multivariate time series |
| `Forecast` | Prediction results with optional confidence intervals |
| `Forecaster` | Trait implemented by all forecasting models (`exog_coefficients()` for OLS inspection) |
| `Pipeline` | Composable transform → model chain, itself implements `Forecaster` |
| `FeatureGenerator` | Deterministic regressor generation (Fourier, DOW, MOY, quarter, holidays) |
| `AccuracyMetrics` | Model evaluation metrics (MAE, MSE, RMSE, MAPE, etc.) |

### Forecasting Models

| Family | Models |
|--------|--------|
| **Auto Selection** | `AutoForecast`, `AutoEnsemble` |
| **ARIMA** | `ARIMA`, `SARIMA`, `AutoARIMA` |
| **Exponential Smoothing** | `SES`, `Holt`, `HoltWinters`, `SeasonalES`, `ETS`, `AutoETS` (with `ModelPool`) |
| **Theta** | `Theta`, `OptimizedTheta`, `DynamicTheta`, `AutoTheta` |
| **Baseline** | `Naive`, `Mean`, `SeasonalNaive`, `RandomWalkWithDrift`, `SMA`, `WindowAverage`, `SeasonalWindowAverage` |
| **Intermittent** | `Croston`, `TSB`, `ADIDA`, `IMAPA` |
| **Complex Seasonality** | `TBATS`, `AutoTBATS`, `MFLES`, `MSTLForecaster` |
| **Volatility** | `GARCH` |
| **Multivariate** | `VAR` (Vector Autoregression) |
| **State-Space** | `KalmanFilter`, `StateSpaceModel` (local level, local linear trend) |
| **Ensemble** | `Ensemble` (Mean, Median, Weighted MSE, InverseAIC, Stacking, HorizonAdaptive) |
| **Regression** | `RegressionForecaster` (OLS, Ridge, ElasticNet, Quantile, WLS, RLS, Tweedie, Poisson, BLS, Dynamic) |
| **Hierarchical** | `HierarchyTree` (BottomUp, TopDown, MiddleOut, MinTraceOls, MinTraceShrink, MinTraceVariance, MinTraceStruct) |
| **Batch/Global** | `GlobalETS`, `GlobalAutoETS`, `GlobalCroston`, `GlobalTheta`, `batch::auto_ets`, `batch::ets`, `batch::mfles` |

### Utilities

| Function / Type | Description |
|-----------------|-------------|
| `compare_models()` | Compare forecasters on the same data with timing |
| `compare_registry()` | Compare all registered models at once |
| `cross_validate()` | Time series cross-validation (parallel with `parallel` feature) |
| `cross_validate_early_stop()` | CV with convergence-based early stopping |
| `rolling_forecast()` | Walk-forward evaluation with rolling/expanding windows |
| `StreamingCVAggregator` | Online metric aggregation using Welford's algorithm |
| `bootstrap_forecast()` | Bootstrap confidence intervals for any model |
| `diagnose_residuals()` | Unified residual diagnostics (Ljung-Box, DW, Jarque-Bera) |
| `ModelDiagnostics` | Comprehensive diagnostics: Ljung-Box, Jarque-Bera, Breusch-Pagan |
| `variance_inflation_factors` / `condition_number` / `multicollinearity_report` | Pre-fit diagnostic on a regression design matrix — surfaces collinear / redundant columns before Ridge/ElasticNet instability bites |
| `IntermittentDiagnostics` | Syntetos-Boylan demand classification with model recommendations |
| `AidAnalyzer` | Automatic Identification of Demand: distribution fitting, anomaly detection |
| `rmsse()` / `wrmsse()` | Root Mean Squared Scaled Error and Weighted RMSSE (M5 competition metric) |
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
| `BinnedConformalPredictor` | Heteroscedastic prediction intervals binned by predicted magnitude |
| `RegressionForecaster` | Multi-backend regression: OLS, Ridge, ElasticNet, Quantile, WLS, RLS, Tweedie, Poisson, BLS, Dynamic |
| `RegressionBackend` | Backend selection enum with convenience constructors (`ridge()`, `quantile()`, `wls_decay()`, `wls_logistic()`, etc.) |
| `WeightStrategy` | WLS weight shape: `Equal`, `ExponentialDecay(decay)`, `Custom(vec)`, or `Logistic { offset }` (sigmoid recency weights centred `offset` steps from the end) |
| `RegressionFeatures` | Feature builder for regression models (trend, seasonal, lags, structural, recursive, exog) |
| `FeatureSafety` | Feature leakage classification: Deterministic, DataDependent, Structural, External |
| `StructuralFeature` | Trait for forward-filled features during prediction (changepoints, outlier indicators) |
| `ChangepointFeature` | Structural feature for regime indicators (StepFunctions, RegimeIndex, CumulativeCount) |
| `RecursiveFeature` | Trait for features recomputed at every horizon step from the rolling history buffer |
| `RollingFeature` / `RollingStatKind` | Rolling window statistics: Mean, Std, Var, Min, Max, Median, Sum, EwmMean, EwmStd, Quantile, Range, Iqr, Skew, Kurt, Slope, Rank, ZScore, CountAbove, CountBelow |
| `EventDistanceFeature` / `EventDistanceMode` | Steps-since-last / steps-until-next event (holidays, promos) — `RecursiveFeature` keyed on absolute timestep |
| `ExogFeatureSpec` | Lag, rolling, polynomial, interaction, and categorical transforms of exog columns; `RegressionFeatures::with_exog_lags()` / `with_exog_rolling()` / `with_exog_polynomial()` / `with_exog_interaction()` / `with_categorical()` |
| `CategoricalStrategy` | `OneHot { drop_first }`, `Ordinal`, `Count`, `Target { smoothing }` — encoding for integer-coded exog with deterministic-shape predeclared categories |
| `Pipeline` / `PipelineBuilder` | Composable transform → model chains (BoxCox → Difference → Model → inverse) |
| `Transform` trait | Reversible transforms: `DifferenceTransform`, `SeasonalDifferenceTransform`, `BoxCoxTransform`, `YeoJohnsonTransform`, `ScaleTransform`, `LogTransform` |
| `FeatureGenerator` | Deterministic feature generation: `fourier()`, `day_of_week()`, `month_of_year()`, `quarter()`, `holiday()` |
| `OLSResult` / `exog_coefficients()` | Inspect OLS pre-regression coefficients (intercept, betas, regressor names) |
| `deseasonalize()` / `seasonal_adjust()` | Remove seasonal component from data or TimeSeries |
| `select_features()` | Automated feature selection (variance, correlation, top-K) |
| `to_json()` / `from_json()` | Serialization for models, `Forecast`, and `TimeSeries` (requires `serde` feature) |
| `to_bincode()` / `from_bincode()` | Binary serialization (requires `serde` feature) |

### Forecastability Analysis (requires `forecastability` feature)

| Function / Type | Description |
|-----------------|-------------|
| `ForecastabilityFingerprint::compute()` | One-call summary: `information_mass`, `information_horizon`, `information_structure`, `nonlinear_share`, `signal_to_noise`, `directness_ratio` |
| `ami_curve(series, max_lag)` | Average Mutual Information at each lag (kNN MI, Kraskov KSG1) |
| `gcmi_curve(series, max_lag)` | Gaussian Copula MI at each lag — captures linear dependence only |
| `pami_curve(series, max_lag, backend)` | Partial AMI — conditions out intermediate lags via residualization |
| `transfer_entropy_curve(source, target, max_lag)` | Directional information flow: TE(X→Y) as conditional MI |
| `knn_mutual_information(x, y, k)` | KSG1 MI estimator with 2D KD-tree (O(n log n)) |
| `gcmi(x, y)` | Gaussian Copula MI: rank → probit → `-0.5 log₂(1-ρ²)` |
| `distance_correlation(x, y)` | Szekely/Rizzo (2007) — detects nonlinear dependence |
| `phase_surrogates(series, n, seed)` | FFT phase randomization for significance testing |
| `significance_bands(series, metric_fn, ...)` | Surrogate-based significance bands for any lag-curve metric |
| `largest_lyapunov_exponent(series, m, tau, ...)` | Rosenstein (1993) — detect chaos via delay-embedding divergence |
| `score(series, Scorer::*)` | 10-scorer registry: Mi, Pearson, Spearman, Kendall, Distance, TE, Gcmi, PermutationEntropy, SpectralEntropy, SpectralPredictability |
| `pearson_curve` / `spearman_curve` / `kendall_curve` | Lag correlation curves (Kendall uses O(n log n) merge-sort) |
| `ar1_theoretical_ami(phi, max_lag)` | Exact AR(1) AMI formula for validation |

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
| Selection | `select_features`, `rank_features`, `select_features_mi`, `rank_features_mi` (MI-based, requires `forecastability`) |
| Cross-series (panel) | `panel_aggregate`, `panel_mean`, `panel_median`, `panel_std`, `panel_rank` — per-timestamp aggregates across N sibling series with optional leave-one-out leakage guard |

### Postprocessing Types

| Type | Description |
|------|-------------|
| `PostProcessor` | Unified API for all postprocessing methods |
| `ConformalPredictor` | Distribution-free prediction intervals |
| `BinnedConformalPredictor` | Heteroscedastic intervals — bins by predicted magnitude |
| `CqrPredictor` / `CqrResult` | Conformalized Quantile Regression — wraps quantile base learners |
| `EnbPiPredictor` / `EnbPiResult` | Ensemble Bootstrap Prediction Interval — bagged LOO residuals with online window |
| `AciPredictor` | Adaptive Conformal Inference — streaming α-adaptation under drift |
| `HistoricalSimulator` | Empirical error distribution |
| `IDRPredictor` | Isotonic Distributional Regression |
| `QRAPredictor` | Quantile Regression Averaging |

## Examples

48 runnable examples covering all major features, each with a companion `.md` description. See [examples/README.md](examples/README.md) for the full categorized index.

```bash
cargo run --example quickstart              # End-to-end forecasting
cargo run --example arima                   # ARIMA family
cargo run --example regression              # 11 regression backends
cargo run --example cross_validation        # Time series CV
cargo run --example postprocess_conformal   # Conformal prediction intervals
```

## Guides

- [Model Selection Guide](docs/model_selection_guide.md) — Which model to use for your data
- [M5 ETS Benchmark](docs/m5_ets_benchmark.md) — AutoETS Complete vs Reduced pool on 30,490 M5 series

## Dependencies

- [chrono](https://crates.io/crates/chrono) - Date and time handling
- [trueno](https://crates.io/crates/trueno) - Linear algebra operations
- [anofox-statistics](https://crates.io/crates/anofox-statistics) - Statistical hypothesis tests (DM, MCS, SPA)
- [statrs](https://crates.io/crates/statrs) - Statistical distributions and functions
- [thiserror](https://crates.io/crates/thiserror) - Error handling
- [rand](https://crates.io/crates/rand) - Random number generation
- [rustfft](https://crates.io/crates/rustfft) - Fast Fourier Transform for spectral analysis

## Acknowledgments

The postprocessing module is a Rust port of [PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl). The sequential monitoring module (`monitor::`) is a Rust port of [changepoint.forecast](https://github.com/grundy95/changepoint.forecast) by Thomas Grundy (Lancaster University), based on [Fremdt (2014)](https://doi.org/10.1080/02331888.2014.921899). The trait-based changepoint surface (`changepoint::Detector` + `Cost`) mirrors [`ruptures`](https://github.com/deepcharles/ruptures) by [Charles Truong / Laurent Oudre / Nicolas Vayatis](https://centre-borelli.github.io/ruptures-docs/) and is validated against `ruptures==1.1.9` via the parity fixtures in `tests/data/ruptures_fixtures/`. Feature extraction is inspired by [tsfresh](https://github.com/blue-yonder/tsfresh). Forecasting models are validated against [StatsForecast](https://github.com/Nixtla/statsforecast) by Nixtla. See [THIRDPARTY_NOTICE.md](THIRDPARTY_NOTICE.md) for full attribution and references to the research papers that inspired this implementation.

## License

MIT License - see [LICENSE](LICENSE) for details.
