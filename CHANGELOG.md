# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Parallel Batch Processing & AutoForecast**
  - `fit_predict_many()` — fit one model across many series in parallel
  - `fit_registry()` — fit all registered models on a series in parallel
  - `compare_models()` / `compare_registry()` — parallel model comparison
  - `AutoForecast` candidate model fits run in parallel when `parallel` feature enabled
  - Bootstrap sampling uses `par_iter` when `parallel` enabled

- **Streaming Cross-Validation**
  - `StreamingCVAggregator` — online metric aggregation using Welford's algorithm
  - `cross_validate_early_stop()` — CV with convergence-based early stopping
  - Eliminates need to store all fold results for aggregation

- **Builder Patterns**
  - `Pelt::new(CostFunction::L2).min_size(5).penalty(5.0).detect(&data)` — PELT builder
  - `StlBuilder::new(period).seasonal_window(7).robust(true).decompose(&data)` — STL builder

- **SIMD Correlation & Autocorrelation**
  - `simd::correlation()` — SIMD-accelerated Pearson correlation
  - `simd::autocorrelation()` — SIMD-accelerated autocorrelation at a given lag

- **Binary Serialization** (requires `serde` feature)
  - `to_bincode()` / `from_bincode()` — compact binary serialization via bincode
  - `save_to_bincode()` / `load_from_bincode()` — file persistence

- **Convenience Methods**
  - `Forecaster::fit_predict()` — fit and predict in a single call
  - `Forecaster::fit_predict_with_intervals()` — fit and predict with confidence intervals

- **Specific Error Variants**
  - `ForecastError::ConvergenceFailure` — optimizer/model convergence failures
  - `ForecastError::SingularMatrix` — linear algebra singularity errors
  - `ForecastError::SerializationError` — serialization/deserialization errors

- **WASM/npm Enhancements**
  - `CalendarAnnotations` — holidays, named regressors, and JSON serialization in JS/TS
  - `TimeSeries.setCalendar()` / `hasCalendar()` / `clearCalendar()` — calendar integration
  - Complete TypeScript type definitions (`types.d.ts`) for all 35 exported classes and 21 functions
  - `package.json` with `"types"` field for TypeScript support

- **CI/CD Improvements**
  - `cargo audit` step for security advisory checks
  - `cargo deny` step for license and supply-chain compliance
  - `deny.toml` configuration for allowed licenses

- **Benchmarks**
  - `ensemble_benchmark` — AutoEnsemble fit/predict benchmarks
  - `cv_benchmark` — cross-validation benchmarks with varying folds/horizons/models

- **Documentation**
  - [Model Selection Guide](docs/model_selection_guide.md) — decision flowchart, model families, common patterns

- **Test Coverage**
  - Postprocess tests: +42 IDR tests, +22 Normal tests, +316 lines backtest, +211 lines QRA
  - Streaming CV tests: 8 new tests for aggregator and early stopping
  - SIMD tests: 18 new tests for correlation and autocorrelation

- **Missing Value Imputation Toolbox**
  - `MissingValuePolicy::BackwardFill` — next-observation-carried-backward
  - `MissingValuePolicy::FillMean` — fill with mean of finite values
  - `MissingValuePolicy::FillMedian` — fill with median of finite values
  - `MissingValuePolicy::Interpolate` — linear interpolation via policy enum
  - `TimeSeries::missing_mask()` — boolean mask of NaN/Inf positions (primary dimension)
  - `TimeSeries::missing_count()` — per-dimension count of missing values
  - `TimeSeries::imputed_forward_backward()` — forward-fill then backward-fill (handles leading + trailing NaN)
  - `TimeSeries::imputed_moving_average(window)` — centered moving average imputation with multi-pass for adjacent gaps
  - `TimeSeries::imputed_seasonal(period)` — seasonal median imputation using same-position values across cycles
  - `TimeSeries::with_imputed_regressors(policy)` — apply imputation policy to regressor vectors independently
  - `nan_mean()` / `nan_median()` — NaN-safe statistics helpers in `utils::stats`

- **OLS NaN/Inf Validation**
  - `ols_fit()` now validates `y` for NaN/Inf (returns `ForecastError::MissingValues`)
  - `ols_fit()` now validates each regressor for NaN/Inf (returns `ForecastError::InvalidParameter`)

- **Hierarchical Forecasting** (`hierarchy` module)
  - `HierarchyTree` — define parent→children structure for grouped forecasts
  - `ReconciliationMethod::BottomUp` — aggregate leaf forecasts upward
  - `ReconciliationMethod::TopDown` — disaggregate top-level using historical proportions
  - `ReconciliationMethod::MinTraceOls` — optimal combination via MinT OLS projection
  - 15 tests covering tree construction, all methods, coherence, and error handling

- **Prophet-style Fourier Seasonality** (`seasonality::fourier` module)
  - `FourierSeasonality` — flexible seasonal modeling using Fourier basis functions
  - `fourier_terms()` — generate sin/cos basis vectors for arbitrary period and order
  - Preset constructors: `daily()`, `weekly()`, `yearly()`
  - Fit via normal equations with Cholesky decomposition (no external dependencies)
  - 25 tests covering recovery, orthogonality, periodicity, edge cases

- **Core Type Improvements**
  - `Serialize`/`Deserialize` for `Forecast`, `TimeSeries`, `CalendarAnnotations` (behind `serde` feature)
  - `Display` impls for `Forecast` and `TimeSeries` with preview summaries
  - `PartialEq` for `Forecast` (epsilon-based) and `CalendarAnnotations`

- **WASM Model Parity**
  - Added JS/WASM bindings: `HoltWintersForecaster`, `SESForecaster`, `CrostonForecaster`, `ADIDAForecaster`, `IMAPAForecaster`, `TSBForecaster`, `GARCHForecaster`
  - TypeScript definitions for all new forecaster classes

- **Validation & Stationarity Tests**
  - 14 new residual diagnostic tests (edge cases, NaN, constant, short series)
  - 9 new stationarity test cases (ADF, KPSS, edge cases)

- **Intermittent Model & GARCH Edge Case Tests**
  - Croston: 6 new tests (all zeros, single demand, negative values, very long gaps, single observation)
  - ADIDA: 6 new tests (same pattern)
  - IMAPA: 6 new tests (same pattern)
  - TSB: 5 new tests (same pattern)
  - GARCH: 10 new tests (constant data, extreme volatility, trending data, NaN handling, short series)

- **VAR (Vector Autoregression)** (`models::var` module)
  - `VAR::new(order)` — VAR(p) model for multivariate time series
  - Equation-by-equation OLS estimation
  - Multi-step forecasting across all variables
  - `granger_causality_test(cause, effect)` — F-statistic for Granger causality
  - 18 tests covering coefficient recovery, dimensions, edge cases

- **Kalman Filter Framework** (`models::kalman` module)
  - `KalmanFilter` — forward filtering and Rauch-Tung-Striebel smoothing
  - `StateSpaceModel` — linear Gaussian state-space specification
  - Convenience constructors: `local_level()`, `local_linear_trend()`
  - `filter()`, `smooth()`, `predict()`, `log_likelihood()` methods
  - Internal dense matrix algebra (no external dependencies)
  - 14 tests covering filtering, smoothing, prediction, edge cases

- **Builder Patterns for Models**
  - `GARCH::builder().p(1).q(1).max_iterations(500).build()`
  - `MFLES::builder().seasonal_period(12).num_rounds(5).learning_rate(0.1).build()`
  - `AutoForecast::builder().seasonal_period(12).include_arima(true).build()`

- **Rolling/Expanding Window Forecast**
  - `rolling_forecast()` — walk-forward evaluation with per-window metrics
  - `RollingForecastConfig` — builder for initial train size, horizon, step size, expanding/rolling mode
  - `RollingForecastResult` — per-window predictions, actuals, and aggregated metrics
  - Parallel window evaluation when `parallel` feature enabled

- **Ensemble Prediction Interval Combination**
  - Widest-envelope interval combination: takes min of lower bounds, max of upper bounds
  - `predict_with_intervals()` now produces meaningful combined intervals

- **STL Buffer Caching**
  - `StlScratch` — pre-allocated scratch buffers for zero-allocation repeated decompositions
  - `STL::decompose_with_scratch()` — decompose with reusable buffers
  - `StlBuilder::decompose_reuse()` — amortized allocation across repeated calls

- **TimeSeries Convenience Methods**
  - `seasonal_strength(period)` — seasonal strength via STL decomposition (0 to 1)
  - `trend_strength(period)` — trend strength via STL decomposition (0 to 1)
  - `with_outliers_replaced(config, window)` — replace outliers with local median
  - `to_json()` / `from_json()` — JSON serialization (requires `serde` feature)
  - `Forecast::to_json()` / `Forecast::from_json()` — forecast serialization

- **WASM PostProcessor Bindings**
  - `JsConformalPredictor`, `JsNormalPredictor`, `JsHistoricalSimulator` — probabilistic intervals in JS
  - `JsPostProcessor` — unified API with `conformal()`, `normal()`, `historicalSim()`
  - `JsBacktestConfig` / `JsBacktestResult` — backtesting support
  - `JsPredictionIntervals` — coverage, widths, midpoints, empirical coverage
  - `MFLESForecaster.predictWithIntervals()` — added to WASM bindings
  - TypeScript definitions for all postprocess types

- **Persistence Module Tests**
  - 27 tests (from 6): JSON/bincode round-trips, file I/O, error cases, helper modules

### Changed

- `parallel` feature now covers AutoForecast, batch processing, model comparison, bootstrap, cross-validation folds, and rolling forecast windows (previously only AutoARIMA)
- `serde` feature now includes bincode for binary serialization alongside JSON
- Several `ComputationError` uses migrated to specific error variants (`ConvergenceFailure`, `SingularMatrix`)
- Removed dead code in `mfles.rs` and `tbats/model.rs`
- MSTL uses in-place decomposition for reduced allocations
- STL decomposition supports buffer reuse via `StlScratch`
- ETS uses `Cow<[f64]>` to avoid cloning series values when no regressors are present
- Cross-validation uses direct slice references to avoid intermediate allocations per fold
- Ensemble `predict_with_intervals()` produces widest-envelope combined intervals
- Test coverage increased to 1,970+ tests
- `MissingValuePolicy` enum has 4 new variants (breaking for exhaustive `match` — acceptable under 0.x semver)

## [0.4.1] - 2026-01-16

### Added

- **ETS Notation Parser & FPP3 Taxonomy Compliance**
  - `ETSSpec::from_notation("AAA")` - Create ETS models from standard notation
  - `ETSSpec::is_valid()` - Validate model combinations before fitting
  - Reject unstable ETS combinations (MAA, MAdA) per [FPP3 taxonomy](https://otexts.com/fpp3/taxonomy.html)
  - New convenience constructors: `ana()`, `anm()`, `aada()`, `aadm()`, `mnm()`, `madm()`

- **WASM/npm Package Enhancements**
  - `ETSForecaster.fromNotation("AAA", period)` - Standard notation in JavaScript
  - `ETSForecaster.isValidSpec(error, trend, seasonal)` - Validation helper
  - Constructor validation rejects unstable ETS combinations
  - npm package documentation with ETS notation examples

- **Comprehensive Test Coverage**
  - 76 WASM tests covering all 29 forecaster classes
  - 23 JavaScript integration tests (Node.js)
  - Edge case tests: NaN handling, single data point, negative values, large/small values
  - ETS notation parsing tests (valid, invalid, unstable combinations)

- **CI/CD Improvements**
  - JavaScript integration tests in CI workflow
  - npm OIDC trusted publishing (no tokens required)
  - Requires npm >= 11.5.1 for OIDC support

- **Documentation**
  - "Use Cases" section in README with DuckDB extension and npm package links
  - Updated npm README with ETS notation API documentation
  - FPP3 taxonomy reference in API docs

### Changed

- Test coverage increased to 1,400+ tests (unit + integration + WASM + JS)
- Installation instructions updated to v0.4

## [0.4.0] - 2026-01-12

### Added

- **Probabilistic Forecasting Module** (`postprocess` feature, enabled by default)
  - `PostProcessor` - Unified API for probabilistic forecast calibration
  - `PredictionIntervals` - Multi-level interval representation with coverage guarantees
  - **Conformal Prediction** - Distribution-free prediction intervals
  - **Conformalized Quantile Regression** - Calibrated quantile forecasts
  - **Quantile Regression Averaging (QRA)** - Ensemble-based probabilistic forecasts
  - **Historical Simulation** - Bootstrap-based uncertainty estimation
  - **Normal Approximation** - Parametric prediction intervals
  - **Isotonic Distributional Regression (IDR)** - Non-parametric calibration
  - **Backtesting** - Horizon-aware backtesting with automatic calibration
    - `BacktestConfig` with expanding/rolling windows
    - Per-horizon calibration for improved accuracy
    - Coverage and calibration error metrics

- **Cross-Validation Enhancements**
  - `CvFoldGenerator` - Standalone fold generation for custom workflows
  - `gap` parameter - Prevents data leakage from lagged features
  - `purge` parameter - Removes observations to prevent lookahead bias (financial applications)
  - `FillStrategy` trait - Handle unknown future features during CV
    - Implementations: `LastValueFill`, `MeanFill`, `MedianFill`, `ZeroFill`, `ConstantFill`, `ModeFill`
  - `train_test_split()` - Simple ratio or index-based splitting
  - `train_test_split_at()` - Split at specific index
  - `grouped_cross_validate()` - Multi-series CV with consistent fold boundaries
  - `GroupedCVResults` - Per-group and aggregated metrics
  - `Fold` struct - Explicit train/test index representation

- **Architecture Documentation**
  - ADR explaining CV split design decisions
  - Documents why CV split lives in DuckDB extension vs this crate
  - Component distribution rationale (fold generation, orchestration, etc.)

- **New Examples**
  - `postprocess/quickstart.rs` - Getting started with probabilistic forecasts
  - `postprocess/conformal.rs` - Conformal prediction intervals
  - `postprocess/conformalize.rs` - Conformalized quantile regression
  - `postprocess/qra_ensemble.rs` - QRA ensemble methods
  - `postprocess/quantile_methods.rs` - Various quantile approaches
  - `postprocess/unified_api.rs` - PostProcessor unified API
  - `postprocess/backtest.rs` - Backtesting workflows

### Changed

- Cross-validation now returns `folds` field in `CVResults` for transparency
- Test coverage increased to 1,316+ tests

### Dependencies

- Added `faer` (optional) for linear algebra in postprocessing
- Added `anofox-regression` v0.5.0 (optional) for quantile regression

## [0.3.2] - 2026-01-07

### Fixed

- **Stable Rust Compatibility**
  - Replaced unstable `is_multiple_of()` method with `% 2 == 0`
  - Fixes WASM builds on stable Rust 1.86.0+

## [0.3.1] - 2026-01-07

### Added

- **WASM Target Support**
  - Compilation for `wasm32-unknown-unknown` target
  - `js` feature flag for browser environments (enables getrandom/js)
  - Compile-time guard preventing `parallel` feature on WASM targets

## [0.3.0] - 2026-01-03

### Added

- **Optional Parallel AutoARIMA**
  - Feature-gated Rayon parallelization for model evaluation
  - Enable with `--features parallel` for 4-8x speedup
  - Default: sequential execution (DuckDB compatible)
- **Bootstrap Confidence Intervals**
  - `BootstrapConfig` for configuring bootstrap parameters
  - `bootstrap_intervals()` for empirical confidence intervals
  - `bootstrap_forecast()` convenience function
  - Residual bootstrap and block bootstrap methods
- **True Stepwise Search for AutoARIMA**
  - Neighbor-based hill climbing algorithm
  - Reduces model evaluations by 60-70%
  - Enable with `AutoARIMAConfig::with_true_stepwise(true)`
- **Property-Based Testing**
  - 20+ proptest cases for model invariants
  - Tests forecast length, finite values, interval ordering
  - Tests fitted values + residuals reconstruction
- **Interval Calibration Testing**
  - Rolling origin cross-validation for coverage rate testing
  - Winkler score for interval quality assessment
  - Coverage rate tests for analytical and bootstrap intervals

### Removed

- **Time Series Clustering** - Removed clustering module (not needed)

### Changed

- Improved test coverage with 1,136+ tests total
- Updated dependencies

## [0.2.0] - 2025-12-17

### Added

- **Periodicity Detection Module**
  - `ACFPeriodicityDetector` - time-domain detection using ACF peaks
  - `FFTPeriodicityDetector` - frequency-domain detection using periodogram
  - `Autoperiod` - hybrid FFT+ACF detector (Vlachos et al. 2005)
  - `CFDAutoperiod` - noise-resistant detector with clustering (Puech et al. 2020)
  - `SAZED` - parameter-free ensemble method (Toller et al. 2019)
  - Convenience functions: `detect_period()`, `detect_period_ensemble()`, `detect_period_range()`
  - `PeriodicityDetector` trait for unified API
- **FFT Utilities**
  - `fft_real()` - FFT for real-valued signals
  - `periodogram()` - power spectral density computation
  - `periodogram_peaks()` - significant peak detection
  - `welch_periodogram()` - Welch's method for reduced variance
- **SIMD-Accelerated Operations**
  - Vector sum, mean, variance, standard deviation
  - Dot product and sum of squares
  - Squared Euclidean and Manhattan distances
  - Element-wise operations (add, subtract, multiply, divide, scale)
  - Uses Trueno for AVX2/SSE2/NEON acceleration
- **Validation Tools**
  - CLI tool for periodicity detection (`examples/analysis/detect_period.rs`)
  - Python cross-validation script against pyriodicity
  - Criterion benchmarks for periodicity detection

### Changed

- Updated documentation with periodicity detection examples
- Added `rustfft` dependency for FFT operations

## [0.1.0] - 2025-12-11

### Added

- Initial release of anofox-forecast
- **Core Data Structures**
  - `TimeSeries` for univariate and multivariate time series data
  - `Forecast` for prediction results with confidence intervals
  - `CalendarAnnotations` for holidays and regressors
- **Forecasting Models (35+)**
  - ARIMA and AutoARIMA with automatic order selection
  - Exponential Smoothing: SES, Holt's Linear, Holt-Winters, ETS, AutoETS
  - Baseline methods: Naive, Seasonal Naive, Random Walk with Drift, SMA
  - Theta method
  - Intermittent demand: Croston, ADIDA, TSB
  - Ensemble methods with multiple combination strategies
- **Feature Extraction (76+ features)**
  - Basic statistics (mean, variance, quantiles, etc.)
  - Distribution features (skewness, kurtosis, etc.)
  - Autocorrelation and partial autocorrelation
  - Entropy features (approximate, sample, permutation, binned)
  - Complexity features (C3, CID, Lempel-Ziv)
  - Trend analysis and stationarity tests
- **Seasonality & Decomposition**
  - STL (Seasonal-Trend decomposition using LOESS)
  - MSTL (Multiple Seasonal-Trend decomposition)
- **Changepoint Detection**
  - PELT algorithm with L1, L2, Normal, and Poisson cost functions
- **Anomaly Detection**
  - Statistical methods (IQR, z-score)
  - Automatic threshold selection
- **Time Series Clustering**
  - Dynamic Time Warping (DTW) distance
  - K-Means clustering with multiple distance metrics
- **Data Transformations**
  - Scaling: standardization, min-max, robust scaling
  - Box-Cox transformation with automatic lambda selection
  - Window functions: rolling and expanding statistics, EWM
- **Model Evaluation**
  - Accuracy metrics (MAE, MSE, RMSE, MAPE, etc.)
  - Time series cross-validation
  - Residual testing and stationarity tests
