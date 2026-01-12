# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
