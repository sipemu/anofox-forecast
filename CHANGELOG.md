# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
