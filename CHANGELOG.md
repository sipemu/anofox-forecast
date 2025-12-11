# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
