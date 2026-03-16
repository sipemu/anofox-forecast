# Kalman Filter

**Run:** `cargo run --example kalman`

## What this example demonstrates

Applies the Kalman filter to state-space models of increasing complexity -- from a constant local level to a local linear trend. The example covers forward filtering, RTS smoothing, multi-step prediction, log-likelihood evaluation, and the `KalmanForecaster` adapter for the `Forecaster` trait with prediction intervals.

## Sections

1. **Local level model** -- Filters 100 noisy observations of a constant level, showing state convergence and covariance reduction over time.
2. **Local linear trend model** -- Filters 200 observations of a linearly trending signal and recovers the level and slope.
3. **Forward filtering diagnostics** -- Examines innovation statistics (mean, variance) and compares predicted vs actual observations.
4. **RTS smoothing** -- Runs Rauch-Tung-Striebel smoothing and compares smoothed states/covariances to filtered estimates, confirming the smoother always reduces uncertainty.
5. **Multi-step prediction** -- Produces 10-step-ahead forecasts from both the local level and local linear trend models.
6. **KalmanForecaster adapter** -- Uses `KalmanForecaster::local_level()`, `local_linear_trend()`, and `with_model()` constructors to access the `Forecaster` trait, including prediction intervals, fitted values, and residuals.

## Key types

- `StateSpaceModel` -- defines F, H, Q, R matrices; constructors `local_level()` and `local_linear_trend()`
- `KalmanFilter` -- forward `filter`, `smooth` (RTS), `predict`, and `log_likelihood`
- `KalmanForecaster` -- adapter implementing the `Forecaster` trait for Kalman filter models
- `TimeSeries` -- univariate time series input
