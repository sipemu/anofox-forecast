# VAR (Vector Autoregression)

**Run:** `cargo run --example var`

## What this example demonstrates

Fits Vector Autoregression models to multivariate time series where each variable depends on its own lags and the lags of other variables. The example generates synthetic bivariate VAR(1) and VAR(2) processes, recovers the true coefficients, runs Granger causality tests, and shows how to use the `VARForecaster` adapter to access the standard `Forecaster` trait with prediction intervals.

## Sections

1. **Generate VAR(1) data** -- Creates 200 observations of a bivariate system with known intercepts and coefficient matrices, plus pseudo-random noise.
2. **Fit VAR(1)** -- Estimates intercepts and the lag-1 coefficient matrix, compares to true values, and reports residual RMSE.
3. **Granger causality test** -- Tests whether each variable Granger-causes the other using F-statistics.
4. **Multi-step forecasting** -- Produces 10-step-ahead point forecasts from the fitted VAR(1).
5. **VAR(2) on higher-order data** -- Generates a VAR(2) process (300 observations), fits a 2-lag model, and compares estimated coefficients to ground truth.
6. **VARForecaster adapter** -- Wraps VAR in the `Forecaster` trait using `VARForecaster`, builds a `TimeSeries` with a regressor, and produces point forecasts with 95% confidence intervals.

## Key types

- `VAR` -- core Vector Autoregression model with `fit`, `predict`, `granger_causality_test`
- `VARForecaster` -- adapter implementing the `Forecaster` trait for VAR
- `TimeSeriesBuilder` / `CalendarAnnotations` -- build a `TimeSeries` with external regressors
- `Forecaster` trait -- `fit`, `predict`, `predict_with_intervals`, `fitted_values`
