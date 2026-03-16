# Quickstart

**Run:** `cargo run --example quickstart`

## What this example demonstrates

End-to-end ARIMA workflow: create a time series, fit a model, generate forecasts with confidence intervals, and evaluate accuracy. This is the recommended starting point for new users of anofox-forecast.

## Sections

1. **Create sample data** -- Builds a 100-observation `TimeSeries` with linear trend, sine seasonality, and cosine noise.
2. **Fit ARIMA(1,1,1)** -- Fits the model and prints AR/MA coefficients, intercept, AIC, and BIC.
3. **Point forecasts** -- Generates 10-step-ahead predictions via `predict`.
4. **Confidence intervals** -- Uses `predict_with_intervals` at 95% to get lower/upper bounds alongside point forecasts.
5. **Accuracy metrics** -- Computes in-sample MAE, RMSE, SMAPE, MAPE, and R-squared from fitted values using `calculate_metrics`.
6. **Residual analysis** -- Extracts residuals and reports mean, variance, and standard deviation.

## Key types

- `TimeSeries` -- core time series container
- `ARIMA` -- ARIMA(p,d,q) model
- `Forecaster` -- trait providing `fit`, `predict`, `predict_with_intervals`, `fitted_values`, `residuals`
- `calculate_metrics` -- utility returning MAE, RMSE, SMAPE, MAPE, R-squared
