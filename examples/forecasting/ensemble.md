# Ensemble Forecasting

**Run:** `cargo run --example ensemble`

## What this example demonstrates

Combines multiple forecasting models into ensembles using different combination strategies (mean, median, weighted MSE, custom weights). Compares individual model forecasts against ensemble forecasts and shows that ensembles typically improve accuracy and robustness.

## Sections

1. **Mean ensemble** -- averages forecasts from Naive, SMA(5), and SES.
2. **Median ensemble** -- takes the median of five models (Naive, SMA, SES, RW+Drift, Holt) to reduce the impact of outlier forecasts.
3. **Weighted MSE ensemble** -- weights four models by the inverse of their in-sample MSE so better-fitting models contribute more.
4. **Custom weighted ensemble** -- assigns user-specified weights (Naive 10%, Holt 30%, Theta 60%) to encode domain knowledge.
5. **Individual vs ensemble comparison** -- prints h=5 forecasts from each individual model alongside the three ensemble methods.
6. **In-sample performance** -- computes MAE, RMSE, and MAPE for individual models and ensembles using fitted values.
7. **Confidence intervals** -- produces 95% prediction intervals from the weighted ensemble over a 10-step horizon.

## Key types

- `Ensemble` -- combines multiple `Forecaster` implementations
- `CombinationMethod` -- `Mean`, `Median`, `WeightedMSE`
- `Forecaster` trait -- shared interface for all models
- `Naive`, `SimpleMovingAverage`, `RandomWalkWithDrift` -- baseline models
- `SimpleExponentialSmoothing`, `HoltLinearTrend` -- exponential smoothing models
- `Theta` -- Theta method forecaster
- `calculate_metrics` -- computes MAE, RMSE, MAPE from actual/predicted vectors
