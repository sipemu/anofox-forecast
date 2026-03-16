# Baseline Models

**Run:** `cargo run --example baseline`

## What this example demonstrates

Shows all baseline forecasting methods: Naive, Random Walk with Drift, Seasonal Naive, Simple Moving Average, and Seasonal Window Average. Includes in-sample performance comparison and residual analysis, plus guidance on when to use each method.

## Sections

1. **Naive forecast** -- Repeats the last observed value; shows forecasts with 95% confidence intervals.
2. **Random Walk with Drift** -- Adds estimated drift (average change) per step; prints drift value and forecasts.
3. **Seasonal Naive** -- Repeats values from the same season one period back (period=12); shows 12-step forecast alongside source values.
4. **Simple Moving Average** -- Compares SMA with windows of 3, 6, and 12; shows the arithmetic behind SMA(3).
5. **Seasonal Window Average** -- Averages the same season across the last 3 periods (period=12, windows=3).
6. **In-sample comparison** -- Fits all models and compares MAE, RMSE, and R-squared via `calculate_metrics`.
7. **Residual analysis** -- Reports residual mean, variance, and standard deviation for Random Walk with Drift.
8. **When to use each method** -- Text summary of appropriate use cases for each baseline.

## Key types

- `Naive` -- last-value forecast
- `RandomWalkWithDrift` -- naive plus estimated drift
- `SeasonalNaive` -- seasonal last-value forecast
- `SimpleMovingAverage` -- rolling mean forecast
- `SeasonalWindowAverage` -- multi-period seasonal average
- `Forecaster` -- trait for `fit`, `predict`, `predict_with_intervals`, `fitted_values`, `residuals`
- `calculate_metrics` -- accuracy metric computation
