# Forecast Export

**Run:** `cargo run --example forecast_export`

## What this example demonstrates

Reads synthetic time series data from CSV files, runs 27 different forecasting models (baseline, exponential smoothing, ARIMA, Theta, intermittent demand, MSTL, MFLES, TBATS, GARCH, and more), and exports point forecasts and confidence intervals to CSV for comparison with the Python statsforecast package.

## Sections

1. **CSV reading** -- Parses timestamp/value CSV files from `validation/data/` for 25 series types (stationary, trend, seasonal, intermittent, etc.).
2. **Model execution** -- Fits each model via the `Forecaster` trait, generates 12-step-ahead point forecasts, and optionally produces confidence intervals at 80%, 90%, and 95% levels.
3. **Result export** -- Writes `point_forecasts.csv` and `confidence_intervals.csv` to `validation/results/rust/`.

## Key types

- `Forecaster` trait -- unified `fit` / `predict` / `predict_with_intervals` interface
- `TimeSeries` -- input data container
- Baseline models: `Naive`, `SeasonalNaive`, `RandomWalkWithDrift`, `HistoricAverage`, `WindowAverage`, `SeasonalWindowAverage`
- Exponential smoothing: `SimpleExponentialSmoothing`, `HoltWinters`, `SeasonalES`, `ETS`, `AutoETS`
- ARIMA family: `ARIMA`, `SARIMA`, `AutoARIMA`
- Theta family: `Theta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta`, `AutoTheta`
- Intermittent: `Croston`, `TSB`, `ADIDA`, `IMAPA`
- Decomposition: `MSTLForecaster`, `MFLES`, `TBATS`, `AutoTBATS`
- Volatility: `GARCH`
