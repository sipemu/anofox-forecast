# Quantile Methods Comparison

**Run:** `cargo run --example postprocess_quantile_methods`

## What this example demonstrates

Compares three approaches for generating quantile forecasts from point forecasts: empirical historical simulation, Gaussian normal assumption, and isotonic distributional regression (IDR). It highlights how distributional assumptions affect quantile estimates and interval widths.

## Sections

1. **Historical Simulator** -- Fits `HistoricalSimulator` on past errors and produces quantiles from the empirical error distribution.
2. **Normal Predictor** -- Fits `NormalPredictor` assuming Gaussian errors, printing the learned standard deviation.
3. **IDR Predictor** -- Fits `IDRPredictor` which learns a monotone forecast-to-quantile mapping without distributional assumptions.
4. **Median comparison** -- Compares the q=0.5 predictions across all three methods.
5. **Interval width comparison** -- Compares the 80% interval width (q0.1 to q0.9) for each method.

## Key types

- `HistoricalSimulator` -- non-parametric quantile forecasts from empirical errors
- `NormalPredictor` -- parametric quantiles assuming Gaussian errors
- `IDRPredictor` -- isotonic distributional regression for adaptive, forecast-dependent quantiles
- `PointForecasts` -- input wrapper
- `QuantileForecasts` -- output with `at_time(i)` accessor returning quantile values per time step
