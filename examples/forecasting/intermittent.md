# Intermittent Demand Forecasting

**Run:** `cargo run --example intermittent`

## What this example demonstrates

Shows how to forecast sporadic demand patterns where many periods have zero demand. It compares five intermittent demand methods (Croston, Croston SBA, TSB, ADIDA) and classifies the demand pattern using the ADI/CV-squared matrix.

## Sections

1. **Demand pattern generation** -- creates a 40-period series with irregular non-zero demands interspersed with zeros, then prints demand statistics (average size, inter-arrival interval, demand ratio).
2. **Croston's Method (Classic)** -- fits Croston with a fixed alpha, prints demand level, interval level, and flat forecast.
3. **Croston's Method (Optimized)** -- lets the library optimize the smoothing parameter automatically.
4. **Syntetos-Boylan Approximation (SBA)** -- applies Croston with bias correction factor `(1 - alpha/2)` and compares against the classic variant.
5. **TSB (Teunter-Syntetos-Babai)** -- forecasts demand probability directly; sweeps several alpha values to show sensitivity.
6. **ADIDA** -- aggregates the series at different levels before forecasting and disaggregating; sweeps aggregation levels 2-5.
7. **Model comparison table** -- prints all five methods' forecasts side by side.
8. **Confidence intervals** -- generates 95% prediction intervals using Croston.
9. **Demand classification** -- computes ADI and CV-squared, then classifies the pattern as Smooth, Intermittent, Erratic, or Lumpy.

## Key types

- `Croston` -- classic and SBA intermittent demand model
- `TSB` -- Teunter-Syntetos-Babai method with separate demand/probability smoothing
- `ADIDA` -- Aggregate-Disaggregate Intermittent Demand Approach
- `Forecaster` trait -- `fit`, `predict`, `predict_with_intervals`
- `TimeSeries` -- input data container
