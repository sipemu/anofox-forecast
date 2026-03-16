# Postprocessing Quickstart

**Run:** `cargo run --example postprocess_quickstart`

## What this example demonstrates

The simplest path from point forecasts to prediction intervals. It creates a conformal `PostProcessor`, trains it on historical forecast-actual pairs to learn the error distribution, and generates 90% prediction intervals for new forecasts.

## Sections

1. **Generate synthetic data** -- Creates 100 historical forecast-actual pairs with known noise.
2. **Create and train PostProcessor** -- Builds a `PostProcessor::conformal(0.90)` and trains it via `train()` using `PointForecasts`.
3. **Generate prediction intervals** -- Calls `predict_intervals` on 7 future point forecasts and prints a table with lower/upper bounds and interval width.

## Key types

- `PostProcessor` -- unified entry point for postprocessing methods (`conformal`, `historical_sim`, `normal`, `idr`)
- `PointForecasts` -- wrapper for a vector of point forecast values
- `PredictionIntervals` -- output with `lower()` and `upper()` accessors
