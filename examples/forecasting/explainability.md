# Forecast Explainability

**Run:** `cargo run --example explainability`

## What this example demonstrates

Decomposes ETS forecasts into their constituent components -- level, trend, and seasonal -- using the `Explainable` trait. For each ETS variant the example verifies that the components sum back to the original point forecast, and checks which components are present or absent depending on the model specification.

## Sections

1. **ETS(A,A,A) decomposition** -- Fits a full additive model (level + trend + seasonal) to 72 observations and prints a table of all components alongside the reconstructed forecast, verifying the reconstruction error is negligible.
2. **ETS(A,A,N) decomposition** -- Fits a trend-only model (no seasonality) to 50 observations and confirms the seasonal component is absent.
3. **ETS(A,N,N) decomposition** -- Fits a simple exponential smoothing model (level only) and confirms both trend and seasonal components are absent.
4. **Length validation** -- Uses `has_correct_lengths()` to verify that all component vectors match the requested horizon.
5. **Named components** -- Inspects the `named_components` map for any non-standard decomposition entries.

## Key types

- `Explainable` trait -- `explain(horizon)` returns a `ForecastExplanation`
- `ForecastExplanation` -- struct with `level`, `trend`, `seasonal`, `residual`, and `named_components` fields; `sum()` reconstructs the point forecast
- `ETSSpec` -- model specification (`aaa()`, `aan()`, `ann()`)
- `ETS` -- exponential smoothing model implementing both `Forecaster` and `Explainable`
