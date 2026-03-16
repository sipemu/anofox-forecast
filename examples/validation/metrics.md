# Accuracy Metrics

**Run:** `cargo run --example metrics`

## What this example demonstrates

How to calculate and interpret forecast accuracy metrics using individual functions and the comprehensive `calculate_metrics` helper. It covers metric comparison across multiple models, seasonal MASE calculation, and edge cases like perfect predictions and data containing zeros.

## Sections

1. **Comprehensive metrics** -- Uses `calculate_metrics` to compute MAE, MSE, RMSE, SMAPE, R-squared, MAPE, and MASE in one call.
2. **Individual metric functions** -- Calls `mae`, `mse`, `rmse`, and `smape` directly on actual/predicted vectors.
3. **MASE with seasonal period** -- Passes a seasonal period (4) to `calculate_metrics` so MASE is computed against a seasonal naive baseline, with interpretation of the result.
4. **Comparing multiple models** -- Runs three models of varying quality through `calculate_metrics` and prints a comparison table.
5. **Edge cases** -- Shows behavior for perfect predictions (all metrics zero/one) and data containing zeros (MAPE undefined, SMAPE still works).
6. **Metric interpretation guide** -- Prints a reference summary explaining when to use each metric.

## Key types

- `calculate_metrics` -- computes all metrics at once, returns a struct with `mae`, `mse`, `rmse`, `smape`, `r_squared`, `mape`, `mase`
- `mae`, `mse`, `rmse`, `smape` -- standalone metric functions
