# Batch Forecasting

**Run:** `cargo run --example batch` or `cargo run --example batch --features parallel`

## What this example demonstrates

Fits multiple models to multiple time series in batch, optionally using Rayon parallelism. It registers models in a `ModelRegistry`, runs cross-series comparisons, and picks the best model per series by in-sample RMSE.

## Sections

1. **Create multiple time series** -- generates four synthetic series (trending, seasonal, trend+seasonal, flat) with a helper function.
2. **Build a model registry** -- registers Naive, RWD, SMA-5, SES, and Holt as named `ModelSpec` entries with optional benchmark flags.
3. **Model comparison per series** -- calls `compare_registry` on each series and prints a formatted `ComparisonTable` of in-sample metrics.
4. **Batch fit-predict** -- uses `fit_predict_many` to run a single model (Naive) across all four series at once, leveraging Rayon when the `parallel` feature is enabled.
5. **Registry fit on one series** -- calls `fit_registry` to fit every registered model on one series and prints RMSE/MAE for each.
6. **Best model per series** -- selects the top model for each series by in-sample RMSE from the comparison results.

## Key types

- `ModelRegistry` -- collection of named model constructors
- `ModelSpec` -- model name, factory closure, and benchmark flag
- `fit_predict_many` -- batch fit-and-predict across multiple series (parallel-aware)
- `fit_registry` -- fit all registry models on a single series
- `compare_registry` / `ComparisonConfig` / `ComparisonTable` -- cross-model comparison utilities
