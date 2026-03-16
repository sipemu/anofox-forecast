# Unified PostProcessor API

**Run:** `cargo run --example postprocess_unified_api`

## What this example demonstrates

Demonstrates the uniform `PostProcessor` interface that lets you swap between Conformal, HistoricalSim, Normal, and IDR postprocessing methods with the same train/predict API. Also shows a model-selection workflow, the `point_to_quantiles` convenience method, and custom model configuration.

## Sections

1. **Method comparison** -- Trains Conformal, HistoricalSim, Normal, and IDR processors on the same data and prints average interval width and first prediction interval for each.
2. **Model selection workflow** -- Splits data into train/validation sets, evaluates each method by coverage and width, and selects the best method using a score that penalizes undercoverage.
3. **Convenience method: point_to_quantiles** -- Uses `point_to_quantiles` for a one-call conversion from point forecasts + actuals to quantile forecasts.
4. **Custom model configuration** -- Constructs a 95% Jackknife+ Conformal model via `PostModel::conformal_with_method` and `PostProcessor::new` to show low-level customization.

## Key types

- `PostProcessor` -- unified entry point with constructors `conformal`, `historical_sim`, `normal`, `idr`, and `new`
- `PostModel` -- model configuration, e.g. `conformal_with_method`
- `ConformalMethod` -- enum for conformal variants (e.g. `JackknifePlus`)
- `PointForecasts` -- wrapper for point forecast values
