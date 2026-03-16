# Cross-Validation

**Run:** `cargo run --example cross_validation`

## What this example demonstrates

Time series cross-validation with expanding and rolling window strategies using the `cross_validate` function and `CVConfig` builder. It shows how to compare baseline models, tune the forecast horizon and step size, and inspect per-fold results.

## Sections

1. **Expanding window CV** -- Creates a `CVConfig::expanding` with initial window 30 and horizon 1, then cross-validates a Naive model. Training data grows each fold.
2. **Rolling window CV** -- Uses `CVConfig::rolling` with a fixed 30-observation window so the training size stays constant across folds.
3. **Multi-step forecast horizon** -- Increases the horizon to 5 and the step size to 5, showing how to evaluate longer-range forecasts with fewer folds.
4. **Model comparison via CV** -- Benchmarks Naive, SeasonalNaive (period 12), and SMA (windows 3, 5, 7) in a single table with MAE, RMSE, SMAPE, and fold count.
5. **Per-fold analysis** -- Iterates over `fold_metrics` to print individual fold accuracy, then finds the best and worst folds by MAE.
6. **Configuration guide and best practices** -- Summarises when to use expanding vs rolling windows, how to choose horizon and step size, and the importance of reporting uncertainty.

## Key types

- `CVConfig` -- builder for cross-validation strategy (`expanding`, `rolling`, `with_step_size`, `with_seasonal_period`)
- `cross_validate` -- runs CV and returns `CVResults` containing `aggregated` metrics and `fold_metrics`
- `Naive`, `SeasonalNaive`, `SimpleMovingAverage` -- baseline forecasters used in the comparison
- `TimeSeries` -- input data container
