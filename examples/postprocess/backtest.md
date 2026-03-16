# Backtesting PostProcessors

**Run:** `cargo run --example postprocess_backtest`

## What this example demonstrates

Shows how to evaluate postprocessor performance using backtesting with rolling and expanding windows, horizon-aware calibration, and per-horizon model training. Also compares multiple postprocessing methods (Conformal, HistoricalSim, Normal) side by side using backtest metrics.

## Sections

1. **Basic backtest (expanding window)** -- Configures an expanding-window backtest with initial window 50, step 10, horizon 5; prints fold count, coverage, interval width, and calibration error.
2. **Per-fold summary** -- Iterates over the first 5 folds showing train size, test size, coverage, and average width for each.
3. **Rolling window backtest** -- Switches to a fixed-size rolling window and verifies that train sizes stay constant across folds.
4. **Horizon-aware backtest** -- Enables `horizon_aware(true)` and prints coverage and average width broken down by forecast horizon 1--7.
5. **Train calibrated model from backtest** -- Pools backtest data to train a single production-ready model and generates interval predictions for new forecasts.
6. **Horizon-specific calibration** -- Trains separate per-horizon models via `calibrated_model_by_horizon` and predicts with `predict_intervals_by_horizon`.
7. **Method comparison** -- Backtests Conformal, HistoricalSim, and Normal processors on the same data and compares coverage, width, and calibration error.

## Key types

- `PostProcessor` -- unified interface with constructors `conformal`, `historical_sim`, `normal`
- `BacktestConfig` -- builder for initial window, step, horizon, expanding/rolling, and horizon-aware settings
- `PointForecasts` -- wrapper for point forecast values
- `BacktestResult` (returned by `backtest`) -- provides `coverage`, `interval_widths`, `calibration_error`, `coverage_by_horizon`, `calibrated_model`, `calibrated_model_by_horizon`
