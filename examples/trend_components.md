# Trend Components

**Run:** `cargo run --example trend_components`

## What this example demonstrates

Shows the trend component system: fitting individual trend models (polynomial, Theil-Sen, exponential, logistic) to synthetic data, using `AutoTrend` for automatic selection via AICc, and observing how recency windows affect slope estimation after regime changes.

## Sections

1. **Linear + noise** -- Fits `PolynomialTrend(1)` and `TheilSenTrend` with 30% recency to a linear series; runs `AutoTrend` and prints AICc score table.
2. **Exponential growth** -- Fits `ExponentialTrend` to exponential data; shows growth rate, R-squared, and AutoTrend selection.
3. **Logistic saturation** -- Fits `LogisticTrend` with a known capacity; prints midpoint, steepness, and 20-step forecast values approaching the asymptote.
4. **Regime change** -- Generates data with a slope reversal at t=60; compares recent-window slope vs global slope and runs AutoTrend with 30% recency.

## Key types

- `PolynomialTrend` -- polynomial (linear, quadratic, etc.) trend fitting
- `TheilSenTrend` -- robust median-based slope estimator
- `ExponentialTrend` -- exponential growth/decay model
- `LogisticTrend` -- S-curve with capacity saturation
- `AutoTrend` -- automatic trend selection using AICc
- `Recency` -- controls which portion of the data is used for fitting (`Full`, `Fraction`, `Window`)
- `TrendComponent` -- trait providing `fit_trend`, `predict_trend`, `trend_features`
