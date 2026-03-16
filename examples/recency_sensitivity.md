# Recency Sensitivity

**Run:** `cargo run --example recency_sensitivity`

## What this example demonstrates

Explores the bias-variance tradeoff of recency windows for trend estimation. Varies both the changepoint location and the recency fraction to show when narrow windows help (regime changes) and when they hurt (stable series with noise). Also demonstrates `Recency::Auto`, which uses PELT changepoint detection to choose the window automatically.

## Sections

1. **Slope recovery table** -- For changepoints at 60%, 80%, 90%, and 95% of a 200-point series, prints estimated slopes across recency fractions (5%-50%), Full, and Window(10). Target slope is -2.0.
2. **Forecast RMSE table** -- Same grid, but reports 20-step-ahead RMSE instead of slope, showing how window choice affects forecast accuracy.
3. **TheilSen vs Polynomial** -- Compares `TheilSenTrend` and `PolynomialTrend(1)` slopes across recency fractions for a 90% changepoint, demonstrating both converge at small windows.
4. **The tradeoff** -- Fits a stable linear series with noise (no regime change); shows that narrower windows increase slope estimation error and forecast RMSE.
5. **Recency::Auto** -- Uses PELT changepoint detection to automatically set the recency window. Compares Auto slope and RMSE against fixed 30% across all changepoint positions. Also demonstrates custom `AutoRecencyConfig` with a lower penalty for more sensitive detection.

## Key types

- `PolynomialTrend` -- linear trend fitting with configurable recency
- `TheilSenTrend` -- robust slope estimator
- `Recency` -- `Full`, `Fraction(f64)`, `Window(usize)`, `Auto(AutoRecencyConfig)`
- `AutoRecencyConfig` -- configures PELT penalty and fallback fraction for `Recency::Auto`
- `TrendComponent` -- trait for `fit_trend` and `predict_trend`
