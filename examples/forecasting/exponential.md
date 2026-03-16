# Exponential Smoothing

**Run:** `cargo run --example exponential`

## What this example demonstrates

Demonstrates the full exponential smoothing family: SES, Holt's linear trend (with and without damping), Holt-Winters (additive and multiplicative seasonality), the ETS state-space framework, and AutoETS for automatic model selection.

## Sections

1. **Simple Exponential Smoothing** -- Fits SES with a fixed alpha (0.3) and an auto-optimized alpha; prints levels and flat forecasts.
2. **Holt's Linear Trend** -- Fits auto-optimized Holt to trend-only data; prints alpha, beta, level, and trend.
3. **Holt's Damped Trend** -- Fits `auto_damped`; prints the damping parameter phi and compares h=10 forecasts between damped and undamped.
4. **Holt-Winters** -- Fits additive and multiplicative seasonal models with period 12; prints smoothing parameters, seasonal indices, and 12-step forecasts with 95% confidence intervals.
5. **ETS state-space** -- Manually specifies an ETS(A,A,A) model and fits it.
6. **AutoETS** -- Automatically selects the best ETS specification; prints the top 5 models by AIC and 6-step forecasts.
7. **Model comparison** -- Compares 1-step-ahead predictions from SES, Holt, Holt-Winters, and AutoETS against the last observed value.

## Key types

- `SimpleExponentialSmoothing` -- SES with fixed or auto-optimized alpha
- `HoltLinearTrend` -- double exponential smoothing with optional damping
- `HoltWinters` -- triple exponential smoothing with `SeasonalType` (Additive/Multiplicative)
- `ETS`, `ETSSpec` -- state-space ETS with explicit `ErrorType`, `TrendType`, `ETSSeasonalType`
- `AutoETS`, `AutoETSConfig` -- automatic ETS model selection by AIC
- `Forecaster` -- trait for `fit`, `predict`, `predict_with_intervals`
