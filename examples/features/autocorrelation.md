# Autocorrelation Features

**Run:** `cargo run --example autocorrelation`

## What this example demonstrates

Computes autocorrelation (ACF), partial autocorrelation (PACF), aggregated autocorrelation, and time-reversal asymmetry on several synthetic series. Illustrates how these features reveal seasonal periods, AR process structure, and stationarity.

## Sections

1. **Series generation** -- creates random-like, AR(1), seasonal (period 12), and trending series.
2. **ACF at various lags** -- tabulates autocorrelation at lags 1, 2, 5, 10, and 12 for each series.
3. **ACF plot** -- prints a text-based correlogram for the seasonal series across lags 1-24.
4. **PACF** -- computes partial autocorrelation at lags 1-4, showing how it isolates direct lag effects.
5. **AR(1) process analysis** -- verifies that ACF decays as phi^k and PACF cuts off after lag 1.
6. **Aggregated autocorrelation** -- summarizes ACF over lags 1-12 using mean, variance, and median.
7. **Time reversal asymmetry** -- measures non-linearity for each series at several lags.
8. **Seasonality detection with ACF** -- finds ACF peaks to detect the seasonal period.
9. **ARIMA model identification** -- reference guide for reading ACF/PACF patterns.
10. **Stationarity indicator** -- compares cumulative |ACF| across series to flag non-stationarity.

## Key types

- `anofox_forecast::features::autocorrelation` -- `autocorrelation`, `partial_autocorrelation`, `agg_autocorrelation`, `time_reversal_asymmetry_statistic`
