# STL Decomposition

**Run:** `cargo run --example stl_decomposition`

## What this example demonstrates

Decomposes time series into trend, seasonal, and remainder components using STL (Seasonal-Trend decomposition using LOESS). Covers basic and custom-parameter STL, robust mode for outlier handling, and MSTL for multiple seasonal periods.

## Sections

1. **Basic STL decomposition** -- decomposes 120 observations (period=12) and verifies the additive reconstruction error is near zero.
2. **Component statistics** -- prints mean and standard deviation for trend, seasonal, and remainder components.
3. **Trend and seasonal strength** -- computes strength metrics and labels them as Strong or Weak (threshold 0.5).
4. **First season of components** -- prints a table of original, trend, seasonal, and remainder values for the first 12 observations.
5. **Custom parameters** -- configures seasonal smoothness, trend smoothness, and inner iterations, then compares decomposition strength to the defaults.
6. **Robust STL** -- injects three large outliers, then compares standard vs robust STL to show how robust mode preserves decomposition quality.
7. **MSTL (multiple seasonal periods)** -- generates hourly data with daily (24h) and weekly (168h) seasonality, decomposes with `MSTL`, and prints variance and strength for each seasonal component.

## Key types

- `STL` -- single-period STL decomposition with builder methods (`with_seasonal_smoothness`, `with_trend_smoothness`, `robust`)
- `MSTL` -- multiple-seasonal-period STL decomposition
- `STL::decompose` returns a result with `trend`, `seasonal`, `remainder` vectors plus `trend_strength()` and `seasonal_strength()` methods
