# Window Functions

**Run:** `cargo run --example window`

## What this example demonstrates

Demonstrates rolling, expanding, and exponentially weighted moving (EWM) window functions on time series data. Compares smoothing behaviour, lag characteristics, and responsiveness to step changes across different window sizes and alpha values.

## Sections

1. **Rolling mean** -- computes rolling mean with windows of 3 and 5, plus a centered variant.
2. **Rolling statistics** -- tabulates mean, std, min, max, sum, and median with a window of 5.
3. **Expanding windows** -- cumulative mean, min, max, and sum that grow with each new observation.
4. **EWM mean** -- exponentially weighted mean at three alpha levels (0.5, 0.33, 0.18) showing different reactivity.
5. **EWM volatility** -- exponentially weighted standard deviation and variance.
6. **Rolling vs EWM comparison** -- side-by-side table of rolling mean (window 5) vs EWM mean (alpha 0.33).
7. **Smoothing noisy data** -- applies rolling and EWM smoothing to a noisy trend series.
8. **Lag and responsiveness** -- tracks how rolling mean and EWM respond to an abrupt step change from 10 to 20.
9. **Use cases** -- practical guidance for rolling mean, rolling std, rolling min/max, expanding mean, EWM mean, and EWM std.
10. **Window selection guide** -- advice on window size, rolling vs EWM trade-offs, and centered vs non-centered windows.

## Key types

- `anofox_forecast::transform` -- `rolling_mean`, `rolling_std`, `rolling_var`, `rolling_min`, `rolling_max`, `rolling_sum`, `rolling_median`, `expanding_mean`, `expanding_min`, `expanding_max`, `expanding_sum`, `ewm_mean`, `ewm_std`, `ewm_var`
