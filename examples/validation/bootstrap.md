# Bootstrap Confidence Intervals

**Run:** `cargo run --example bootstrap`

## What this example demonstrates

Generates empirical confidence intervals for forecasts using residual bootstrap and block bootstrap methods. The example shows configuration options, reproducibility via seeds, the difference between residual and block resampling, and how interval width grows with confidence level.

## Sections

1. **BootstrapConfig** -- Demonstrates default and custom configuration (number of samples, block size, seed).
2. **Residual bootstrap intervals** -- Fits a Naive model and produces 95% bootstrap intervals over a 10-step horizon using `bootstrap_intervals`.
3. **bootstrap_forecast()** -- Fits a Simple Exponential Smoothing model and produces a combined `Forecast` with point predictions and bootstrap intervals, showing interval widths.
4. **Reproducibility with seeds** -- Runs two bootstrap passes with the same seed and confirms the results are identical.
5. **Block bootstrap vs residual bootstrap** -- Compares interval widths from standard residual resampling and block resampling (block size 5), which preserves autocorrelation structure.
6. **Different confidence levels** -- Computes intervals at 80%, 90%, 95%, and 99% levels, showing that wider confidence requires wider intervals.

## Key types

- `BootstrapConfig` -- builder for sample count, block size, and seed
- `bootstrap_intervals()` -- returns `BootstrapResult` with lower/upper bounds
- `bootstrap_forecast()` -- returns a `Forecast` combining point predictions with bootstrap intervals
- `Naive` / `SimpleExponentialSmoothing` -- models used as examples
