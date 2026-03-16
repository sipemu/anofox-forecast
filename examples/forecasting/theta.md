# Theta Method

**Run:** `cargo run --example theta`

## What this example demonstrates

Demonstrates the Theta method (winner of the M3 competition), which decomposes a series into "theta lines" combining SES and linear trend. Covers the standard method, custom theta/alpha parameters, seasonal Theta, confidence intervals, and the effect of trend on forecasts.

## Sections

1. **Standard Theta (theta=2)** -- Fits the default Theta model; prints theta, optimized alpha, slope, and 10-step forecasts.
2. **Custom theta parameter** -- Sweeps theta over {0.5, 1.0, 1.5, 2.0, 3.0} and shows how it affects h=1 and h=10 forecasts.
3. **Fixed alpha** -- Sweeps alpha over {0.1, 0.3, 0.5, 0.7, 0.9} with fixed SES smoothing.
4. **Seasonal Theta** -- Fits `Theta::seasonal(12)` to seasonal data; prints alpha, slope, and one full season of forecasts.
5. **Confidence intervals** -- Generates 10-step 95% intervals and prints the interval width at each horizon.
6. **Theta decomposition explained** -- Text explanation of how theta lines, SES, and linear trend combine.
7. **Effect of trend** -- Compares Theta forecasts on trending data (slope=2) vs flat data to show how drift drives long-horizon predictions.
8. **Residual analysis** -- Reports residual count, mean, and standard deviation.

## Key types

- `Theta` -- Theta method with constructors `new`, `with_theta`, `with_alpha`, `seasonal`
- `Forecaster` -- trait for `fit`, `predict`, `predict_with_intervals`, `residuals`
