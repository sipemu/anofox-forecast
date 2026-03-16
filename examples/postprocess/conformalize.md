# Conformalize

**Run:** `cargo run --example postprocess_conformalize`

## What this example demonstrates

Demonstrates how to recalibrate overconfident (too-narrow) quantile forecasts using conformal prediction. The `conformalize` function learns per-quantile adjustments from a calibration set and applies them to widen intervals so they achieve the target coverage.

## Sections

1. **Generate miscalibrated forecasts** -- Creates synthetic calibration and test data where the q10/q90 intervals are deliberately too tight (only +/-0.5 around the true value).
2. **Check original coverage** -- Measures 80%-interval coverage on the calibration set to confirm intervals are overconfident.
3. **Apply conformalize** -- Calls `conformalize(&test, &calib, &calib_actuals)` and prints the learned per-quantile adjustments.
4. **Before vs after comparison** -- Shows original and calibrated q10/q90 bounds alongside actuals for the first 5 test points.
5. **Coverage comparison** -- Reports test-set coverage before and after conformalization against the 80% target.
6. **Interval width comparison** -- Compares average interval width before and after, showing the percentage increase needed for proper calibration.

## Key types

- `conformalize` -- free function that recalibrates quantile forecasts using conformal prediction residuals
- `QuantileForecasts` -- matrix of quantile forecasts constructed via `from_values(quantile_levels, values)`
- `ConformalizationResult` (returned by `conformalize`) -- provides `forecasts()` and `adjustments()`
