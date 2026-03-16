# Conformal Prediction Methods

**Run:** `cargo run --example postprocess_conformal`

## What this example demonstrates

Compares three conformal prediction methods -- split conformal, cross-validation conformal, and Jackknife+ -- showing how each constructs prediction intervals from the same data, with trade-offs in speed, data efficiency, and coverage guarantees.

## Sections

1. **Split conformal** -- Uses `ConformalMethod::Split { cal_fraction: 0.2 }` to hold out 20% of the data for calibration. Fastest method.
2. **Cross-validation conformal** -- Uses `ConformalMethod::CrossVal { n_folds: 5 }` to calibrate with all data across 5 folds.
3. **Jackknife+** -- Uses `ConformalMethod::JackknifePlus` for leave-one-out calibration with finite-sample coverage guarantees.
4. **Comparison** -- Prints a table comparing average interval width and quantile value across all three methods.

## Key types

- `ConformalPredictor` -- core conformal prediction engine
- `ConformalMethod` -- enum selecting `Split`, `CrossVal`, or `JackknifePlus`
- `PointForecasts` -- input wrapper
- `PredictionIntervals` -- output with `lower()` / `upper()` slices
