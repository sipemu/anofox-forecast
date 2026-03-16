# Scaling and Normalization

**Run:** `cargo run --example scaling`

## What this example demonstrates

Applies five scaling methods -- centering, Z-score standardization, min-max normalization, custom range scaling, and robust (IQR-based) scaling -- to a series containing an outlier. Compares their behavior and shows how robust scaling handles outliers better than Z-score or min-max approaches.

## Sections

1. **Centering** -- subtracts the mean, verifies the result has zero mean.
2. **Standardization (Z-score)** -- transforms to zero mean and unit variance, demonstrates inverse transform round-trip.
3. **Min-max normalization** -- scales to [0, 1] range.
4. **Custom range scaling** -- scales to [-1, 1] using `scale_to_range`.
5. **Robust scaling** -- uses median and IQR; compares the outlier's Z-score vs robust-scaled value.
6. **Comparison table** -- side-by-side view of all five methods for every data point.
7. **When to use each method** -- guidance on centering, Z-score, min-max, custom range, and robust scaling.
8. **Effect on different distributions** -- compares Z-score and robust scaling ranges for normal-like, skewed, and outlier-heavy series.

## Key types

- `anofox_forecast::transform` -- `center`, `standardize`, `normalize`, `scale_to_range`, `robust_scale`
- `Standardized` / `Centered` / `Normalized` / `RobustScaled` result structs (with `.data` and `.inverse()`)
