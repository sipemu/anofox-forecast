# Outlier Detection

**Run:** `cargo run --example outlier_detection`

## What this example demonstrates

Identifies anomalous values in a time series using three statistical methods (IQR, Z-score, Modified Z-score). Compares detection sensitivity across methods and thresholds, and covers edge cases like empty, constant, and single-outlier series.

## Sections

1. **IQR method (default)** -- runs `detect_outliers_auto` on a 100-point series with four injected outliers; reports indices, count, and percentage.
2. **IQR with different multipliers** -- sweeps multipliers 1.0, 1.5, 2.0, 3.0 to show sensitivity trade-offs.
3. **Z-score method** -- detects outliers using standard Z-scores at threshold 3.0 and prints per-outlier scores.
4. **Z-score with different thresholds** -- sweeps thresholds 2.0 through 4.0.
5. **Modified Z-score (MAD-based)** -- uses median absolute deviation for robust scoring; prints modified Z-scores for known outlier positions.
6. **Method comparison** -- runs all five method/threshold combinations side by side in a summary table.
7. **Outlier score analysis** -- separates scores into outlier vs non-outlier groups and prints mean/min/max statistics.
8. **Edge cases** -- tests empty series, constant series, and single extreme outlier.

## Key types

- `detect_outliers` -- detect outliers with an explicit `OutlierConfig`
- `detect_outliers_auto` -- detect outliers with default IQR settings
- `OutlierConfig` -- `iqr(multiplier)`, `z_score(threshold)`, `modified_z_score(threshold)`
- `OutlierResult` -- contains `outlier_indices`, `scores`, `is_outlier()`, `outlier_count()`, `outlier_percentage()`
