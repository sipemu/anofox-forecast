# Distribution Features

**Run:** `cargo run --example distribution`

## What this example demonstrates

Explores distribution-shape features (skewness, kurtosis, quantiles, symmetry) and counting features (thresholds, crossings, peaks, strikes, duplicates) across several synthetic distributions. Shows how these features distinguish symmetric, skewed, heavy-tailed, and uniform data.

## Sections

1. **Comparing distributions** -- computes skewness and kurtosis for symmetric, right-skewed, left-skewed, heavy-tailed, and uniform series.
2. **Quantiles** -- calculates quantiles at several levels and the interquartile range.
3. **Variation coefficient** -- relative variability (std dev / mean) for each distribution.
4. **Symmetry analysis** -- checks whether each distribution looks symmetric within a tolerance.
5. **Large standard deviation check** -- tests if std dev exceeds a fraction of the range for several thresholds.
6. **Variance vs standard deviation** -- reports whether variance exceeds standard deviation.
7. **Ratio beyond R sigma** -- percentage of values more than R standard deviations from the mean.
8. **Counting features** -- counts above/below a threshold and above/below the mean.
9. **Range count** -- counts values within specified intervals.
10. **Zero/mean crossings** -- counts crossings at given levels on an oscillating series.
11. **Longest strikes** -- longest consecutive runs above and below the mean.
12. **Peak detection** -- number of peaks at different support widths.
13. **Location of extremes** -- normalized positions of first/last min and max.
14. **Index mass quantile** -- index where cumulative sum reaches a quantile of the total.
15. **Duplicate detection** -- checks for duplicate values, duplicate max, and duplicate min.
16. **Feature interpretation guide** -- explains skewness, kurtosis, variation coefficient, and ratio beyond R sigma.

## Key types

- `anofox_forecast::features::distribution` -- `skewness`, `kurtosis`, `quantile`, `variation_coefficient`, `symmetry_looking`, `large_standard_deviation`, `variance_larger_than_standard_deviation`, `ratio_beyond_r_sigma`
- `anofox_forecast::features::counting` -- `count_above`, `count_below`, `count_above_mean`, `count_below_mean`, `range_count`, `number_crossing_m`, `longest_strike_above_mean`, `longest_strike_below_mean`, `number_peaks`, `first_location_of_maximum`, `last_location_of_maximum`, `first_location_of_minimum`, `last_location_of_minimum`, `index_mass_quantile`, `has_duplicate`, `has_duplicate_max`, `has_duplicate_min`
