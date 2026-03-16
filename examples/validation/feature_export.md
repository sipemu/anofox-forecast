# Feature Export

**Run:** `cargo run --example feature_export --release`

## What this example demonstrates

Extracts a comprehensive set of time series features from synthetic CSV data and exports the results to CSV for comparison with the Python tsfresh package. It exercises every feature category in the `features` module across 11 series types.

## Sections

1. **CSV reading** -- Loads values from `validation/data/*.csv` for series types including stationary, trend, seasonal, intermittent, structural break, and others.
2. **Feature extraction** -- Calls functions from seven feature sub-modules (basic, distribution, autocorrelation, counting, entropy, complexity, trend, change) with various parameterisations to produce ~100 features per series.
3. **Result export** -- Writes all features to `validation/results/rust/features.csv` with columns `series_type`, `feature_name`, `value`.

## Key types

- `basic` -- mean, variance, standard deviation, median, min/max, abs energy, mean absolute change, root mean square, etc.
- `distribution` -- skewness, kurtosis, quantiles, variation coefficient, ratio beyond r sigma
- `autocorrelation` -- ACF, PACF, aggregated autocorrelation, time reversal asymmetry
- `counting` -- count above/below mean, number of peaks, zero crossings, longest strikes, duplicate detection, index mass quantile
- `entropy` -- sample entropy, approximate entropy, permutation entropy, binned entropy
- `complexity` -- CID-CE, C3, Lempel-Ziv complexity
- `trend` -- linear trend (slope/intercept/r-value/stderr/p-value), aggregated linear trend, AR coefficients, augmented Dickey-Fuller
- `change` -- change quantiles, energy ratio by chunks, reoccurring datapoints/values
