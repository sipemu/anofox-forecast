# Changepoint Detection: Cost Function Types

**Run:** `cargo run --example changepoint_types`

## What this example demonstrates

Shows every PELT cost function on purpose-built synthetic data so you can see which cost function detects which kind of change. Ends with a side-by-side comparison of all cost functions on the same dataset.

## Sections

1. **L2 (mean change)** -- detects a simple level shift from 10 to 30; general-purpose default.
2. **L1 (robust to outliers)** -- repeats the level shift with extreme outliers injected; compares L1 vs L2 robustness.
3. **Normal (variance change)** -- detects a 10x variance increase while the mean stays constant.
4. **MeanVariance (joint detection)** -- detects simultaneous mean and variance changes; compared against Normal cost.
5. **LinearTrend (slope change)** -- detects a slope reversal (+1 to -1) in piecewise-linear data; compared against L2.
6. **Cusum (sustained shift)** -- detects a sustained level shift using cumulative sums, suited for process monitoring.
7. **Poisson (rate change)** -- detects a rate change in count data from ~2 to ~10 events per period.
8. **Comparative analysis** -- runs L1, L2, Normal, MeanVariance, LinearTrend, and Cusum on the same mean+variance shift data.
9. **Selection guide** -- prints when to use each cost function.

## Key types

- `pelt_detect` -- PELT changepoint detection function
- `PeltConfig` -- configuration builder (penalty, cost function, min segment length)
- `CostFunction` -- `L1`, `L2`, `Normal`, `MeanVariance`, `LinearTrend`, `Cusum`, `Poisson`
