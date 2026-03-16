# Changepoint Detection

**Run:** `cargo run --example changepoint`

## What this example demonstrates

Detects structural breaks in time series using the PELT (Pruned Exact Linear Time) algorithm. Explores how penalty values, cost functions, and minimum segment lengths affect detection, and shows segment-level analysis of the results.

## Sections

1. **Clear level shift** -- detects two changepoints in a three-level series (10, 50, 25) with a fixed penalty.
2. **Effect of penalty parameter** -- sweeps penalties from 0.5 to 100 on a two-level series, showing how higher penalties yield fewer changepoints.
3. **BIC and AIC penalties** -- uses `PeltConfig::with_bic_penalty` and `with_aic_penalty` as automatic penalty selection methods.
4. **Cost functions** -- compares L2 (mean-only) vs Normal (mean+variance) on data with a variance change.
5. **Minimum segment length** -- demonstrates how `min_segment_length` prevents detection of very short segments.
6. **Gradual vs step changes** -- contrasts PELT behavior on an abrupt step change versus a linear ramp.
7. **Multiple changepoints** -- generates a series with four known changepoints and checks how many PELT recovers.
8. **Segment analysis** -- prints start, end, and mean for each detected segment.
9. **No changepoints case** -- runs PELT on a near-constant series with high penalty to show the single-segment result.

## Key types

- `pelt_detect` -- runs PELT and returns a result with `changepoints`, `segments`, and `segment_means`
- `PeltConfig` -- builder for penalty, cost function, and minimum segment length
- `CostFunction` -- `L2`, `Normal`, `L1`, and others
