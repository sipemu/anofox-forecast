# Complexity Features

**Run:** `cargo run --example complexity`

## What this example demonstrates

Computes three complexity measures -- C3 statistic, CID_CE, and Lempel-Ziv complexity -- on seven synthetic series (sine, complex wave, pseudo-random, constant, linear, step, and square wave). Demonstrates how each metric captures a different aspect of structural complexity.

## Sections

1. **C3 statistic** -- third-order non-linearity measure at lags 1, 2, and 3.
2. **CID_CE** -- complexity-invariant distance estimate, both normalized and raw.
3. **Lempel-Ziv complexity** -- algorithmic complexity based on unique pattern counts.
4. **Series length effect** -- shows how Lempel-Ziv scales with series length (50, 100, 200 points).
5. **Complexity comparison** -- side-by-side table of C3, CID_CE, and Lempel-Ziv for all series.
6. **Complexity ranking** -- ranks series by CID_CE from most to least complex.
7. **Understanding C3** -- compares symmetric (sine) vs asymmetric (sawtooth) series to explain C3.
8. **CID_CE interpretation** -- explains the formula and what low vs high values mean.
9. **Practical applications** -- use cases for each complexity metric.
10. **Feature selection guide** -- when to choose C3, CID_CE, Lempel-Ziv, or all three.

## Key types

- `anofox_forecast::features::complexity` -- `c3`, `cid_ce`, `lempel_ziv_complexity`
