# Entropy Features

**Run:** `cargo run --example entropy`

## What this example demonstrates

Computes several entropy measures -- approximate, sample, permutation, binned, and Fourier -- on series ranging from a regular sine wave to pseudo-random data. Shows how entropy quantifies complexity and predictability, and how parameters affect the results.

## Sections

1. **Series generation** -- creates regular sine, complex multi-frequency, pseudo-random, step function, and constant series.
2. **Approximate entropy (ApEn)** -- compares ApEn at two tolerance levels across all series.
3. **Sample entropy (SampEn)** -- similar comparison using the more consistent sample entropy variant.
4. **Permutation entropy** -- ordinal-pattern entropy at embedding dimensions 3 and 5.
5. **Effect of embedding dimension** -- permutation entropy of the complex series for dimensions 3-7.
6. **Binned entropy** -- histogram-based entropy at 5, 10, and 20 bins.
7. **Fourier entropy** -- entropy of the spectral power distribution.
8. **Parameter sensitivity** -- approximate entropy across tolerance values 0.1-0.5.
9. **Complexity ranking** -- ranks all series by sample entropy.
10. **Practical applications** -- guidance on when to use each entropy type.
11. **Parameter selection guidelines** -- recommended ranges for m, r, dimension, delay, and bin count.

## Key types

- `anofox_forecast::features::entropy` -- `approximate_entropy`, `sample_entropy`, `permutation_entropy`, `binned_entropy`, `fourier_entropy`
