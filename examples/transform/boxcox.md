# Box-Cox Transformation

**Run:** `cargo run --example boxcox`

## What this example demonstrates

Walks through the Box-Cox power transformation: checking data suitability, automatic lambda selection, manual lambda effects, inverse transforms, shifted Box-Cox for non-positive data, and normality improvement. Provides a practical forecasting workflow using Box-Cox.

## Sections

1. **Suitability check** -- verifies whether a series has all-positive values required by Box-Cox.
2. **Automatic lambda selection** -- uses `boxcox_auto` to find the optimal lambda for a right-skewed series.
3. **Different lambda values** -- explains common lambdas (-1 through 2) and transforms a test value with each.
4. **Inverse transformation** -- round-trips data through `boxcox_auto` and `inv_boxcox`, measuring reconstruction error.
5. **Box-Cox with shift** -- applies `boxcox_shifted` to a series containing zeros.
6. **Normality improvement** -- compares skewness before and after Box-Cox for right-skewed, exponential, and log-normal-like data.
7. **Lambda estimation** -- uses `boxcox_lambda` and compares variance/skewness across a range of fixed lambdas.
8. **Practical applications** -- when to use Box-Cox and common scenarios.
9. **Forecasting workflow** -- step-by-step guide for using Box-Cox in a forecast pipeline (transform, model, inverse-transform).

## Key types

- `anofox_forecast::transform` -- `boxcox`, `boxcox_auto`, `boxcox_lambda`, `boxcox_shifted`, `inv_boxcox`, `is_boxcox_suitable`
