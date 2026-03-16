# Missing Value Imputation

**Run:** `cargo run --example imputation`

## What this example demonstrates

Walks through every imputation strategy available in anofox-forecast, from simple fill policies (forward, backward, mean, median, interpolation) to moving-average, seasonal, and regressor-aware imputation. Finishes with an end-to-end impute-then-forecast workflow.

## Sections

1. **Create series with missing values** -- builds a 30-day series and injects NaN at leading, interior, adjacent, and trailing positions plus an Infinity value.
2. **Missing value metadata** -- inspects `has_missing_values`, `missing_count`, and `missing_mask` to locate gaps.
3. **Policy-based imputation** -- demonstrates `ForwardFill`, `BackwardFill`, combined forward+backward (`imputed_forward_backward`), `FillMean`, `FillMedian`, and `Interpolate`.
4. **Moving average imputation** -- fills gaps using a rolling window of size 5 with multi-pass for adjacent gaps.
5. **Seasonal imputation** -- creates a 42-day series with weekly pattern, removes two Mondays, and fills them using the median of same-weekday values (period=7).
6. **Regressor imputation** -- attaches `CalendarAnnotations` with missing regressor values ("price", "promo") and imputes them via `with_imputed_regressors(FillMean)`.
7. **End-to-end workflow** -- imputes, confirms no missing values remain, fits a Naive model, and produces a 5-step forecast.

## Key types

- `TimeSeries` -- `sanitized(policy)`, `imputed_forward_backward()`, `imputed_moving_average(window)`, `imputed_seasonal(period)`, `with_imputed_regressors(policy)`
- `MissingValuePolicy` -- `ForwardFill`, `BackwardFill`, `FillMean`, `FillMedian`, `Interpolate`
- `CalendarAnnotations` -- holds external regressors that can also contain missing values
- `Naive` / `Forecaster` -- used in the end-to-end forecast step
