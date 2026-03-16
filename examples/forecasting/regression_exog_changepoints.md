# Regression with Exogenous Features, Changepoints, and Backtesting

**Run:** `cargo run --example regression_exog_changepoints`

## What this example demonstrates

How external (exogenous) features interact with changepoint detection during
cross-validation and pipeline evaluation. Covers the full lifecycle from
feature attachment through backtesting to batch result filtering.

## Sections

1. **OLS with exogenous regressor** — Attaches a "temperature" regressor to the
   `TimeSeries` via `CalendarAnnotations`. When `use_exog` is true (the default),
   `fit()` picks up all regressors automatically. At prediction time, `predict()`
   errors because future regressor values are needed — you must call
   `predict_with_exog()` with a `HashMap<String, Vec<f64>>`.

2. **Ridge with exog + changepoint** — Combines exogenous features with a
   `ChangepointFeature` step function at the level-shift index. R² jumps from
   ~0.74 to ~0.99 because the model can now capture both the external driver
   and the structural break.

3. **Feature safety classification** — Shows how `classify_features()` labels
   each column: trend and Fourier terms are `Deterministic`, the changepoint
   step function is `Structural`, and temperature is `External`. This audit
   tells you which features need special handling during cross-validation.

4. **Cross-validation with exogenous features** — Demonstrates the CV framework
   automatically slicing regressors along with the series and calling
   `predict_with_exog()` when the model has exogenous features. Compares
   accuracy with and without exogenous regressors.

5. **Pipeline structural break detection** — Runs a full pipeline on a series
   with a large level shift near the end. PELT detects the changepoint in the
   holdout window, sets `structural_break_in_holdout = true`, and the
   `PipelineReport` includes a Warnings section.

6. **Batch partition by structural break** — Uses
   `partition_by_structural_break()` to separate pipeline results into clean
   (safe for aggregated metrics) and flagged (exclude from summaries) index
   vectors.

## How it works under the hood

### Exogenous regressor flow

```
TimeSeries + CalendarAnnotations("temperature" → [values])
    │
    ▼  fit()
RegressionFeatures.build_matrices()
    │  → series.all_regressors() → HashMap
    │  → for each regressor: populate design matrix column
    │
    ▼  predict_with_exog(horizon, future_regs)
RegressionFeatures.build_future_matrix()
    │  → fill regressor columns from future_regs HashMap
    │  → model.predict(&x_future)
    ▼
Forecast
```

### Changepoint during cross-validation

```
cross_validate()
    │
    ▼  per fold
series.slice(train_start, train_end)
    │  → values sliced to [start..end]
    │  → calendar regressors sliced to [start..end]  ← fixed (was full-length)
    │
    ▼  fit + predict
model.fit(&train_slice)
    │  → ChangepointFeature.compute(n_train) → binary columns
    │  → forward-fill values = last training value per column
    │
model.predict_with_exog(horizon, test_regs)  ← auto when has_exog()
    │  → step function frozen at last training regime
    │  → exog values from test slice
    ▼
compare actual vs predicted
```

### Pipeline structural break detection

```
Pipeline.execute()
    │
    ▼  profile
DataProfile::from_series() → PELT changepoint detection
    │
    ▼  holdout split
train = series[0..n-holdout], test = series[n-holdout..n]
    │
    ▼  check
any changepoint in [holdout_start, n)?
    │  yes → log warning + set structural_break_in_holdout = true
    │  no  → flag = false
    │
    ▼  result
PipelineResult { structural_break_in_holdout, ... }
    │
    ▼  batch
partition_by_structural_break(&results) → (clean_idx, flagged_idx)
```

## Key types

- `CalendarAnnotations` — attaches regressors to a TimeSeries
- `predict_with_exog()` — prediction with future regressor values
- `structural_break_in_holdout` — pipeline flag for holdout contamination
- `partition_by_structural_break()` — batch filtering helper
- `PipelineReport` — includes Warnings section when break detected
