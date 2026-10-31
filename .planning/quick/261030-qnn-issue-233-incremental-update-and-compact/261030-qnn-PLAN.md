---
quick_id: 261030-qnn
issue: 233
branch: feat/233-incremental-update-state
---

# Plan: #233 incremental model update + compact serializable state

## What
Issue #233 asks for `update(&mut self, new_obs: &[f64]) -> Result<()>` (filter the fitted state
forward at fixed parameters) and `state()` / `from_state()` returning a serde-serializable state
that is O(1) in history length, so the DuckDB extension can store a compact aggregate state and
`combine_aggr` it with new rows at O(new rows) cost. When to refit is left to the caller.

## Scope (first set, end to end)
Exponential-smoothing family: `SimpleExponentialSmoothing`, `HoltLinearTrend`, `HoltWinters`,
`ETS` (all 15+ specs incl. damped / multiplicative), `AutoETS` (delegates to its selected ETS).
Follow-ups (named in the issue, not in this task): Theta family, Croston/TSB, Naive /
SeasonalNaive / window averages (their interval code recomputes variance from the full residual
vector, so they need an internal refactor first).

## How
- New module `src/models/incremental.rs`: trait `IncrementalForecaster: Forecaster` with
  `update`, `state`, `from_state` (Sized-bound so the trait stays object-safe), plus
  `#[non_exhaustive] enum ModelState` (one variant per model, per-model state structs, serde
  derives behind the `serde` feature) and `ModelState::restore()` -> `Box<dyn
  IncrementalForecaster>`. Purely additive: `Forecaster` is untouched.
- Each model's per-observation recursion is extracted into one private step helper used by both
  `fit()` and `update()`, so update is bit-for-bit the fit recursion (no duplicated equations).
- Running error sums (needed for `interval_sigma2` / `residual_variance` / ETS loglik/AIC) are
  stored as new private fields with `serde(default)` so old serialized models still load (and
  give `FitRequired` on update rather than wrong numbers).
- Models fit with exogenous regressors reject `update()` (future regressor values are not part
  of the API).
- After `update()`, fitted/residuals/training values are appended when present; a restored
  model holds none of them.

## Verification
1. Exactness: fit on n + update with k == fit on n+k with the same fixed parameters/initial
   states (state, point forecasts, intervals) to <= 1e-9 relative, for every model and every
   ETS spec class.
2. Tracking: auto-fit on n + update k vs full auto refit on n+k: report forecast deviation.
3. Round-trip: `state()` -> JSON / bincode -> `from_state()` gives identical forecasts and
   identical subsequent updates.
4. Measure: state size (bincode/JSON bytes) vs full serialized model across n; update cost vs
   full refit (release build).
5. Gates: fmt, clippy -D warnings, full `cargo test --all-features`, WASM check.
