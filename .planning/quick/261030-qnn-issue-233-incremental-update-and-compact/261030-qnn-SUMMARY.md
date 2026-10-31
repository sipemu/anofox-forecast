---
quick_id: 261030-qnn
issue: 233
branch: feat/233-incremental-update-state
status: complete
commits: [ebd583f]
---

# Summary: #233 incremental update + compact serializable state

## What was built
- `src/models/incremental.rs` (new, re-exported as `anofox_forecast::models::{IncrementalForecaster,
  ModelState}`):
  - `trait IncrementalForecaster: Forecaster { fn update(&mut self, new_obs: &[f64]) -> Result<()>;
    fn state(&self) -> Result<ModelState>; fn from_state(state: ModelState) -> Result<Self> where
    Self: Sized; }` -- additive, `Forecaster` untouched, trait is object-safe.
  - `#[non_exhaustive] enum ModelState` with per-model state structs (`SesState`, `HoltState`,
    `HoltWintersState`, `EtsState`; public fields, serde derives behind the `serde` feature),
    `model_name()`, `n_obs()`, and `restore()` -> `Box<dyn IncrementalForecaster>` for callers
    that store states of mixed models.
- Implemented for `SimpleExponentialSmoothing`, `HoltLinearTrend` (incl. damped),
  `HoltWinters` (additive / multiplicative), `ETS` (all 18 valid specs) and `AutoETS`
  (delegates to its selected ETS; exports an ETS state).
- Each model's per-step recursion was extracted into one private `step` helper that both `fit()`
  and `update()` call, and the variance / sigma2 / loglik / AIC computation into one
  `set_variances` / `set_fit_statistics` helper, so update cannot drift from fit.
- New private fields holding the running error sums (`sse` on SES/Holt/HW, `error_sums` on ETS),
  `#[serde(default)]`, so models serialized by older versions still deserialize and predict; on
  those, `update()` returns `FitRequired` instead of producing wrong intervals (tested).
- `update()` rejects models fit with exogenous regressors (`InvalidParameter`), non-finite
  observations (`MissingValues`, state untouched), and unfitted models (`FitRequired`). An empty
  slice is a no-op. On a fully fitted model, fitted values / residuals / training values are
  appended; a model restored from a state holds none of them.

## Verification (tests/incremental_update.rs, 12 tests)
| Contract | Result |
|---|---|
| fit(n) + update(k) == fit(n+k) at fixed params: states, point forecasts, fitted values, residuals | equal to 1e-9 relative, SES / Holt / damped Holt / HW add+mult / ETS x 18 specs |
| same, prediction-interval half-widths | ETS: 1e-9; SES/Holt/HW: 1e-6 (see deviation 1) |
| same, ETS AIC / BIC | 1e-9 |
| auto-fit(n=120) + update(k=24) vs full auto refit(144): max rel. 12-step forecast deviation | SES 0, Holt 4.3e-4, HoltWinters 5.3e-4, ETS(A,A,A) 1.5e-3 (test bound 2e-2) |
| `state()` -> `from_state()` -> predict / update identical to the original | exact (bincode), 1e-12 (JSON) |
| state size independent of history | ETS(A,A,A) p=12: 204 B bincode at n = 48, 480, 4800 |

Existing suites unaffected: lib exponential unit tests (146), `ets_r_reference`,
`ets_validation`, `ets_intervals_reference`, `es_family_intervals_reference`,
`persistence_integration` all pass unchanged.

## Measured (examples/validation/incremental_update_cost.rs, release, median of 7)
Update = restore from the compact state + `update(k)`; refit = `fit()` on n+k.

| model | n | k | refit us | update us | state B (bincode) | full model B (bincode) |
|---|--:|--:|--:|--:|--:|--:|
| SES auto | 120 | 1 | 5.5 | 0.07 | 38 | 2,963 |
| SES auto | 12,000 | 30 | 405 | 0.12 | 38 | 288,083 |
| Holt auto | 1,200 | 30 | 680 | 0.28 | 54 | 28,901 |
| HoltWinters auto (p=12) | 1,200 | 30 | 1,570 | 0.36 | 177 | 29,026 |
| ETS(A,A,A) auto (p=12) | 1,200 | 30 | 61,355 | 0.62 | 204 | 19,478 |
| ETS(A,A,A) auto (p=12) | 12,000 | 30 | 581,194 | 0.57 | 204 | 192,278 |
| AutoETS (p=12) | 1,200 | 30 | 23,718 | 0.43 | 100-204 | n/a (not serializable) |

State size is O(seasonal period) (38-204 B) vs O(n) for the full model (2.9 KB at n=120 to
288 KB at n=12,000); update cost is O(k) and independent of n (0.05-0.6 us), 2-6 orders of
magnitude below a refit, which also re-estimates the parameters (update does not; when to refit
is the caller's decision, as the issue specifies).

## Deviations / findings
1. **Interval tolerance 1e-6 for SES/Holt/HW (not 1e-9).** Their `fit()` computes the residual SSE
   with `crate::simd::sum_of_squares`, which converts to **f32** before summing (`src/simd.rs`).
   The fit-on-(n+k) reference is therefore only f32-accurate (~1e-7 relative in sigma2), while
   `update()` extends the stored sum in f64. ETS uses an f64 running sum and matches to 1e-9.
   The f32 summation is pre-existing; it also means `residual_variance` / `interval_sigma2` of
   these models carry ~1e-7 relative error today. Not changed here (it would shift existing
   interval outputs); flagged for the maintainer.
2. **JSON round-trip is not bit-exact** (last-ulp differences from serde_json's default float
   parser); bincode is exact. Recommend bincode for the DuckDB state blob.
3. AutoETS `from_state` restores with the default `AutoETSConfig` (the config is only needed to
   re-run selection on a later `fit()`).

## Follow-ups (named in the issue, not implemented)
- Theta family (Theta / OptimizedTheta / DynamicOptimizedTheta), Croston / TSB, Naive /
  SeasonalNaive / window averages (SMA, SeasonalWindow). Naive/SeasonalNaive compute interval
  variance from the full residual vector inside `predict_with_intervals`, so they need the same
  running-sum refactor first.
- The DuckDB extension side (`combine_aggr` of two states) is out of scope for this crate: states
  are combined by restoring the stored state and calling `update()` with the new rows.

## Gates
- `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- `cargo check --target wasm32-unknown-unknown --features postprocess,js`: OK (same 6
  pre-existing `smart.rs` warnings as `origin/main`, none new).
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`: clean; module doctest passes.
- Full `cargo test --all-features --no-fail-fast`: 3779 passed, 0 failed (3766 on origin/main + 12 new integration tests + 1 module doctest).
