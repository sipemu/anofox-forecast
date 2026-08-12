# Codebase Concerns

**Analysis Date:** 2026-08-09

## Tech Debt

**Incomplete OLS Prediction Intervals:**
- Issue: Regression-based forecasting doesn't fully leverage prediction interval methods from `anofox-regression`
- Files: `src/models/regression.rs:485` (comment indicates TODO)
- Impact: Users cannot get prediction intervals from regression forecasters without workarounds
- Fix approach: Implement full OLS interval extraction using `anofox_regression::core::IntervalType` and expose via `predict_intervals()` or similar method

**Unused Smart Forecaster Infrastructure:**
- Issue: `SmartForecaster` (`src/models/smart.rs`) contains unused routing logic and dead code paths
- Files:
  - `src/models/smart.rs:69` - Unused constant `TREND_R2_TRIGGER`
  - `src/models/smart.rs:171` - Unused function `trend_r_squared()`
  - `src/models/smart.rs:225` - Unused function `is_autoets_favorable()`
- Impact: Code bloat, confusing API surface, maintenance burden
- Fix approach: Either integrate these functions into the router or remove them; clarify selection criteria in documentation

**Unused WASM Playground Imports:**
- Issue: Dead code in the WASM binding layer
- Files: `crates/anofox-forecast-js/src/laplace_playground.rs:15` - Unused `RecipeKind` import
- Impact: WASM bundle bloat, maintenance confusion
- Fix approach: Remove unused imports; audit other WASM module entry points for similar dead code

**Multiple Unused Inner Accessor Methods (WASM):**
- Issue: Several `inner()` methods in WASM forecaster wrappers are never called
- Files:
  - `crates/anofox-forecast-js/src/forecaster.rs:384` - `SESForecaster::inner()`
  - `crates/anofox-forecast-js/src/forecaster.rs:434` - `HoltForecaster::inner()`
  - `crates/anofox-forecast-js/src/forecaster.rs:517` - `HoltWintersForecaster::inner()`
  - `crates/anofox-forecast-js/src/forecaster.rs:1209` - `CrostonForecaster::inner()`
  - `crates/anofox-forecast-js/src/forecaster.rs:1261` - `TSBForecaster::inner()`
  - `crates/anofox-forecast-js/src/forecaster.rs:1315` - `ADIDAForecaster::inner()`
  - `crates/anofox-forecast-js/src/forecaster.rs:1369` - `IMAPAForecaster::inner()`
- Impact: WASM binary bloat (~7+ unused methods), confusion about public surface
- Fix approach: Mark as `#[doc(hidden)]` if used internally, remove entirely if truly dead

## Code Quality Issues

**High Volume of unwrap/expect Calls:**
- Issue: ~4,768 instances of `unwrap()`, `expect()`, and `panic!` across source
- Files: Concentrated in:
  - `src/changepoint/` - Cost functions, metrics, algorithms
  - `src/core/forecast.rs` - Forecast interval construction
  - `src/models/arima/model.rs` - Singular matrix detection, convergence failures
  - Test files (acceptable)
- Impact: Crashes possible on edge-case inputs; partial coverage means some panics are unreachable in practice
- Fix approach: Gradual conversion to `Result` types or documented invariants; prioritize public API boundaries and I/O paths first

**NaN Return Values as Error Signal:**
- Issue: Several modules use `f64::NAN` as an error signal for invalid computations
- Files: `src/features/entropy.rs` - Sample entropy, approximate entropy, permutation entropy all return NaN on insufficient data
- Impact: Silent error propagation; NaN comparisons are always false, breaking downstream logic
- Fix approach: Convert entropy functions to return `Result<f64, ForecastError>` or add explicit guards at call sites

**Unsafe Code in Anomaly Detection:**
- Issue: Unsafe block used for FFI call to special function
- Files: `src/anomaly/chi2.rs:111` - `unsafe { lgamma(x) }`
- Impact: Undefined behavior if FFI contract violated; hard to audit
- Fix approach: Add safety documentation explaining FFI invariants; consider pure-Rust alternative (statrs crate has Gamma PDF)

**Manual Clamp Pattern Instead of Built-in Method:**
- Issue: Clippy warning about manual clamp chain
- Files: `src/models/smart.rs:199` - Uses `.max(0.0).min(1.0)` instead of `.clamp(0.0, 1.0)`
- Impact: Slight inconsistency; NaN handling differs (`.clamp()` propagates NaN, manual chain doesn't)
- Fix approach: Replace with `.clamp()` after verifying NaN behavior is acceptable (R² should never be NaN)

**Unnecessary Mutability in WASM Kalman:**
- Issue: Mutable binding for immutable value
- Files: `crates/anofox-forecast-js/src/forecaster.rs:2141` - `let mut kf` declared but never mutated
- Impact: Misleading code; unused variable warning
- Fix approach: Remove `mut` keyword

## Known Bugs & Fixes

**GarchWrappedLeaf Variance Initialization (Fixed in v0.15.7):**
- Issue: First observation on billion-scale data caused 60-million-times mixture-σ inflation
- Files: `src/models/laplace/` (Laplace forecaster with GARCH leaf variants)
- Impact: Catastrophic WQL degradation on large-scale continuous data (300,000× on cif_2016)
- Status: **FIXED** in v0.15.7 — initial `var` now floors by `y²` to ensure O(1) standardized values
- Prevention: New regression test `leaf_init_pathology_sweep.rs` catches similar init bugs immediately

**Mixture Component Summation Precision (Fixed in v0.15.6):**
- Issue: Neumaier-compensated summation missing on `GaussianMixture` weight normalization
- Files: `src/models/laplace/`
- Impact: ~2-ULP drift on 20+ component mixtures, cascading into quantile-bisection differences
- Status: **FIXED** in v0.15.6 — integrated Neumaier fsum (matching CPython 3.12+'s `sum()`)
- Prevention: Added serde bit-identity tests

## Performance Bottlenecks

**Large Files with Potential Complexity:**
- File: `crates/anofox-forecast-js/src/forecaster.rs` (3,316 lines)
  - Concern: Monolithic WASM wrapper layer; hard to navigate, test, and optimize per-model
  - Mitigation: Well-documented entry points per forecaster type; browser benchmarks track overhead
  - Improvement path: Gradual refactor into per-model `forecaster_sas.rs`, `forecaster_arima.rs`, etc.

- File: `src/models/arima/model.rs` (very large)
  - Concern: Complex parameter estimation loop; convergence detection relies on many heuristics
  - Mitigation: Extensive test coverage on M-series data; calibrated against statsforecast reference
  - Improvement path: Extract convergence logic into `src/models/arima/convergence.rs` module

**SIMD Precision Tradeoff:**
- Issue: SIMD functions use f32 internally for ~7 decimal digits precision
- Files: `src/simd.rs:1-30`
- Impact: Fallback to f64 computation happens silently on SIMD errors; users unaware of precision loss
- Mitigation: Documentation clearly marks use cases (statistical aggregates, not p-value calculations)
- Fix approach: Add `#[inline(never)]` on fallback paths and measure SIMD success rate in benchmarks

## Fragile Areas

**Changepoint Detection Edge Cases:**
- Files: `src/changepoint/metrics.rs:165` - `.last().unwrap()` assumes non-empty breakpoint list
- Why fragile: Partial input validation; empty or malformed changepoint arrays crash
- Safe modification: Validate `bkps.len() > 0` in all public entry points; return `ForecastError::InsufficientData` instead
- Test coverage: Unit tests cover happy path; need adversarial tests for malformed input

**Cross-Validation Fold Selection:**
- Files: `src/utils/cross_validate()`, `src/models/cv_select.rs`
- Why fragile: Fold logic assumes `n >= 2 * min_train_size`; splits can fail silently if data is too short
- Safe modification: Add explicit validation in `CVConfig::validate()` with helpful error messages
- Test coverage: Gaps on boundary conditions (n == min_train_size, n < min_train_size)

**Seasonality MSTL Decomposition Failures:**
- Files: `src/seasonality/mstl.rs:48, 134, 140` - Multiple early returns with `None`
- Why fragile: No clear error cause; MSTL fails silently on certain data patterns (constant series, very short series, non-integer periods)
- Safe modification: Convert `Option<_>` returns to `Result<_, ForecastError>` with diagnostic messages
- Test coverage: Happy path only; need tests for edge cases (constant, missing seasonal pattern)

**Ensemble Voting with Empty Model Registry:**
- Files: `src/models/ensemble/model.rs:104, 111` - Returns `None` if no models fit
- Why fragile: Ensemble construction doesn't validate model count upfront
- Safe modification: Require minimum model count in `EnsembleForecaster::builder()` before `.build()`
- Test coverage: No test for zero-model ensemble

## Scaling Limits

**Panel (Global) Model Per-Series Overhead:**
- Current capacity: `GlobalETS` processes 30k+ series but each retains separate state
- Limit: Memory scales O(n_series × state_size); 10M series would exhaust typical RAM
- Scaling path: Implement true streaming global models with online EM for smoothing parameters

**Hierarchical Reconciliation Dense Covariance:**
- Current capacity: `MinTraceVariance` uses full N×N covariance matrix; safe for ~100k series
- Limit: Memory O(n_series²); 100k series = 10B elements = 80GB for f64 covariance
- Current mitigation: `MinTraceStruct` uses sparse summing matrix (O(n_series) memory)
- Scaling path: Make sparse reconciliation the default; dense path becomes opt-in for research

**ARIMA Convergence Loop:**
- Current capacity: Parameter search over (p, d, q) × (P, D, Q) space; fine for seasonal_period < 100
- Limit: Combinations explode for complex seasonality; can take minutes on high-frequency data
- Scaling path: Add adaptive grid search or use pre-filtering on periodogram to limit search space

## Dependencies at Risk

**Feature Gating & Optional Dependencies:**
- Risk: Several features are optional but widely assumed:
  - `postprocess` - OLS regression, conformal methods, bootstrap (default enabled)
  - `distributional` - Laplace forecaster, CRPS metrics (opt-in, required for examples)
  - `seasonal-detection` - fdars-core integration (opt-in, heavyweight)
  - `anomaly` - Mahalanobis detection (opt-in, requires distributional)
- Impact: Downstream crates may silently fail to compile if features not enabled
- Migration plan: Document feature matrix clearly; consider collapsing optional features into "lite" and "full" profiles

**Transitive Dependency on lbfgs (Optimization):**
- Risk: Older lbfgs v0.3 may have numerical stability issues; no recent updates
- Impact: ARIMA/ETS convergence may diverge or oscillate on ill-conditioned data
- Alternative: Migrate to `argmin` (used by `anofox-regression`) for unified optimization infrastructure

**Regression Dependency Version:**
- Risk: `anofox-regression v0.5.3` is external; breakage requires coordinated release
- Impact: Custom regression backends added to `SmartForecaster` depend on specific regression solver API
- Mitigation: Semantic versioning should protect minor updates; document minimum version in CHANGELOG

## Missing Critical Features

**Systematic Prediction Interval Gap:**
- Problem: No unified API for all models to expose prediction intervals
  - ARIMA / ETS have Gaussian prediction intervals
  - Laplace forecaster has full quantile functions
  - Bootstrap has empirical quantiles
  - Regression only has point forecasts (TODO noted)
- Blocks: Production deployments that require uncertainty quantification; risk assessment use cases
- Priority: High (mentioned in regression module as TODO)

**Streaming / Online Learning:**
- Problem: All models are batch-fit; no online update mechanism
- Blocks: Real-time forecasting pipelines, continuous model refinement, memory-constrained edge devices
- Priority: Medium (architectural; would require trait redesign)

**Exogenous Regressor Support Gaps:**
- Problem: MSTL with exogenous regressors is documented but implementation incomplete
- Files: Pre-regression exogenous support mentioned in MSTL, not fully wired through ensemble
- Blocks: Forecasting with external drivers (temperature, price, holiday calendars)
- Priority: Medium (feature parity with Prophet, statsmodels)

## Test Coverage Gaps

**Entropy Feature Insufficient Data Handling:**
- What's not tested: Behavior when n < embedding_dim for entropy features
- Files: `src/features/entropy.rs` returns NaN without explicit test
- Risk: Silent failure if used in feature-based pipelines (feature generator, ML pre-processing)
- Priority: Medium

**Changepoint Metric Input Validation:**
- What's not tested: Malformed breakpoint lists (unsorted, duplicates, out-of-bounds)
- Files: `src/changepoint/metrics.rs` assumes sorted monotone-increasing breakpoints
- Risk: Panic on user-provided metrics if data is dirty
- Priority: High (public API)

**Cross-Validation Boundary Conditions:**
- What's not tested: CV folds when n < 2 * min_train_size, n == min_train_size + 1
- Files: `src/utils/cross_validate()` and dependent models
- Risk: Off-by-one crashes or empty test folds
- Priority: High (used by AutoForecast in production)

**WASM Binding Integration:**
- What's not tested: WASM module behavior under extreme inputs (NaN, Inf, empty series in JS)
- Files: `crates/anofox-forecast-js/src/forecaster.rs` - Minimal error handling from JS perspective
- Risk: Browser crashes if JS calls WASM with invalid data; no graceful error messages
- Priority: Medium (user-facing library)

**Hierarchical Reconciliation with Missing Descendants:**
- What's not tested: `HierarchyTree` behavior when leaf forecasts are unavailable or fail
- Files: `src/hierarchy/` - Assumes all leaves produce forecasts
- Risk: Reconciliation panics instead of returning diagnostic error
- Priority: Medium

---

*Concerns audit: 2026-08-09*
