# Phase 3: Numerical Robustness & Coverage Baseline — Research

**Researched:** 2026-08-11
**Domain:** Rust edge-case testing, cargo-llvm-cov coverage measurement, proptest property-based testing, CI ratchet gates
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Coverage floor is a ratchet from the measured baseline** (baseline minus ~1% margin) — can only trend up; floor is bumped when coverage rises, never silently lowered.
- **Enforce in the EXISTING `ci.yml` coverage job** (`.github/workflows/ci.yml:146`). CI fails when coverage drops below the floor — hard gate matching ROADMAP SC-4. Do NOT create a new workflow; extend the existing job.
- **Metric: line coverage** (cargo-llvm-cov default) — simplest, matches the existing job.
- **Feature set for coverage: a fixed, documented set** (`--all-features` excluding wasm/js-only features) so the number is reproducible run-to-run. Document the exact invocation next to the baseline.
- **Coverage baseline committed to `.planning/baselines/coverage.json`** — consistent with `criterion.json / iai.json / dhat.json / statsforecast_reference.json / wasm_size.json`. Include a provenance block (tool version, feature set, commit) as the other baselines do.
- **One representative model per family** (ETS/AutoETS, ARIMA/AutoARIMA, Theta, TBATS, a baseline model, an intermittent model, plus VAR/Laplace where applicable) plus any model flagged fragile during the gap scan. Not all 30+ models exhaustively.
- **Edge-case input set = ROADMAP list as-is:** constant series, n=2, all-zeros/intermittent, NaN/Inf-containing, zero-length, extreme-scale inputs.
- **Assertions: assert the exact `ForecastError` variant where the outcome is deterministic** (`EmptyData`, `InsufficientData`, `MissingValues`); otherwise assert `is_err()` + no-panic. No test may trigger a panic.
- **Layout: a single new `tests/edge_case_robustness.rs`** integration suite.
- **Missing `validate_series_complete()` calls:** fix trivial/safe cases inline in this phase; file risky/complex cases as P1 improvement-backlog items.
- **Proptest coverage** of changepoint metrics, MSTL decomposition, CV boundary conditions — asserting no-panic and no-NaN invariants. Reuse existing proptest patterns.
- **Gap inventory: a committed markdown document** listing uncovered paths and assertion-free tests — file + function + missing invariant per row — structured for Phase 4 backlog consumption.

### Claude's Discretion
- Exact representative-model selection per family, the precise ratchet margin (~1%), and the gap-inventory file name/location within the phase directory.

### Deferred Ideas (OUT OF SCOPE)
- Exhaustive edge-case coverage of ALL 30+ models — representative-per-family chosen instead.
- Risky/complex missing-validation refactors — filed as P1 backlog items for Phase 4.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ROBUST-01 | Edge-case input suite: constant, n=2, all-zeros, NaN/Inf, zero-length, extreme-scale — correct `ForecastError`, no panic | §Edge-Case Testing Approach; §ForecastError Variants; §Representative Model Selection |
| ROBUST-02 | `fit()` audit: `validate_series_complete()` (or equivalent) runs before parameter estimation across model families | §validate_series_complete Audit; §Gap Files Identified |
| ROBUST-03 | Proptest: changepoint metrics, MSTL decomposition, CV boundary conditions — no-panic / no-NaN invariants | §Proptest Strategies |
| COVER-01 | Coverage baseline via cargo-llvm-cov, committed; CI floor enforced | §Coverage Baseline & Ratchet; §CI Extension Pattern |
| COVER-02 | Gap inventory: uncovered paths + assertion-free tests, filed as backlog candidates | §Gap Inventory Format; §Assertion Density Detection |
</phase_requirements>

---

## Summary

Phase 3 delivers three interlocking artifacts: (1) a structured edge-case test suite confirming every model family returns correct `ForecastError` values on malformed inputs, never panicking; (2) a line-coverage baseline captured via `cargo-llvm-cov` and committed to `.planning/baselines/coverage.json`, with a CI ratchet that fails the existing `coverage:` job when coverage drops below the recorded floor; (3) a gap inventory markdown document structured as Phase 4 backlog input.

The codebase already has a strong foundation: `validate_series_complete()` is called in 37 of ~41 `Forecaster`-implementing files, and proptest is already used in four test files (`property_tests.rs`, `interval_property_tests.rs`, `laplace_robustness.rs`, `laplace_component_robustness.rs`). The six model files that have `fit()` but lack `validate_series_complete()` calls are "global" or "internal" models with non-`TimeSeries` signatures (`global_ets.rs`, `global_croston.rs`, `global_theta.rs`, `var.rs`, `gpd_tails.rs`, `multiscale.rs`) — they have their own inline guards and are structurally different from the standard `Forecaster` trait implementations; this distinction is critical for the audit plan.

The `cargo-llvm-cov 0.8.4` binary is already installed. The existing `coverage:` job in `ci.yml` (line 146) uses `--all-features --lcov` — extending it with `--fail-under-lines <FLOOR>` is the only CI change needed. The `--all-features` flag is safe on the native target (the `js` feature gates only `getrandom/js` behind `cfg(target_arch = "wasm32")` and adds no native code).

**Primary recommendation:** Deliver the three artifacts in three sequenced plans: (1) edge-case suite + inline fixes, (2) coverage baseline capture + `scripts/update_coverage.sh` + CI ratchet extension, (3) proptest additions for fragile areas + gap inventory document.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Edge-case input validation | `tests/` integration tests | `src/models/` (fix missing guards) | Tests live outside `src/`; fixes touch guard sites in model `fit()` bodies (MEAS-04) |
| Coverage measurement | CI job extension (`ci.yml`) | `scripts/update_coverage.sh` (local capture) | CI enforces floor; local script captures baseline on quiet machine |
| Baseline storage | `.planning/baselines/coverage.json` | — | Established baseline store; CI reads, never writes (MEAS-01) |
| Proptest property tests | `tests/` integration tests | — | Same pattern as existing `property_tests.rs` |
| Gap inventory | `.planning/phases/03-.../03-GAP-INVENTORY.md` | — | Planning artifact consumed by Phase 4 backlog |
| `validate_series_complete` audit | `src/models/` — inspect each `fit()` | `tests/edge_case_robustness.rs` confirms | Grep + read confirms coverage; tests prove it works |

---

## Standard Stack

### Core (already present, no new dependencies)

| Tool / Library | Version | Purpose | Status |
|----------------|---------|---------|--------|
| `cargo-llvm-cov` | 0.8.4 | Line coverage measurement and floor enforcement | `[VERIFIED: cargo llvm-cov --version]` — already installed |
| `proptest` | 1.5 | Property-based testing | `[VERIFIED: Cargo.toml:66]` — already a dev-dependency |
| `cargo test` | stable toolchain | Integration test runner | `[VERIFIED: existing CI]` |
| `taiki-e/install-action@cargo-llvm-cov` | (pinned by CI) | CI install of cargo-llvm-cov | `[VERIFIED: ci.yml:156]` |

No new Cargo dependencies are required for Phase 3. All tooling is pre-existing.

### Supporting Tools

| Tool | Purpose | Invocation |
|------|---------|------------|
| `cargo llvm-cov --json --summary-only` | Machine-readable line% for baseline capture | `[VERIFIED: cargo llvm-cov --help]` |
| `cargo llvm-cov --fail-under-lines <N>` | CI ratchet enforcement | `[VERIFIED: cargo llvm-cov --help]` |
| `jq` | Extract `lines.percent` from JSON summary in `update_coverage.sh` | `[ASSUMED]` — standard on Linux CI runners |

---

## ForecastError Variants (Assertion Targets)

All variants are `[VERIFIED: src/error.rs:10-73]`. Verbatim enum definition:

```rust
pub enum ForecastError {
    EmptyData,
    InsufficientData { needed: usize, got: usize, hint: Option<String> },
    InvalidParameter(String),
    DimensionMismatch { expected: usize, got: usize },
    TimestampError(String),
    FitRequired { model: Option<String> },
    SubModelError { model_name: String, source: Box<ForecastError> },
    MissingValues,
    FrequencyInference(String),
    IndexOutOfBounds { index: usize, size: usize },
    ComputationError(String),
    ConvergenceFailure(String),
    SingularMatrix(String),
    SerializationError(String),
}
```

**Key deterministic variant mappings for edge-case assertions:**

| Input condition | Expected variant | Assertion idiom |
|----------------|-----------------|-----------------|
| `values.len() == 0` (empty `TimeSeries`) | `ForecastError::EmptyData` | `assert!(matches!(result, Err(ForecastError::EmptyData)))` |
| Series contains `f64::NAN` or `f64::INFINITY` | `ForecastError::MissingValues` | `assert!(matches!(result, Err(ForecastError::MissingValues)))` |
| Series length below model minimum (e.g., ARIMA(1,0,1) needs ≥ 3) | `ForecastError::InsufficientData { .. }` | `assert!(matches!(result, Err(ForecastError::InsufficientData { .. })))` |
| `predict()` before `fit()` | `ForecastError::FitRequired { .. }` | `assert!(matches!(result, Err(ForecastError::FitRequired { .. })))` |
| Model where outcome is uncertain | `is_err()` | `assert!(result.is_err(), "expected error for {input_desc}")` |

**Important:** `validate_series_complete()` `[VERIFIED: src/models/traits.rs:35-42]` only checks `has_missing_values()` (NaN or Inf). It does NOT check for empty data or minimum length — those are checked inline in each model's `fit()`. This means:
- NaN/Inf inputs consistently return `MissingValues` from any model that calls `validate_series_complete()`.
- Empty series: models must explicitly check `raw_values.is_empty()` — Naive does `[VERIFIED: src/models/baseline/naive.rs:102-103]`, but not all models do.
- Too-short series: models must check `n < minimum` inline — varies per model.

---

## validate_series_complete() Audit

### Validated: what the function covers

`[VERIFIED: src/models/traits.rs:35-42]` — verbatim:
```rust
pub fn validate_series_complete(series: &TimeSeries) -> Result<()> {
    if series.has_missing_values() {
        return Err(ForecastError::MissingValues);
    }
    Ok(())
}
```
`has_missing_values()` `[VERIFIED: src/core/time_series.rs:665-668]` returns true for any `f64::is_nan() || f64::is_infinite()` value. It does NOT check for zero-length.

### Files WITH validate_series_complete (37 files) — CONFIRMED COVERED

`[VERIFIED: grep src/models/]` — these 37 files call `validate_series_complete` and are covered for NaN/Inf validation:

```
arima/auto_arima.rs, arima/model.rs (×2 fit paths), auto_forecast.rs,
baseline/naive.rs, baseline/random_walk.rs, baseline/seasonal_naive.rs,
baseline/seasonal_window.rs, baseline/sma.rs,
cv_select.rs, ensemble/auto.rs, ensemble/model.rs,
exponential/auto_ets.rs, exponential/ets.rs, exponential/holt.rs,
exponential/holt_winters.rs, exponential/seasonal_es.rs, exponential/ses.rs,
garch.rs, intermittent/adida.rs, intermittent/croston.rs, intermittent/imapa.rs,
intermittent/tsb.rs, kalman_forecaster.rs, laplace/forecaster.rs, mfles.rs,
mstl_forecaster.rs, regression.rs, smart.rs,
tbats/auto.rs, tbats/model.rs,
theta/auto.rs, theta/dynamic.rs, theta/model.rs, theta/optimized.rs,
var_forecaster.rs
```

### Files WITH fit() but WITHOUT validate_series_complete (6 files) — AUDIT REQUIRED

`[VERIFIED: diff of fit()-bearing files vs validate_series_complete users]`

| File | fit() signature | Existing inline guard | Audit verdict |
|------|-----------------|-----------------------|---------------|
| `src/models/exponential/global_ets.rs` | `fit(&mut self, all_series: &[Vec<f64>])` | `[VERIFIED: src/models/exponential/global_ets.rs:81-96]` checks `InsufficientData` | NOT a `Forecaster` impl; takes raw `Vec<f64>` not `TimeSeries` — no `validate_series_complete` is appropriate. File as P1 if empty-series path unguarded. |
| `src/models/intermittent/global_croston.rs` | `fit(&mut self, all_series: &[Vec<f64>])` | `[VERIFIED: global_croston.rs:76-78]` checks `InsufficientData` | Same as above — non-`Forecaster` raw-vec API. |
| `src/models/theta/global_theta.rs` | `fit(&mut self, all_series: &[Vec<f64>])` | `[VERIFIED: global_theta.rs:66-68]` checks `InsufficientData` | Same pattern. |
| `src/models/var.rs` | `fit(&mut self, data: &[Vec<f64>])` | `[VERIFIED: src/models/var.rs:96-129]` checks `EmptyData` and `InsufficientData` | Non-`Forecaster`; raw matrix API. Guards are present. |
| `src/models/laplace/gpd_tails.rs` | `fn fit(&mut self, series: &TimeSeries)` | Delegates immediately to `self.inner.fit(series)?` which calls validate | Indirect — covered via delegation. Trivial inline add for clarity, or note as documentation-only fix. |
| `src/models/laplace/multiscale.rs` | `fn fit(&mut self, series: &TimeSeries)` | No explicit call; passes to `LaplaceForecaster::fit` which calls it | Indirect — covered via delegation. Same as gpd_tails. |

### Fix-vs-File Decision Rule

**Fix inline (this phase):** A `fit()` that implements `Forecaster` for a `TimeSeries` parameter AND directly does parameter estimation without calling `validate_series_complete()` first. Only `gpd_tails.rs` and `multiscale.rs` match this pattern; both delegate immediately to an inner forecaster that does call it, so the fix is a single-line insert: `validate_series_complete(series)?;` before the delegate call. These are trivial and safe.

**File as P1 (Phase 4 backlog):** Any non-`Forecaster` raw-vec API (`global_ets`, `global_croston`, `global_theta`, `var.rs`) that lacks NaN/Inf guards on the raw floats. These require a different validation approach (iterate each `Vec<f64>` and check for NaN/Inf) which is a non-trivial change to internal APIs. File as P1.

**Audit gate for the edge-case suite:** After adding the inline fixes, run the full edge-case suite. Any `fit()` call that panics instead of returning `Err` is a missed fix — flag for P1.

---

## Edge-Case Testing Approach

### Test File Layout

Single integration test: `tests/edge_case_robustness.rs`

Pattern mirrors `tests/laplace_robustness.rs` and `tests/interval_property_tests.rs`:
- Module-level `make_ts()` helper (already available in `tests/property_tests.rs` — replicate).
- One `#[test]` per (model, input-condition) combination named `{model_family}_{condition}`.
- No `unwrap()` on `fit()` or `predict()` calls — every call is either `assert!(result.is_err())` or uses `match`/`assert!(matches!(...))`.

### TimeSeries Construction for Each Edge Case

`[VERIFIED: src/core/time_series.rs:417-428]` — `TimeSeries::univariate(timestamps, values)` requires equal-length vecs; empty vecs produce a zero-length `TimeSeries` (no error from constructor).

```rust
fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    TimeSeries::univariate(stamps, values.to_vec()).unwrap()
    // Note: .unwrap() here is safe — we are constructing valid-shaped inputs.
    // The unwrap is on the constructor, not on fit/predict.
}

fn make_empty_ts() -> TimeSeries {
    // Zero-length: constructor succeeds; fit() should return EmptyData or InsufficientData.
    // Can't use univariate([],[]) with chrono — use make_ts(&[]) which creates 0-len.
    make_ts(&[])
}
```

Edge-case value patterns:
```rust
// Constant series (n=30)
let constant = vec![5.0f64; 30];

// n=2 (below most model minimums)
let n2 = vec![1.0, 2.0];

// All-zeros
let zeros = vec![0.0f64; 30];

// Intermittent (sparse non-zero)
let intermittent = {
    let mut v = vec![0.0f64; 30];
    v[2] = 3.0; v[7] = 1.0; v[15] = 4.0; v[22] = 2.0;
    v
};

// NaN-containing
let with_nan = { let mut v = vec![1.0f64; 30]; v[14] = f64::NAN; v };

// Inf-containing
let with_inf = { let mut v = vec![1.0f64; 30]; v[5] = f64::INFINITY; v };

// Extreme scale (large)
let extreme_large = vec![1e15f64; 30];

// Extreme scale (small)
let extreme_small = vec![1e-15f64; 30];
```

### Assertion Patterns

```rust
// Exact variant match (deterministic cases):
assert!(matches!(model.fit(&ts), Err(ForecastError::EmptyData)),
    "expected EmptyData for empty series");

assert!(matches!(model.fit(&ts), Err(ForecastError::MissingValues)),
    "expected MissingValues for NaN-containing series");

assert!(matches!(model.fit(&ts), Err(ForecastError::InsufficientData { .. })),
    "expected InsufficientData for n=2 series");

// Non-deterministic (model-dependent) error:
assert!(model.fit(&ts).is_err(),
    "expected error for {describe input}, got Ok");

// Ensure no panic and valid output (constant/extreme-scale that succeeds):
let result = model.fit(&ts);
// Check: either returns Ok (valid but potentially constant forecast) or Err (not panic).
if let Ok(()) = result {
    let fc = model.predict(1).unwrap();
    assert!(fc.primary().iter().all(|v| v.is_finite()),
        "non-finite forecast on constant series");
}
```

**Critical no-panic rule:** Tests must NOT use `.unwrap()` on operations expected to fail. In a `#[test]` function, an `unwrap()` panic propagates as a test failure with a panic message — hard to diagnose and looks like a code bug rather than a missing guard. Use `assert!(result.is_err())` instead.

### Representative Model Selection Per Family

`[VERIFIED: directory scan src/models/]` — one model per family that exercises the full `fit() → predict()` path:

| Family | Representative model | Rationale |
|--------|---------------------|-----------|
| Exponential (ETS) | `AutoETS` | Most complex; exercises full state-space search |
| ARIMA | `ARIMA::new(1, 0, 1)` | Standard configuration; covers differencing + estimation |
| Theta | `Theta::new()` | Base model; `AutoTheta` delegates to it |
| TBATS | `TBATS::new(12)` | Covers seasonal Fourier; period=12 for monthly data |
| Baseline | `Naive::new()` | Simplest; edge cases already partially tested |
| Intermittent | `Croston::new()` | Standard intermittent model |
| Laplace / distributional | `LaplaceForecaster::new().auto()` (behind `#[cfg(feature = "distributional")]`) | Laplace-specific robustness already in `laplace_robustness.rs`; include for NaN/extreme inputs |
| VAR | `VARForecaster` (if applicable to univariate edge inputs) | Multivariate — edge cases differ; include for the EmptyData / InsufficientData paths |
| MSTL | `MSTLForecaster::new(vec![12])` | MSTL decomposition path; flagged fragile |
| GARCH | `GARCH::new()` | Distributional; extreme-scale inputs are its primary risk |

**Additional fragile models to add during gap scan:** Any model where the gap inventory reveals uncovered `fit()` paths or where `grep` shows unguarded `.expect()` / `.unwrap()` in production code.

---

## Proptest Strategies

### Runtime Bounding

All proptest blocks that involve model fitting must use `ProptestConfig::with_cases(50)` or fewer to keep CI runtime under 30s per test file. `[VERIFIED: tests/property_tests.rs:69]` — existing tests already use `with_cases(50)`.

For MSTL and CV tests where the model fitting is heavier, use `with_cases(20)` and bound input lengths to 50–200 observations.

### Changepoint Metrics Property Tests

Target: `changepoint::metrics::{precision_recall, hausdorff, randindex}` `[VERIFIED: src/changepoint/metrics.rs:33,94,127]`

**Key invariants to assert:**
1. `precision_recall(bkps, bkps, margin)` returns `(precision=1.0, recall=1.0, f1=1.0)` for any valid `bkps`.
2. `hausdorff(a, a)` = 0.0 for any non-empty `a`.
3. `randindex(bkps, bkps, n)` = 1.0 for any valid `bkps` and matching `n`.
4. All functions return `Ok(result)` where result fields are finite — no NaN in metrics output.
5. Functions return `Err` (not panic) for malformed input (empty `bkps`, terminal breakpoint ≠ n).

**Note on `last().unwrap()` in metrics.rs:** `[VERIFIED: src/changepoint/metrics.rs:165]` — `bkps.last().unwrap()` at line 165 is guarded by the `bkps.is_empty()` check at line 160 which returns `Err` first. This is safe. The proptest generator must still avoid malformed inputs to test the happy path; test malformed inputs in separate unit tests (not proptest).

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn changepoint_precision_recall_self_match_is_perfect(
        n in 50usize..200,
        n_bkps in 1usize..5
    ) {
        // Generate valid breakpoints: strictly increasing, last == n
        let step = n / (n_bkps + 1);
        let mut bkps: Vec<usize> = (1..=n_bkps).map(|i| i * step).collect();
        bkps.push(n);
        let pr = precision_recall(&bkps, &bkps, 0).unwrap();
        prop_assert!((pr.precision - 1.0).abs() < 1e-10, "precision: {}", pr.precision);
        prop_assert!((pr.recall - 1.0).abs() < 1e-10, "recall: {}", pr.recall);
        prop_assert!(pr.f1.is_finite(), "f1 NaN");
    }

    #[test]
    fn changepoint_hausdorff_reflexive(n in 10usize..200, n_bkps in 1usize..4) {
        let step = n / (n_bkps + 1);
        let mut bkps: Vec<usize> = (1..=n_bkps).map(|i| i * step).collect();
        bkps.push(n);
        let h = hausdorff(&bkps, &bkps).unwrap();
        prop_assert!((h - 0.0).abs() < 1e-12, "hausdorff reflexive: {}", h);
    }

    #[test]
    fn changepoint_randindex_self_is_one(n in 10usize..200, n_bkps in 1usize..4) {
        let step = n / (n_bkps + 1);
        let mut bkps: Vec<usize> = (1..=n_bkps).map(|i| i * step).collect();
        bkps.push(n);
        let r = randindex(&bkps, &bkps, n).unwrap();
        prop_assert!((r - 1.0).abs() < 1e-10, "randindex self: {}", r);
        prop_assert!(r.is_finite());
    }
}
```

### MSTL Decomposition Property Tests

Target: `MSTL::decompose(&[f64])` returns `Option<MSTLResult>` — never panics, never produces NaN in components when it returns `Some`.

`[VERIFIED: src/seasonality/mstl.rs:130-141]` — `decompose` returns `None` when `n < 2 * max_period` or periods are empty. This is correct behavior to verify, not an error.

**Invariants:**
1. `decompose` never panics for any valid `&[f64]` (including all-zeros, constant, extreme-scale, random).
2. When it returns `Some(result)`, all fields (`trend`, `seasonal[i]`, `remainder`) are finite (no NaN/Inf).
3. `trend.len() == seasonal[i].len() == remainder.len() == n`.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn mstl_decompose_never_panics(
        values in prop::collection::vec(-1000.0f64..1000.0, 10..100),
        period in 2usize..8
    ) {
        let mstl = MSTL::new(vec![period]);
        // Must not panic — None is a valid return for short series
        let result = mstl.decompose(&values);
        if let Some(r) = result {
            // All components must be finite
            for v in r.trend.iter()
                .chain(r.seasonal.iter().flat_map(|s| s.iter()))
                .chain(r.remainder.iter())
            {
                prop_assert!(v.is_finite(), "NaN/Inf in MSTL component: {}", v);
            }
            // Length invariant
            prop_assert_eq!(r.trend.len(), values.len());
        }
    }

    #[test]
    fn mstl_decompose_constant_series(
        n in 10usize..100,
        c in -1000.0f64..1000.0,
        period in 2usize..8
    ) {
        let values = vec![c; n];
        let mstl = MSTL::new(vec![period]);
        let result = mstl.decompose(&values);
        if let Some(r) = result {
            prop_assert!(r.trend.iter().all(|v| v.is_finite()));
            prop_assert!(r.remainder.iter().all(|v| v.is_finite()));
        }
    }
}
```

### CV Boundary Conditions Property Tests

Target: `CvFoldGenerator::generate(series_len)` — never panics; returns `Ok(folds)` or `Err` (not both silent NaN and not panic).

`[VERIFIED: src/utils/cross_validation.rs:233-296]` — `generate()` returns `Err(ForecastError::InsufficientData)` when `series_len < horizon` at the earliest fold; returns `Err(ForecastError::InvalidParameter)` for policy violations.

**Invariants:**
1. `generate()` never panics for any input combination.
2. When it returns `Ok(folds)`, every fold satisfies `train_end <= test_start` and `test_end <= series_len`.
3. `fold.train_size() >= min_initial_window` for every fold in `Ok` result.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn cv_generate_never_panics(
        series_len in 0usize..500,
        horizon in 1usize..50,
        n_folds in 1usize..10,
        min_window in 2usize..50,
        gap in 0usize..10
    ) {
        let result = CvFoldGenerator::new()
            .n_folds(n_folds)
            .horizon(horizon)
            .min_initial_window(min_window)
            .gap(gap)
            .generate(series_len);
        // Must not panic — Ok or Err are both valid
        if let Ok(folds) = result {
            for fold in &folds {
                prop_assert!(fold.train_end <= fold.test_start,
                    "temporal leakage: train_end {} > test_start {}", fold.train_end, fold.test_start);
                prop_assert!(fold.test_end <= series_len,
                    "test_end {} > series_len {}", fold.test_end, series_len);
                prop_assert!(fold.train_size() >= min_window,
                    "train_size {} < min_window {}", fold.train_size(), min_window);
            }
        }
    }
}
```

### Proptest Seed Stability

**Pitfall:** proptest by default uses a random seed each run, which can cause flaky CI failures when a shrunk counterexample is found. To avoid this, do NOT set a fixed seed globally — proptest's regression corpus (`.proptest-regressions/`) handles discovered counterexamples automatically. Commit the `.proptest-regressions/` directory if it exists. If a proptest test intermittently fails CI, check whether a regression file was found but not committed.

For tests that are very slow to shrink (MSTL with complex inputs), limit shrinking with:
```rust
#![proptest_config(ProptestConfig { cases: 20, max_shrink_iters: 100, ..Default::default() })]
```

---

## Coverage Baseline & Ratchet Floor

### Exact Feature Set for Coverage

The CI `test:` job uses `--all-features` including `js`. The `js` feature only activates `getrandom/js` which is gated behind `cfg(target_arch = "wasm32")` `[VERIFIED: Cargo.toml:44-45]` and adds no native code. Running `cargo llvm-cov --all-features` on the native x86-64 target is confirmed safe — the existing CI `test:` job already does this `[VERIFIED: ci.yml:28]`.

**Canonical coverage invocation (for `scripts/update_coverage.sh`):**
```bash
cargo llvm-cov \
  --all-features \
  --json \
  --summary-only \
  --output-path /tmp/coverage_summary.json
```

This produces a compact JSON file. The `lines.percent` field under `totals.lines` holds the line coverage percentage:
```json
{
  "data": [{ "totals": { "lines": { "count": N, "covered": M, "percent": P.P } } }]
}
```

Extract with: `jq '.data[0].totals.lines.percent' /tmp/coverage_summary.json`

### coverage.json Baseline Schema

Following the established provenance-block convention `[VERIFIED: .planning/baselines/criterion.json, dhat.json, wasm_size.json]`:

```json
{
  "provenance": {
    "git_sha": "<git rev-parse HEAD>",
    "timestamp_iso": "<ISO 8601 UTC>",
    "rustc_version": "<rustc --version>",
    "cargo_llvm_cov_version": "0.8.4",
    "host_cpu": "<grep model name /proc/cpuinfo | head -1>",
    "host_os": "<uname -sr>",
    "active_features": "all (parallel + postprocess + forecastability + seasonal-detection + serde + distributional + anomaly + js)"
  },
  "coverage": {
    "metric": "line",
    "invocation": "cargo llvm-cov --all-features --json --summary-only",
    "lines_total": 0,
    "lines_covered": 0,
    "lines_percent": 0.0,
    "ratchet_floor_percent": 0.0
  }
}
```

`ratchet_floor_percent = lines_percent - 1.0` (the ~1% ratchet margin). This is the value that goes into `--fail-under-lines` in CI.

### scripts/update_coverage.sh

Following the pattern of `scripts/update_wasm_size.sh` and `scripts/update_criterion.sh`:

```bash
#!/usr/bin/env bash
# scripts/update_coverage.sh — capture line-coverage baseline into .planning/baselines/coverage.json
# Run on a local machine after Phase 3 tests are merged.
# Usage: bash scripts/update_coverage.sh
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUSTC=$(rustc --version)
COV_VER=$(cargo llvm-cov --version | head -1)
OS=$(uname -sr)
CPU=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")

cargo llvm-cov \
  --all-features \
  --json \
  --summary-only \
  --output-path /tmp/coverage_summary.json

LINES_PCT=$(jq '.data[0].totals.lines.percent' /tmp/coverage_summary.json)
LINES_COVERED=$(jq '.data[0].totals.lines.covered' /tmp/coverage_summary.json)
LINES_TOTAL=$(jq '.data[0].totals.lines.count' /tmp/coverage_summary.json)
FLOOR=$(echo "$LINES_PCT - 1.0" | bc -l | xargs printf "%.1f")

python3 - <<EOF > "$REPO_ROOT/.planning/baselines/coverage.json"
import json
data = {
  "provenance": {
    "git_sha": "$GIT_SHA",
    "timestamp_iso": "$TIMESTAMP",
    "rustc_version": "$RUSTC",
    "cargo_llvm_cov_version": "$COV_VER",
    "host_cpu": "$CPU",
    "host_os": "$OS",
    "active_features": "all (parallel + postprocess + forecastability + seasonal-detection + serde + distributional + anomaly + js)"
  },
  "coverage": {
    "metric": "line",
    "invocation": "cargo llvm-cov --all-features --json --summary-only",
    "lines_total": $LINES_TOTAL,
    "lines_covered": $LINES_COVERED,
    "lines_percent": $LINES_PCT,
    "ratchet_floor_percent": $FLOOR
  }
}
print(json.dumps(data, indent=2))
EOF
echo "Coverage baseline written: ${LINES_PCT}% (floor: ${FLOOR}%)"
```

### CI Extension Pattern

Extend the **existing `coverage:` job** in `.github/workflows/ci.yml` at line ~157 (between "Generate coverage" and "Upload to Codecov"). Add a new step that:
1. Generates the JSON summary (a separate, faster-to-parse run after the lcov run).
2. Reads `ratchet_floor_percent` from `.planning/baselines/coverage.json`.
3. Passes it to `--fail-under-lines`.

```yaml
# Existing step (unchanged):
- name: Generate coverage
  run: cargo llvm-cov --all-features --lcov --output-path lcov.info

# NEW step — floor ratchet enforcement:
- name: Enforce coverage floor
  run: |
    FLOOR=$(jq '.coverage.ratchet_floor_percent' .planning/baselines/coverage.json)
    echo "Coverage ratchet floor: ${FLOOR}%"
    cargo llvm-cov \
      --all-features \
      --json \
      --summary-only \
      --fail-under-lines "$FLOOR" \
      --output-path /tmp/coverage_check.json
    ACTUAL=$(jq '.data[0].totals.lines.percent' /tmp/coverage_check.json)
    echo "Actual coverage: ${ACTUAL}% (required: ${FLOOR}%)"
```

`--fail-under-lines` causes `cargo llvm-cov` to exit with code 1 if line coverage is below the threshold. `[VERIFIED: cargo llvm-cov --help]` — this exits the CI step with failure, which fails the job.

**Important:** The two `cargo llvm-cov` invocations (lcov + json) both instrument and run the test suite. This roughly doubles the coverage step runtime. If CI budget is a concern, replace the lcov invocation with a combined `--lcov --json` run writing both outputs simultaneously. `[ASSUMED]` — check whether `cargo llvm-cov` supports multiple output formats in one invocation (the `--output-path` flag may accept only one path; use `--output-dir` for multiple formats if available).

**Alternative (single-pass):** Use `cargo llvm-cov --all-features --lcov --output-path lcov.info --fail-under-lines "$FLOOR"`. This produces the lcov output AND enforces the floor in one pass. The exit code is 1 if the floor is not met; the lcov file is still written (for Codecov upload). This is the preferred single-pass approach.

### Ratchet Mechanics

The floor in `coverage.json` is written once by `scripts/update_coverage.sh` (manually, after Phase 3 tests land). The CI job reads it every run. To bump the floor after coverage improves: re-run `update_coverage.sh` and commit the new `coverage.json`. The floor never decreases automatically — only a deliberate maintainer commit changes it.

---

## Gap Inventory Format

### File Location

`.planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md`

This is a planning artifact, not a test file. It lives in the phase directory alongside `03-CONTEXT.md` and `03-RESEARCH.md`.

### Schema

```markdown
# Phase 3: Coverage Gap Inventory

**Generated:** {date}
**Coverage at time of inventory:** {N}%
**Tool:** cargo-llvm-cov {version} --all-features --json --summary-only

> Structured for Phase 4 backlog. Each row = one actionable improvement item.
> Priority: P1 = correctness risk (missing guard, silent NaN); P2 = coverage gap (low hit rate); P3 = assertion density (test exists but asserts nothing meaningful).

## Uncovered Paths

| # | File | Function / Line range | Missing invariant | Priority | Notes |
|---|------|-----------------------|-------------------|----------|-------|
| G-01 | `src/foo.rs` | `fn bar()` (lines 45–92) | No test covers the `n < minimum` branch | P1 | Identified by `--show-missing-lines` |
| G-02 | `src/baz.rs` | `fn qux()` (lines 12–18) | Error path on singular matrix | P2 | lcov: 0/6 lines hit |

## Assertion-Free Tests

> Tests that exist and pass but contain no `assert!`, `prop_assert!`, or `panic!` — they prove no-panic but assert no invariant.

| # | File | Test name | Missing assertion | Priority |
|---|------|-----------|-------------------|----------|
| A-01 | `tests/foo_test.rs` | `test_bar_happy_path` | No check on output values | P3 |

## Missing-Validation P1 Items

> fit() paths missing validate_series_complete() or equivalent, filed as P1 for Phase 4.

| # | File | Function | Risk | Recommended fix |
|---|------|----------|------|-----------------|
| V-01 | `src/models/exponential/global_ets.rs` | `fit(&[Vec<f64>])` | NaN propagates silently into fitted params | Add per-series NaN scan before estimation loop |
```

### Assertion Density Detection

Assertion density is the ratio of assertions to test functions. A test with no `assert!`, `prop_assert!`, or `debug_assert!` calls is assertion-free. Detect with:

```bash
# Count assertion-free tests in a file:
grep -n "fn test_\|#\[test\]" tests/some_test.rs | while read line_info; do
    # For each test, check if it contains an assert in the next N lines
    echo "$line_info"
done
```

Or more practically, use `cargo llvm-cov --show-missing-lines` to see which lines are uncovered — tests that are "covered" but have no assertions in them are identifiable by grep:

```bash
# Tests with no assert in their body:
python3 - <<'EOF'
import re, pathlib

def find_assertion_free_tests(path):
    src = pathlib.Path(path).read_text()
    test_blocks = re.split(r'#\[test\]', src)[1:]
    for i, block in enumerate(test_blocks):
        fn_match = re.search(r'fn\s+(\w+)\s*\(', block)
        if fn_match:
            name = fn_match.group(1)
            body_end = block.find('\n}')
            body = block[:body_end] if body_end > 0 else block[:500]
            if 'assert' not in body:
                print(f"  {path}: {name} — no assertion")

import sys
for f in sys.argv[1:]:
    find_assertion_free_tests(f)
EOF
tests/*.rs
```

The gap inventory is populated manually during the "gap scan" task in Plan 2 of Phase 3, using `cargo llvm-cov --all-features --text --output-path coverage_report.txt` to identify uncovered functions, combined with the assertion-free test scan above.

---

## Pitfalls & Landmines

### Pitfall 1: validate_series_complete() Does Not Check for Empty Data

**What goes wrong:** A test expects `EmptyData` from a model but `validate_series_complete()` only checks for NaN/Inf, not zero-length. The model reaches parameter estimation on an empty slice and either panics at `values[0]` or returns a wrong error type.

**Root cause:** `[VERIFIED: src/models/traits.rs:35-42]` — `validate_series_complete` calls only `has_missing_values()`. Empty data is checked inline by each model separately (e.g., Naive at `[VERIFIED: naive.rs:102-103]`). Not all models check it.

**How to avoid:** In the edge-case suite, test empty-series inputs against each representative model and document which variant is returned. If a model panics, fix with an inline `if series.is_empty() { return Err(ForecastError::EmptyData); }` before calling `validate_series_complete()`.

**Warning signs:** A test for empty series that produces a test failure message containing "index out of bounds" or "attempt to subtract with overflow" — these are panic messages, not `Err` returns.

### Pitfall 2: Coverage Nondeterminism from Feature-Gated Code

**What goes wrong:** `--all-features` on native activates `distributional`, `anomaly`, `postprocess`, etc. Code paths gated behind `#[cfg(feature = "...")]` are only exercised when the feature is on. Running coverage without `--all-features` produces a lower number because feature-gated branches are excluded entirely. Switching feature sets between baseline capture and CI comparison causes the floor to be invalid.

**How to avoid:** Lock the feature set in `coverage.json` under `coverage.invocation`. Always use `--all-features` for both baseline capture and CI enforcement. Document this explicitly in the provenance block.

**Warning signs:** Coverage percentage varies by ±5% between runs without code changes — almost always a feature-set inconsistency.

### Pitfall 3: cargo-llvm-cov Runtime Budget

**What goes wrong:** `cargo llvm-cov` instruments the test binary and runs the full test suite under LLVM coverage instrumentation. The full test suite takes significantly longer than `cargo test` alone — instrumentation adds ~30–50% overhead, and the coverage step already runs after the main `test:` job.

**Current CI state:** `[VERIFIED: ci.yml:157-159]` — the `coverage:` job runs `cargo llvm-cov --all-features --lcov --output-path lcov.info`. Adding a second `--fail-under-lines` run doubles the instrumentation cost.

**How to avoid:** Use the single-pass approach: `cargo llvm-cov --all-features --lcov --output-path lcov.info --fail-under-lines "$FLOOR"`. This generates lcov AND enforces the floor in one instrumented run. The `--fail-under-lines` flag causes an exit-code-1 failure if the floor is missed, but the lcov file is still written before exit (verified by the existing cargo-llvm-cov behavior `[ASSUMED]`).

**Warning signs:** The `coverage:` CI job times out. Solution: consider `--exclude` to skip the bench-harness crate if it adds large irrelevant code paths to instrumentation.

### Pitfall 4: Proptest Seed Flakiness

**What goes wrong:** proptest generates a random failing case, the shrinking produces a minimal example, but if the `.proptest-regressions/` directory is not committed, the next CI run uses a different seed and may not hit the same case.

**How to avoid:** Add `.proptest-regressions/` to git (do not gitignore it). Existing proptest files in this project do not reference a regressions directory — add it now when first introducing the file, not after a failure.

### Pitfall 5: WASM-Restricted Paths in Coverage

**What goes wrong:** Some code is gated `#[cfg(target_arch = "wasm32")]` (e.g., `getrandom/js` integration points). These lines will never be covered by `cargo llvm-cov` on the native x86-64 target. They appear as uncovered lines in the gap inventory. Filing them as "gaps" is incorrect — they are structurally untestable in the coverage context.

**How to avoid:** When building the gap inventory, annotate any uncovered line with `// WASM-only` if it is inside a `#[cfg(target_arch = "wasm32")]` block. These are excluded from P1/P2 backlog. WASM paths are tested by the `wasm-test:` CI job, not by llvm-cov.

### Pitfall 6: MSTL Returns Option Not Result

**What goes wrong:** `MSTL::decompose()` returns `Option<MSTLResult>` `[VERIFIED: src/seasonality/mstl.rs:130]`, not `Result`. Proptest code written to match `Result` patterns will fail to compile. The `None` return is the correct "series too short" signal — it is NOT an error from the library's perspective.

**How to avoid:** Proptest for MSTL must handle `Option` — use `if let Some(r) = result { ... }` patterns. A `None` return for a short series is a valid, correct outcome and should not be asserted as a failure.

### Pitfall 7: Global Model fit() Takes &[Vec<f64>], Not &TimeSeries

**What goes wrong:** Attempting to write edge-case tests for `global_ets::GlobalETS::fit()` using `TimeSeries` inputs will fail because the function signature is `fit(&mut self, all_series: &[Vec<f64>])` `[VERIFIED: src/models/exponential/global_ets.rs:81]`. These are panel/batch models, not standard `Forecaster` implementors.

**How to avoid:** Edge-case tests for global models use raw `Vec<Vec<f64>>` inputs. They are secondary targets in Phase 3 (the representative-per-family selection uses standard `Forecaster` models). Global models go in the gap inventory as "different API surface" rather than in `edge_case_robustness.rs`.

---

## Architecture Patterns

### System Architecture Diagram

```
Edge-Case Input Suite (tests/edge_case_robustness.rs)
  │
  ├─ [per model family] fit(malformed_ts) ──→ assert variant / is_err()
  │                                               ↓
  │                          src/models/{family}/model.rs
  │                          validate_series_complete()  ←── src/models/traits.rs
  │
  ├─ [proptest block] CvFoldGenerator::generate(N)
  │                         ↓
  │                   assert no panic, temporal integrity
  │
  └─ [proptest block] MSTL::decompose(&values)
                            ↓
                      assert no panic, finite components

Coverage Measurement (local machine)
  scripts/update_coverage.sh
  ├─ cargo llvm-cov --all-features --json --summary-only
  ├─ jq extract lines_percent
  └─ write .planning/baselines/coverage.json (provenance block)

CI Ratchet Enforcement (.github/workflows/ci.yml, existing coverage: job)
  ├─ [existing] cargo llvm-cov --all-features --lcov --output-path lcov.info
  ├─ [new step] read ratchet_floor_percent from coverage.json
  ├─ [new step] cargo llvm-cov --all-features ... --fail-under-lines $FLOOR
  └─ exit 1 if below floor → PR blocked

Gap Inventory (planning artifact)
  .planning/phases/03-.../03-GAP-INVENTORY.md
  ├─ cargo llvm-cov --show-missing-lines (identify uncovered paths)
  ├─ assertion-free test scan (grep / python script)
  └─ P1/P2/P3 triage → Phase 4 backlog input
```

### Recommended Project Structure (new artifacts only)

```
tests/
└── edge_case_robustness.rs       # NEW — single integration file
scripts/
└── update_coverage.sh            # NEW — local baseline capture
.planning/baselines/
└── coverage.json                 # NEW — committed baseline + floor
.planning/phases/03-.../
├── 03-CONTEXT.md                 # existing
├── 03-RESEARCH.md                # this file
└── 03-GAP-INVENTORY.md           # NEW — gap inventory (planning artifact)
.github/workflows/
└── ci.yml                        # MODIFIED — extend existing coverage: job
src/models/laplace/
├── gpd_tails.rs                  # MODIFIED — add validate_series_complete (trivial)
└── multiscale.rs                 # MODIFIED — add validate_series_complete (trivial)
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Line coverage percentage extraction | Custom LLVM report parser | `cargo llvm-cov --json --summary-only` + `jq` | `[VERIFIED: cargo llvm-cov --help]` — built-in JSON output; parsing LLVM's raw `.profraw` is complex and brittle |
| Coverage floor enforcement | Custom shell script comparing floats | `--fail-under-lines <N>` flag | `[VERIFIED: cargo llvm-cov --help]` — native flag; handles float comparison and exit code |
| Property-based random input generation | LCG or hand-coded random walks | `proptest` strategies | `[VERIFIED: Cargo.toml:66]` — already present; shrinking, regression corpus, and determinism are built in |
| Variant matching on `ForecastError` | `format!("{}", err).contains(...)` string matching | `matches!(result, Err(ForecastError::VariantName { .. }))` | Structural match is refactor-safe; string matching breaks on display text changes |
| Test isolation for proptest | Custom seed management | `.proptest-regressions/` directory | proptest's native regression corpus is automatic |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `cargo test` + `proptest 1.5` |
| Config file | none (project uses default `cargo test` configuration) |
| Quick run command | `cargo test --test edge_case_robustness --all-features` |
| Full suite command | `cargo test --all-features` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ROBUST-01 | Edge-case inputs return correct `ForecastError`, no panic | integration | `cargo test --test edge_case_robustness --all-features` | No — Wave 1 |
| ROBUST-02 | `validate_series_complete()` audit — all `fit()` paths confirmed | audit (grep) + integration | `grep validate_series_complete src/models/**/*.rs` | N/A — audit task |
| ROBUST-03 | Proptest: changepoint/MSTL/CV — no panic, no NaN | property | `cargo test --test property_robustness --all-features` | No — Wave 2 |
| COVER-01 | Coverage baseline committed; CI floor enforced | CI gate | `cargo llvm-cov --all-features --fail-under-lines $FLOOR` | No — Wave 2/3 |
| COVER-02 | Gap inventory committed | planning artifact | manual review | No — Wave 3 |

### Sampling Rate

- **Per task commit:** `cargo test --test edge_case_robustness --all-features` (fast, ~10s)
- **Per wave merge:** `cargo test --all-features` (full suite, ~2 min)
- **Phase gate:** Full suite green + coverage floor green before `/gsd-verify-work`

### Wave 0 Gaps

- `tests/edge_case_robustness.rs` — covers ROBUST-01
- `scripts/update_coverage.sh` — covers COVER-01 baseline capture
- `.planning/baselines/coverage.json` placeholder — covers COVER-01 schema
- `.planning/phases/03-.../03-GAP-INVENTORY.md` template — covers COVER-02

---

## Security Domain

> `security_enforcement` is enabled. Phase 3 adds test code only — no new src/ changes beyond trivial guard inserts.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | Yes (this is the phase goal) | `validate_series_complete()` + inline guards per model |
| V6 Cryptography | No | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN/Inf propagation in numerical computation | Tampering | `validate_series_complete()` at every `fit()` entry point |
| Integer underflow/overflow in index arithmetic | Tampering | Checked arithmetic in CV fold generation; proptest verifies no panic |
| Panic on malformed input (denial-of-service in library) | Denial of Service | Edge-case suite asserts `is_err()` not panic for all malformed inputs |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cargo llvm-cov --lcov --fail-under-lines $FLOOR` writes lcov file AND exits with code 1 when floor is missed (both happen, not just exit) | CI Extension Pattern | If lcov is not written on failure, Codecov upload step is skipped — minor; floor still enforced |
| A2 | `jq` is available on ubuntu-latest GitHub Actions runners | scripts/update_coverage.sh | Coverage baseline script fails; use `python3 -c 'import json,sys; ...'` as fallback |
| A3 | Global model files (`global_ets.rs` etc.) have no NaN-propagation risk because input is `&[Vec<f64>]` already validated by callers | validate_series_complete audit | If callers can pass NaN floats, silent NaN in fitted params — P1 risk, file for Phase 4 |
| A4 | `proptest-regressions/` directory does not exist and needs to be created | Proptest pitfall | No risk — proptest creates it on first failing case |
| A5 | `bc -l` is available in the `update_coverage.sh` script for float arithmetic | scripts/update_coverage.sh | Script fails; use `python3 -c 'print(round(float("$LINES_PCT") - 1.0, 1))'` instead |

---

## Open Questions

1. **Single-pass lcov + floor enforcement**
   - What we know: `cargo llvm-cov --all-features --lcov --output-path lcov.info --fail-under-lines "$FLOOR"` should produce both outputs in one instrumented run.
   - What's unclear: Whether `cargo-llvm-cov 0.8.4` supports combining `--lcov` with `--fail-under-lines` in a single invocation without requiring `--json` too.
   - Recommendation: Test locally before committing the ci.yml change. If it fails, use two sequential calls (lcov first, json+floor second).

2. **Coverage workspace scope**
   - What we know: The workspace includes `crates/anofox-bench-harness` and `crates/anofox-forecast-js` — these may be included in `--all-features` coverage measurement.
   - What's unclear: Whether bench-harness code (which has little meaningful logic) inflates or deflates the line coverage percentage.
   - Recommendation: Run `cargo llvm-cov --all-features --json --summary-only` before and after `--package anofox-forecast` to see if scoping to the main crate changes the number significantly. Document the chosen scope in `coverage.json`.

3. **Exact minimum series lengths per model**
   - What we know: ARIMA(1,0,1) needs at least 3 observations; most models need 2–10; TBATS with period 12 needs ≥ 24.
   - What's unclear: Exact minimums for AutoARIMA, AutoETS, and MFLES — these may vary based on parameter ranges.
   - Recommendation: Discover these empirically during the edge-case suite writing (try n=2 and observe the error; use that as the assertion value).

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cargo-llvm-cov` | COVER-01 | Yes | 0.8.4 | None — must be installed |
| `llvm-tools-preview` component | cargo-llvm-cov | Yes (stable toolchain) | — | `rustup component add llvm-tools-preview` |
| `jq` | `update_coverage.sh` | `[ASSUMED]` | — | `python3 -c 'import json...'` inline |
| `proptest` | ROBUST-03 | Yes | 1.5 | None — already in Cargo.toml |
| `bc` | `update_coverage.sh` float arithmetic | `[ASSUMED]` | — | `python3` inline calculation |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `tarpaulin` (grcov predecessor) | `cargo-llvm-cov` (LLVM instrumentation) | ~2022 | More accurate line counts; better feature-gated code handling |
| Fixed coverage thresholds | Ratchet-floor (monotonically increasing) | This phase | Prevents slow erosion; aligns with project's "measurement-first" philosophy |
| Proptest cases=1000 (slow) | `ProptestConfig::with_cases(50)` | Established pattern in this codebase | Keeps CI runtime bounded |

**Deprecated/outdated:**
- `cargo-tarpaulin`: Not used in this project; mentioned only for context. `cargo-llvm-cov` is the current standard.
- Fixed `#[should_panic]` tests for error assertions: Use `matches!(result, Err(SpecificVariant { .. }))` instead — more informative on failure.

---

## Sources

### Primary (HIGH confidence)

- `src/error.rs:10-73` — ForecastError enum verbatim (assertion targets)
- `src/models/traits.rs:35-42` — validate_series_complete implementation
- `src/core/time_series.rs:417-428, 665-668` — TimeSeries constructor and has_missing_values
- `src/models/baseline/naive.rs:100-103` — representative empty-data guard pattern
- `src/models/exponential/global_ets.rs:81-96` — global model fit() signature
- `src/models/laplace/gpd_tails.rs:452-499` — delegation fit() pattern
- `src/models/laplace/multiscale.rs:188-250` — delegation fit() pattern
- `src/seasonality/mstl.rs:130-141` — MSTL::decompose signature and early returns
- `src/changepoint/metrics.rs:33, 94, 127, 160-168` — public API and edge guards
- `src/utils/cross_validation.rs:233-296` — CvFoldGenerator::generate error paths
- `.github/workflows/ci.yml:146-164` — existing coverage job (extension target)
- `.planning/baselines/criterion.json, dhat.json, wasm_size.json` — provenance block schema
- `Cargo.toml:31-45, 66` — feature definitions and proptest dev-dep
- `cargo llvm-cov --help` — `--fail-under-lines`, `--json`, `--summary-only` flags
- `tests/property_tests.rs:1-130` — existing proptest patterns (cases=50, make_ts helper)
- `tests/laplace_robustness.rs:1-86` — adversarial soak test pattern

### Secondary (MEDIUM confidence)

- Existing proptest patterns in this codebase — representative of Rust community proptest usage

### Tertiary (LOW confidence / ASSUMED)

- `jq` and `bc` availability on ubuntu-latest runners (A2, A5)
- single-pass `--lcov --fail-under-lines` behavior in cargo-llvm-cov 0.8.4 (A1)

---

## Metadata

**Confidence breakdown:**
- ForecastError variants: HIGH — read source verbatim
- validate_series_complete audit: HIGH — grep confirmed all 37 call sites and all 6 non-calling files
- Edge-case testing patterns: HIGH — existing tests in this codebase provide canonical examples
- cargo-llvm-cov invocation: HIGH — `--version` confirms 0.8.4; `--help` confirms all flags
- CI extension approach: HIGH — existing job structure confirmed; extension is additive
- Proptest strategies: HIGH — code patterns drawn from existing working tests in this codebase
- Gap inventory format: HIGH — follows established planning artifact conventions
- `jq`/`bc` in scripts: LOW — not verified on CI runners, flagged ASSUMED

**Research date:** 2026-08-11
**Valid until:** 2026-11-11 (stable toolchain APIs; cargo-llvm-cov has been stable for 2+ years)
