# Phase 2: Accuracy Harness & Statistical Methodology - Pattern Map

**Mapped:** 2026-08-10
**Files analyzed:** 8 new/modified files
**Analogs found:** 7 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/anofox-bench-harness/src/loader.rs` | utility | file-I/O | `examples/skaters_m3_monthly_benchmark.rs` (parse_tsf fn) | role-match |
| `crates/anofox-bench-harness/src/naive2.rs` | model | request-response | `src/models/baseline/` (Naive + SeasonalNaive) | partial-match |
| `crates/anofox-bench-harness/src/dm_test.rs` | utility | transform | `src/utils/metrics.rs` (standalone fn pattern) | role-match |
| `crates/anofox-bench-harness/tests/accuracy.rs` | test | request-response | `tests/m4_daily_accuracy_regression.rs` | exact |
| `crates/anofox-bench-harness/src/lib.rs` | config | — | `crates/anofox-bench-harness/src/lib.rs` (extend) | exact |
| `src/utils/metrics.rs` (modify: D-03 fix) | utility | transform | self | exact |
| `.github/workflows/accuracy.yml` | config | — | `.github/workflows/bench.yml` | role-match |
| `.planning/baselines/accuracy.json` | config | — | `.planning/baselines/` existing JSON files | role-match |

---

## Pattern Assignments

### `crates/anofox-bench-harness/src/loader.rs` (utility, file-I/O)

**Analog:** `examples/skaters_m3_monthly_benchmark.rs` (lines 52–87) and `examples/m4_hourly_diagnostic.rs`

**Imports pattern** — follow harness convention (no `use std::fs` from crate root; bring in explicitly):
```rust
use std::fs;
use std::path::Path;
```

**Core TSF parse pattern** (from `examples/skaters_m3_monthly_benchmark.rs` lines 52–87):
```rust
// Latin-1 decode: MANDATORY — Monash .tsf is not UTF-8.
// std::fs::read_to_string will panic on accented category labels.
let bytes = fs::read(path).expect("read tsf");
let content: String = bytes.iter().map(|&b| b as char).collect();

let mut in_data = false;
for line in content.lines() {
    if !in_data {
        if line.trim_start().starts_with("@data") {
            in_data = true;
        }
        continue;
    }
    // Data line: id:start_ts:v1,v2,v3,...
    let mut parts = line.splitn(3, ':');
    let id = match parts.next() { Some(s) => s.to_string(), None => continue };
    let _start_ts = parts.next();
    let vals_str = match parts.next() { Some(s) => s, None => continue };
    let values: Vec<f64> = vals_str
        .split(',')
        .filter_map(|tok| tok.trim().parse::<f64>().ok())
        .collect();
    if values.len() > HORIZON + 6 {
        series.push((id, values));
    }
}
```

**Metadata extraction extension** — add before `@data` branch (pattern derived from RESEARCH.md Pattern 1):
```rust
// Before `if line.trim_start().starts_with("@data")`:
if let Some(val) = trimmed.strip_prefix("@frequency") {
    frequency = val.trim().to_string();
} else if let Some(val) = trimmed.strip_prefix("@horizon") {
    horizon = val.trim().parse().unwrap_or(0);
}
```

**Env-gate pattern** — follow existing test convention from `tests/m4_daily_accuracy_regression.rs`:
```rust
// At top of any function that reads the corpus:
let dataset_dir = match std::env::var("ANOFOX_DATASET_DIR") {
    Ok(d) => d,
    Err(_) => {
        eprintln!("ANOFOX_DATASET_DIR not set — skipping dataset-dependent test");
        return;
    }
};
```

**Security: finite-value guard** — add after parse (RESEARCH.md security pattern):
```rust
let values: Vec<f64> = vals_str
    .split(',')
    .filter_map(|tok| {
        let v: f64 = tok.trim().parse().ok()?;
        v.is_finite().then_some(v)  // drop NaN/Inf from malformed TSF
    })
    .collect();
```

---

### `crates/anofox-bench-harness/src/naive2.rs` (model, request-response)

**Analog:** `src/models/baseline/` structs (`Naive`, `SeasonalNaive`) — used as inner models. No exact analog for the struct itself.

**Module doc pattern** — follow `src/models/baseline/` convention:
```rust
//! Naive2 baseline model for competition accuracy evaluation.
//!
//! A harness-only model (D-07): not part of the public `anofox-forecast` API.
//! Composes [`Naive`] and [`SeasonalNaive`] from the library, gated by a
//! 90%-confidence autocorrelation test at the seasonal lag (D-08).
```

**Imports pattern** — access library types via the harness dependency:
```rust
use anofox_forecast::models::baseline::{Naive, SeasonalNaive};
use anofox_forecast::models::Forecaster;
use anofox_forecast::core::TimeSeries;
use anofox_forecast::error::ForecastError;
use chrono::{Duration, TimeZone, Utc};
```

**Struct + enum pattern** — mirror baseline model style (PascalCase, private inner state):
```rust
pub struct Naive2 {
    seasonal_period: usize,
    inner: Naive2Inner,
}

enum Naive2Inner {
    Seasonal(SeasonalNaive),
    Random(Naive),
}
```

**Constructor pattern** — follow `Naive::new()` / `SeasonalNaive::new(period)` convention; no `Default` impl (explicit period required):
```rust
impl Naive2 {
    pub fn new(seasonal_period: usize) -> Self {
        Self { seasonal_period, inner: Naive2Inner::Random(Naive::new()) }
    }
}
```

**ACF-at-lag helper** (pure function, no struct state — follows metric standalone fn pattern from `src/utils/metrics.rs`):
```rust
fn acf_at_lag(series: &[f64], lag: usize) -> f64 {
    let n = series.len();
    if n <= lag { return 0.0; }
    let mean = series.iter().sum::<f64>() / n as f64;
    let numer: f64 = (lag..n)
        .map(|t| (series[t] - mean) * (series[t - lag] - mean))
        .sum();
    let denom: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum();
    if denom == 0.0 { 0.0 } else { numer / denom }
}
```

**TimeSeries construction helper** — follow `tests/m4_daily_accuracy_regression.rs` lines 142–145:
```rust
fn make_ts_from_slice(values: &[f64]) -> Result<TimeSeries, ForecastError> {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec())
}
```

---

### `crates/anofox-bench-harness/src/dm_test.rs` (utility, transform)

**Analog:** `src/utils/metrics.rs` standalone function pattern (lines 172–357) — module-level pure functions, no struct.

**Module doc pattern:**
```rust
//! Diebold-Mariano test with HLN small-sample correction and HAC variance.
//!
//! Implements squared-error loss + Harvey–Leybourne–Newbold (HLN) correction
//! per D-09. Reference: real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/
```

**Standalone function pattern** — follow `src/utils/metrics.rs` style (pub fn, slice inputs, documented return):
```rust
/// Run the Diebold-Mariano test (HLN + HAC variant, D-09).
///
/// # Arguments
/// * `e1` - Forecast errors from model 1 (actual − predicted₁); length T
/// * `e2` - Forecast errors from model 2 (actual − predicted₂); length T
/// * `h`  - Forecast horizon (controls HAC lag window and HLN correction)
///
/// # Returns
/// `(p_value, reject_h0)` — two-sided p-value under normal approximation,
/// and whether H₀ (equal predictive accuracy) is rejected at α = 0.05.
pub fn diebold_mariano_hln(e1: &[f64], e2: &[f64], h: usize) -> (f64, bool) {
    // ...
}
```

**Error guard pattern** — follow metrics.rs NaN-return-on-bad-input:
```rust
if e1.len() != e2.len() || e1.is_empty() {
    return (f64::NAN, false);
}
```

**Normal CDF helper** — use `std` only (no statrs in harness dev-deps; see RESEARCH.md open question 3):
```rust
/// Approximate standard normal CDF via Abramowitz-Stegun (max error 7.5e-8).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530 + t * (-0.356563782
        + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * poly;
    if x >= 0.0 { cdf } else { 1.0 - cdf }
}
```

---

### `crates/anofox-bench-harness/tests/accuracy.rs` (test, request-response)

**Analog:** `tests/m4_daily_accuracy_regression.rs` (lines 1–230) — exact match for pattern, structure, and assertion style.

**File-level doc comment pattern** (from `tests/m4_daily_accuracy_regression.rs` lines 1–45):
```rust
//! M3 accuracy harness: Naive2 + AutoETS over M3 Yearly/Quarterly/Monthly.
//!
//! ## Data
//! Reads Monash TSF files from `$ANOFOX_DATASET_DIR`. Test skips when the
//! env var is not set (ACCUR-01).
//!
//! ## Reference baseline
//! Validates AutoETS M3-monthly MASE ≈ 0.93 ± 0.02 against
//! `.planning/baselines/statsforecast_reference.json` (ACCUR-08).
//!
//! ## Assertions
//! 1. Temporal integrity: every fold satisfies `train_end <= test_start` (ACCUR-02).
//! 2. No NaN in per-frequency aggregates (ACCUR-03).
//! 3. Per-frequency stratification: Yearly/Quarterly/Monthly reported separately (ACCUR-07).
//! 4. AutoETS M3-monthly MASE anchor (ACCUR-08).
```

**Imports pattern** (from `tests/m4_daily_accuracy_regression.rs` lines 46–52 + `tests/full_statsforecast_comparison.rs` lines 4–16):
```rust
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
use anofox_bench_harness::loader::{load_m3, DatasetSeries};
use anofox_bench_harness::naive2::Naive2;
use anofox_bench_harness::dm_test::diebold_mariano_hln;
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;
```

**Reference loading pattern** (from `tests/full_statsforecast_comparison.rs` lines 37–42):
```rust
fn load_reference() -> serde_json::Value {
    let path = ".planning/baselines/statsforecast_reference.json";
    let content = std::fs::read_to_string(path)
        .expect("statsforecast_reference.json missing — run validation/run_statsforecast.py first");
    serde_json::from_str(&content).unwrap()
}
```

**Env-gate + per-frequency loop pattern** (from `tests/m4_daily_accuracy_regression.rs` lines 113–167):
```rust
fn run_accuracy_harness() -> HashMap<String, FrequencyResult> {
    let dataset_dir = match std::env::var("ANOFOX_DATASET_DIR") {
        Ok(d) => d,
        Err(_) => return HashMap::new(),  // caller detects empty map → skip
    };
    // ... load + loop
}
```

**Temporal assertion pattern** (per ACCUR-02 — assert on every fold):
```rust
for fold in &folds {
    assert!(
        fold.train_end <= fold.test_start,
        "temporal integrity violation: train_end={} > test_start={}",
        fold.train_end, fold.test_start
    );
}
```

**MASE computation pattern** (from `examples/skaters_m3_monthly_benchmark.rs` lines 313–322 — training-denominator form):
```rust
fn mase_scale(train: &[f64], period: usize) -> f64 {
    if train.len() <= period { return 1.0; }
    let n = train.len() - period;
    let sum: f64 = (period..train.len())
        .map(|i| (train[i] - train[i - period]).abs())
        .sum();
    (sum / n as f64).max(1e-9)
}
```

**Assertion style** (from `tests/m4_daily_accuracy_regression.rs` lines 197–230):
```rust
#[test]
fn accur08_anchor_m3_monthly_autoets() {
    let results = run_accuracy_harness();
    if results.is_empty() {
        eprintln!("ANOFOX_DATASET_DIR not set — skipping ACCUR-08");
        return;
    }
    let monthly = results.get("monthly").expect("monthly result missing");
    let mase = monthly.autoets_mase;
    assert!(
        mase.is_finite(),
        "AutoETS M3-monthly MASE is NaN/Inf — check D-03 fix and training denominator"
    );
    assert!(
        (mase - 0.93).abs() <= 0.02,
        "AutoETS M3-monthly MASE={:.4} outside ±0.02 of 0.93 anchor (ACCUR-08)",
        mase
    );
}
```

**NaN-guard in aggregate** (ACCUR-03 harness layer — apply before summing):
```rust
let finite_mases: Vec<f64> = per_series_mase
    .iter()
    .copied()
    .filter(|m| m.is_finite())
    .collect();
let skipped = per_series_mase.len() - finite_mases.len();
if skipped > 0 {
    eprintln!("WARNING: {} series had non-finite MASE and were excluded from aggregate", skipped);
}
let mean_mase = finite_mases.iter().sum::<f64>() / finite_mases.len() as f64;
```

---

### `crates/anofox-bench-harness/src/lib.rs` (extend existing)

**Analog:** Current `crates/anofox-bench-harness/src/lib.rs` (lines 1–8) — extend by adding new module declarations.

**Current file** (lines 1–8):
```rust
//! Shared measurement harness for anofox-forecast performance tracking.
//!
//! Provides:
//! - `baseline`: D-02 provenance schema structs for JSON baseline files
//! - `fixtures`: deterministic seeded time series for reproducible benchmarks (D-08)

pub mod baseline;
pub mod fixtures;
```

**Extension pattern** — add three lines; update doc comment to list new modules:
```rust
pub mod loader;   // NEW: Monash TSF + JSON dataset loader (ACCUR-01)
pub mod naive2;   // NEW: Naive2 model (ACCUR-06, D-07)
pub mod dm_test;  // NEW: Diebold-Mariano + HLN + HAC (BENCH-02, D-09)
```

---

### `src/utils/metrics.rs` — D-03 MASE collapse guard fix

**Analog:** self — modify `calculate_mase` function (lines 135–168).

**Current broken code** (lines 156–158, VERIFIED):
```rust
if naive_mae == 0.0 {
    return None;  // propagates as NaN via unwrap_or(NaN) callers
}
```

**Fix pattern** (D-03/D-04 — period-1 fallback denominator; matches statsforecast behavior):
```rust
// Denominator-collapse guard (D-04): when seasonal naive MAE is zero
// (constant training window or fewer than one season of data), substitute
// a period-1 naive denominator rather than dropping the series. Matches
// statsforecast MASE behavior; keeps series count in the aggregate stable.
let naive_mae = if naive_mae == 0.0 {
    let p1_mae: f64 = actual
        .iter()
        .skip(1)
        .zip(actual.iter())
        .map(|(curr, prev)| (curr - prev).abs())
        .sum::<f64>()
        / (n - 1) as f64;
    if p1_mae == 0.0 {
        return None; // truly constant series — no scaling possible
    }
    p1_mae
} else {
    naive_mae
};
```

**Regression test pattern** — add to the existing `#[cfg(test)]` block at bottom of `metrics.rs`, following existing test style:
```rust
#[test]
fn mase_constant_series_no_nan() {
    // Constant training window collapses the seasonal denominator to 0.
    // D-03 fix: period-1 fallback must return a finite MASE, not None/NaN.
    let constant = vec![5.0_f64; 20];
    let predicted = vec![5.1_f64; 20];
    let result = calculate_mase(&constant, &predicted, Some(12));
    // Before fix: returns None (propagates as NaN). After fix: Some(finite).
    assert!(result.is_some(), "MASE must not be None on constant series (D-03)");
    assert!(result.unwrap().is_finite(), "MASE must be finite on constant series (D-03)");
}
```

---

### `.github/workflows/accuracy.yml` (config)

**Analog:** `.github/workflows/bench.yml` (lines 1–61) — same permissions model, same Rust toolchain setup, same cache step.

**Trigger pattern** — `workflow_dispatch`-only (MEAS-03 locked constraint); contrast with `bench.yml` which uses push/PR:
```yaml
name: Accuracy Harness

on:
  workflow_dispatch:  # MEAS-03: never gates PR merges; manual trigger only
```

**Permissions pattern** (from `bench.yml` lines 26–27 — minimal read-only):
```yaml
permissions:
  contents: read
```

**Rust toolchain + cache steps** (from `bench.yml` lines 30–34 — copy verbatim):
```yaml
- uses: actions/checkout@v4
- uses: dtolnay/rust-toolchain@stable
- uses: Swatinem/rust-cache@v2
```

**Job step pattern** (from `bench.yml` lines 59–60 — cargo test, not bench):
```yaml
- name: Run accuracy harness
  env:
    ANOFOX_DATASET_DIR: ${{ github.workspace }}/validation/data
  run: cargo test -p anofox-bench-harness --test accuracy -- --nocapture
```

---

## Shared Patterns

### File I/O env-gate (ANOFOX_DATASET_DIR)
**Source:** `examples/skaters_m3_monthly_benchmark.rs` lines 90–95 (DATA_PATH env var pattern)
**Apply to:** `loader.rs`, `tests/accuracy.rs`
```rust
let path = std::env::var("ANOFOX_DATASET_DIR")
    .unwrap_or_else(|_| { /* skip or default */ });
```

### TimeSeries construction from Vec<f64>
**Source:** `tests/m4_daily_accuracy_regression.rs` lines 142–145; `examples/skaters_m3_monthly_benchmark.rs` lines 137–143
**Apply to:** `naive2.rs`, `tests/accuracy.rs`
```rust
let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
let timestamps: Vec<_> = (0..values.len())
    .map(|i| base + Duration::days(30 * i as i64))
    .collect();
let ts = TimeSeries::univariate(timestamps, values).unwrap();
```

### Serde JSON baseline loading with provenance
**Source:** `crates/anofox-bench-harness/src/baseline.rs` lines 1–81; `tests/full_statsforecast_comparison.rs` lines 37–42
**Apply to:** `tests/accuracy.rs` (reading `accuracy.json`, `statsforecast_reference.json`)
```rust
use anofox_bench_harness::baseline::ProvenanceFingerprint;
// ProvenanceFingerprint carries git_sha, timestamp_iso, rustc_version, host_cpu, host_os, active_features
```

### Standalone metric function style
**Source:** `src/utils/metrics.rs` lines 172–357 (mae, rmse, smape, msis, coverage)
**Apply to:** `dm_test.rs` (normal_cdf helper), `loader.rs` (mase_scale helper)
- Pure functions, slice inputs, return `f64` (NaN on bad input) or typed result
- No struct state; no `Result<T>` unless error context is needed
- One-line guard at top: check lengths and emptiness

### Module doc comment format
**Source:** `src/utils/cross_validation.rs` lines 1–10
**Apply to:** all new `.rs` files
```rust
//! <One-line module purpose>.
//!
//! <Lists main types or functions and their design rationale.>
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `.planning/baselines/accuracy.json` | config | — | No accuracy JSON yet in `.planning/baselines/`; schema derived from `baseline.rs` `ProvenanceFingerprint` + RESEARCH.md §Accuracy JSON Schema |

---

## Metadata

**Analog search scope:** `src/utils/`, `crates/anofox-bench-harness/src/`, `examples/`, `tests/`, `.github/workflows/`
**Files scanned:** 12
**Pattern extraction date:** 2026-08-10
