# Phase 01: Measurement Infrastructure & Compute Baselines — Pattern Map

**Mapped:** 2026-08-09
**Files analyzed:** 13 new/modified files
**Analogs found:** 10 / 13

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/anofox-bench-harness/Cargo.toml` | config | — | `crates/anofox-forecast-js/Cargo.toml` | role-match |
| `crates/anofox-bench-harness/src/lib.rs` | utility | — | `crates/anofox-forecast-js/src/lib.rs` | role-match |
| `crates/anofox-bench-harness/src/baseline.rs` | model (serde structs) | transform | `crates/anofox-forecast-js/Cargo.toml` (serde pattern) | partial |
| `crates/anofox-bench-harness/src/fixtures.rs` | utility | batch | `benches/ets_benchmark.rs` (generator fns) | exact |
| `benches/baseline_suite.rs` | utility | batch | `benches/ets_benchmark.rs` | exact |
| `crates/anofox-bench-harness/benches/iai_suite.rs` | utility | batch | `benches/ets_benchmark.rs` + `Cargo.toml` [[bench]] | partial |
| `crates/anofox-bench-harness/tests/dhat_peak.rs` | test | request-response | `tests/batch_validation.rs` | role-match |
| `scripts/update_criterion.sh` | utility | batch | `scripts/fred_laplace_bakeoff.py` (structure only) | no analog (shell) |
| `scripts/update_iai.sh` | utility | batch | — | no analog |
| `scripts/update_dhat.sh` | utility | batch | — | no analog |
| `scripts/update_wasm_size.sh` | utility | batch | — | no analog |
| `.github/workflows/bench.yml` | config | event-driven | `.github/workflows/ci.yml` | role-match |
| `.github/workflows/wasm-size.yml` | config | event-driven | `.github/workflows/ci.yml` + `npm.yml` + `deploy-playground.yml` | exact |
| `crates/anofox-forecast-js/src/forecaster.rs` (PERF-06 edits) | — | — | itself | self |
| `crates/anofox-forecast-js/src/laplace_playground.rs` (PERF-06 edits) | — | — | itself | self |

---

## Pattern Assignments

### `crates/anofox-bench-harness/Cargo.toml` (config)

**Analog:** `crates/anofox-forecast-js/Cargo.toml`

**Package header pattern** (lines 1–11):
```toml
[package]
name = "anofox-forecast-js"
version = "0.15.8"
edition = "2021"
authors = ["Simon M"]
```

**New harness crate — copy and adapt:**
```toml
[package]
name = "anofox-bench-harness"
version = "0.1.0"
edition = "2021"
authors = ["Simon Müller"]
publish = false          # internal tooling only; never published to crates.io

[lib]
name = "anofox_bench_harness"

[dependencies]
# anofox-forecast as a path dep — same pattern as anofox-forecast-js line 19
anofox-forecast = { path = "../..", default-features = false, features = [] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"

[dev-dependencies]
iai-callgrind = "0.16.1"
dhat = "0.3.3"
criterion = "0.5"

[[bench]]
name = "iai_suite"
harness = false
```

**Root `Cargo.toml` workspace members** (line 2 — add harness crate):
```toml
# Current (Cargo.toml line 1–3):
[workspace]
members = [".", "crates/anofox-forecast-js"]
resolver = "2"

# After edit:
[workspace]
members = [".", "crates/anofox-forecast-js", "crates/anofox-bench-harness"]
resolver = "2"
```

---

### `crates/anofox-bench-harness/src/fixtures.rs` (utility, batch)

**Analog:** `benches/ets_benchmark.rs` lines 10–26

**Timestamp generator pattern** (ets_benchmark.rs lines 10–13):
```rust
fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}
```

**LCG series generator pattern** (ets_benchmark.rs lines 15–26):
```rust
fn generate_seasonal_series(n: usize) -> Vec<f64> {
    let mut rng_state: u64 = 123;
    let mut series = Vec::with_capacity(n);
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
        let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
        let trend = 0.05 * i as f64;
        series.push(trend + seasonal + noise + 20.0); // +20 to keep values positive for multiplicative
    }
    series
}
```

**TimeSeries construction pattern** (ets_benchmark.rs lines 29–30):
```rust
let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();
```

**Harness crate public fixture — adapts the above into a seeded function:**
```rust
// crates/anofox-bench-harness/src/fixtures.rs
use anofox_forecast::core::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

/// Deterministic synthetic time series with trend + seasonality + noise.
/// Uses the same LCG as existing project benches (seed-parameterized for D-08).
pub fn make_seeded_series(n: usize, seed: u64) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n)
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    let mut rng_state: u64 = seed;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let trend = 0.05 * i as f64;
            trend + seasonal + noise + 20.0
        })
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}
```

---

### `benches/baseline_suite.rs` (utility, batch)

**Analog:** `benches/ets_benchmark.rs` (full file)

**Imports pattern** (ets_benchmark.rs lines 1–8):
```rust
use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, HoltLinearTrend, HoltWinters, SimpleExponentialSmoothing,
};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
```

**bench_function pattern** (ets_benchmark.rs lines 94–103):
```rust
fn bench_auto_ets(c: &mut Criterion) {
    let values = generate_seasonal_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    c.bench_function("auto_ets_n200_p12", |b| {
        b.iter(|| {
            let mut model = AutoETS::with_period(12);
            model.fit(&ts).unwrap();
        })
    });
```

**criterion_group! / criterion_main! registration** (ets_benchmark.rs lines 124–131):
```rust
criterion_group!(
    benches,
    bench_ses_fit,
    bench_holt_fit,
    bench_holt_winters_fit,
    bench_auto_ets
);
criterion_main!(benches);
```

**Standardized group with explicit sample_size** (D-06 requires this — adapt from ets_benchmark):
```rust
// baseline_suite.rs — tracked group uses named config (not the default inline form)
criterion_group!(
    name = baselines;
    config = Criterion::default().sample_size(20);   // standardized per D-06
    targets = bench_auto_ets, bench_auto_arima, bench_auto_theta,
              bench_naive, bench_croston, bench_auto_ensemble, bench_laplace
);
criterion_main!(baselines);
```

**`[[bench]]` registration in root `Cargo.toml`** — copy the pattern at lines 74–109:
```toml
[[bench]]
name = "baseline_suite"
harness = false
```

**Note on circular dependency (RESEARCH.md open question 3):** `benches/baseline_suite.rs` lives in the root crate, which `anofox-bench-harness` already depends on. Adding `anofox-bench-harness` as a dev-dependency of the root crate would create a cycle. Solution: inline the 10-line LCG fixture in `baseline_suite.rs` instead of importing it from the harness crate. The harness crate version is authoritative for iai/dhat; the criterion suite has its own copy (identical code, same seed convention).

---

### `crates/anofox-bench-harness/benches/iai_suite.rs` (utility, batch)

**No exact analog** — iai-callgrind does not exist in the codebase yet. Closest analog for `[[bench]]` registration and black_box usage: `benches/ets_benchmark.rs` + root `Cargo.toml` lines 74–109.

**`harness = false` registration pattern** (root Cargo.toml lines 74–76):
```toml
[[bench]]
name = "ets_benchmark"
harness = false
```

**Model construction + fit pattern** (ets_benchmark.rs lines 98–103):
```rust
c.bench_function("auto_ets_n200_p12", |b| {
    b.iter(|| {
        let mut model = AutoETS::with_period(12);
        model.fit(&ts).unwrap();
    })
});
```

**batch API pattern** (batch_validation.rs lines 1–3, 58–59):
```rust
use anofox_forecast::batch;
// ...
let batch_results = batch::mfles(&values, period, Some(horizon));
```

**iai-callgrind skeleton** (from RESEARCH.md Pattern 1 — no codebase analog):
```rust
// crates/anofox-bench-harness/benches/iai_suite.rs
use iai_callgrind::{
    library_benchmark, library_benchmark_group, main,
    LibraryBenchmarkConfig, Callgrind, EventKind,
};
use std::hint::black_box;
use anofox_bench_harness::fixtures::make_seeded_series;
use anofox_forecast::models::{arima::AutoARIMA, exponential::AutoETS, Forecaster};

fn setup_ts_n200() -> anofox_forecast::core::TimeSeries {
    make_seeded_series(200, 42)
}

#[library_benchmark]
#[bench::n200(setup_ts_n200())]
fn bench_auto_ets_fit(ts: anofox_forecast::core::TimeSeries) {
    let mut model = AutoETS::with_period(12);
    black_box(model.fit(black_box(&ts)).unwrap());
}

library_benchmark_group!(
    name = hot_paths;
    config = LibraryBenchmarkConfig::default()
        .callgrind(Callgrind::default()
            .soft_limits([(EventKind::Ir, 1.0f64)]));  // D-10: fail CI if Ir rises > 1%
    benchmarks = bench_auto_ets_fit, bench_auto_arima_fit, bench_batch_100
);

main!(library_benchmark_groups = hot_paths);
```

**IMPORTANT — verify batch API before writing:** RESEARCH.md open question 5 flags that `batch::auto_ets()` signature is unverified. Read `src/batch.rs` before implementing the `bench_batch_100` function.

---

### `crates/anofox-bench-harness/tests/dhat_peak.rs` (test, request-response)

**Analog:** `tests/batch_validation.rs` — for integration test structure (imports, helper fns, `#[test]` blocks).

**Test file top-of-file module doc pattern** (batch_validation.rs line 1):
```rust
//! Batch API validation: correctness and performance comparison.
```

**Helper function pattern** (batch_validation.rs lines 14–18):
```rust
fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}
```

**`#[test]` structure** (batch_validation.rs lines 39–59):
```rust
#[test]
fn batch_mfles_matches_individual() {
    let period = 12;
    // setup, run, assert
}
```

**dhat-specific additions** (no codebase analog — from RESEARCH.md Pattern 2):
```rust
// crates/anofox-bench-harness/tests/dhat_peak.rs
// NOTE: dhat::Alloc must be the ONLY #[global_allocator] in this test binary.
// Do not add other integration tests to this file.

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
fn auto_ets_peak_memory_within_baseline() {
    let _profiler = dhat::Profiler::builder().testing().build();

    let ts = anofox_bench_harness::fixtures::make_seeded_series(1000, 42);
    let mut model = anofox_forecast::models::exponential::AutoETS::with_period(12);
    model.fit(&ts).unwrap();
    let _forecast = model.predict(12).unwrap();

    let stats = dhat::HeapStats::get();
    let baseline_bytes = load_dhat_baseline("auto_ets_n1000");  // reads .planning/baselines/dhat.json
    assert!(
        stats.max_bytes <= (baseline_bytes as f64 * 1.15) as usize,
        "AutoETS peak memory {} bytes exceeds baseline {} × 1.15",
        stats.max_bytes,
        baseline_bytes
    );
}
```

---

### `crates/anofox-bench-harness/src/baseline.rs` (serde structs)

**Analog:** serde derive usage in `crates/anofox-forecast-js/Cargo.toml` line 22 (`serde = { version = "1.0", features = ["derive"] }`).

**No struct analog in codebase** — this is a new serde schema. Use the pattern from RESEARCH.md Pattern 5 directly. Key conventions to follow from project CLAUDE.md:

- Structs use `PascalCase`
- Fields use `snake_case`
- Derive order: `Serialize, Deserialize, Debug, Clone`

```rust
// crates/anofox-bench-harness/src/baseline.rs
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProvenanceFingerprint {
    pub git_sha: String,
    pub timestamp_iso: String,
    pub rustc_version: String,
    pub host_cpu: String,
    pub host_os: String,
    pub active_features: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CriterionEntry {
    pub name: String,    // e.g. "auto_ets_fit_n200_parallel"
    pub profile: String, // "parallel" | "no_parallel"  (D-09)
    pub median_ns: f64,
    pub mad_ns: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IaiEntry {
    pub name: String,
    pub instruction_count: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DhatEntry {
    pub name: String,
    pub peak_bytes: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WasmSizeBaseline {
    pub provenance: ProvenanceFingerprint,
    pub filename: String,
    pub bytes: u64,
}
```

---

### `.github/workflows/bench.yml` (config, event-driven)

**Analog:** `.github/workflows/ci.yml` — for trigger block, runner/toolchain/cache steps, job structure.

**Trigger + env pattern** (ci.yml lines 1–12):
```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

env:
  CARGO_TERM_COLOR: always
```

**Rust toolchain + cache step pattern** (ci.yml lines 23–28 — stable only, per D-10):
```yaml
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable   # D-10: pinned stable only for iai gate
      - uses: Swatinem/rust-cache@v2
```

**`taiki-e/install-action` pattern** (ci.yml lines 58–60):
```yaml
      - name: Install cargo-audit
        uses: taiki-e/install-action@cargo-audit
```

**bench.yml structure:**
```yaml
name: Bench Gate

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

env:
  CARGO_TERM_COLOR: always

jobs:
  iai-gate:
    name: iai-callgrind Instruction Gate
    runs-on: ubuntu-latest        # ubuntu-latest = 24.04, valgrind 3.22 (sufficient)
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable   # D-10: stable only
      - uses: Swatinem/rust-cache@v2
      - name: Install Valgrind
        run: sudo apt-get update && sudo apt-get install -y valgrind
      - name: Install iai-callgrind-runner
        run: cargo install iai-callgrind-runner --version 0.16.1 --locked  # must match crate version exactly
      - name: Run iai bench (instruction gate)
        run: cargo bench -p anofox-bench-harness --bench iai_suite
      # NOTE: do NOT run cargo bench --bench baseline_suite here — D-03/D-04 forbid CI wall-clock capture
```

---

### `.github/workflows/wasm-size.yml` (config, event-driven)

**Analog:** `.github/workflows/ci.yml` wasm job (lines 93–107) + `npm.yml` wasm-pack build steps (lines 30–38) + `deploy-playground.yml` wasm-pack pattern (lines 58–63).

**wasm-pack install step** (ci.yml line 119 / npm.yml line 29):
```yaml
      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

**wasm-pack build command** (npm.yml lines 33–34):
```yaml
      - name: Build WASM package
        run: wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js
```

**git checkout to restore package.json** (npm.yml lines 35–37):
```yaml
      - name: Restore custom package.json and README
        run: |
          git checkout js/package.json js/README.md
```

**wasm-size.yml structure:**
```yaml
name: WASM Size Gate

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

env:
  CARGO_TERM_COLOR: always

jobs:
  wasm-size:
    name: WASM Binary Size Gate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
      - uses: Swatinem/rust-cache@v2
      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
      - name: Build release WASM
        run: wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js --release
      - name: Restore custom package.json and README
        run: git checkout js/package.json js/README.md
      - name: Check WASM size gate (D-11: fail if > 1% growth)
        run: |
          CURRENT=$(stat --format=%s js/anofox_forecast_js_bg.wasm)
          BASELINE=$(python3 -c "import json; d=json.load(open('.planning/baselines/wasm_size.json')); print(d['bytes'])")
          python3 -c "
          current, baseline = $CURRENT, $BASELINE
          delta = (current - baseline) / baseline * 100
          print(f'WASM size: current={current}B baseline={baseline}B delta={delta:.2f}%')
          if delta > 1.0:
              raise SystemExit(f'WASM size grew {delta:.2f}% (threshold: 1.0%)')
          "
```

---

### `scripts/update_*.sh` (utility, batch)

**No shell-script analog** — existing `scripts/` contains only Python files. There is no shell precedent in the codebase.

**Conventions to adopt from existing Python scripts** (`scripts/fred_laplace_bakeoff.py` header pattern — shebang, purpose comment):
```python
#!/usr/bin/env python3
"""
<one-line description>
"""
```

**Shell equivalent pattern (new, no codebase source):**
```bash
#!/usr/bin/env bash
# scripts/update_wasm_size.sh — capture WASM size baseline into .planning/baselines/wasm_size.json
# Run on a quiet local machine after PERF-06 dead-code removal is complete.
set -euo pipefail

# Guard: must have zero warnings before capturing WASM baseline (PERF-06 sequencing)
WARNINGS=$(cargo build -p anofox-forecast-js --target wasm32-unknown-unknown 2>&1 | grep "warning:" | wc -l)
if [ "$WARNINGS" -ne 0 ]; then
    echo "ERROR: $WARNINGS compiler warnings in anofox-forecast-js. Run PERF-06 cleanup first."
    exit 1
fi

# Build + measure
wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js --release
git checkout js/package.json js/README.md

BYTES=$(stat --format=%s js/anofox_forecast_js_bg.wasm)
GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUSTC=$(rustc --version)
OS=$(uname -sr)

python3 -c "
import json, sys
record = {
    'provenance': {
        'git_sha': '$GIT_SHA',
        'timestamp_iso': '$TIMESTAMP',
        'rustc_version': '$RUSTC',
        'host_cpu': '$(grep \"model name\" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo unknown)',
        'host_os': '$OS',
        'active_features': []
    },
    'filename': 'anofox_forecast_js_bg.wasm',
    'bytes': $BYTES
}
with open('.planning/baselines/wasm_size.json', 'w') as f:
    json.dump(record, f, indent=2)
print(f'Wrote wasm_size.json: {$BYTES} bytes')
"
```

---

### PERF-06: Edits to `crates/anofox-forecast-js/src/forecaster.rs`

**Source file** — reading itself as the analog.

**Pattern of unused `inner()` methods to DELETE** (forecaster.rs lines 383–387, then repeated):
```rust
// DELETE this entire impl block for each of the 8 structs:
impl SESForecaster {
    pub(crate) fn inner(&self) -> &SimpleExponentialSmoothing {
        &self.model
    }
}
```

**Exact line numbers to remove** (confirmed by live build — RESEARCH.md lines 533–542):
- Lines 383–387: `impl SESForecaster { pub(crate) fn inner() ... }`
- Lines 433–437: `impl HoltForecaster { pub(crate) fn inner() ... }`
- Lines ~515–520: `impl HoltWintersForecaster { pub(crate) fn inner() ... }`
- Lines ~1207–1212: `impl CrostonForecaster { pub(crate) fn inner() ... }`
- Lines ~1259–1264: `impl TSBForecaster { pub(crate) fn inner() ... }`
- Lines ~1313–1318: `impl ADIDAForecaster { pub(crate) fn inner() ... }`
- Lines ~1367–1372: `impl IMAPAForecaster { pub(crate) fn inner() ... }`
- Lines ~1631–1636: `impl GARCHForecaster { pub(crate) fn inner() ... }`
- Line 2141: remove `mut` from `let mut kf = ...`

**Do NOT remove** `inner()` in `time_series.rs:144`, `calendar.rs:212`, `postprocess.rs:67` — those are used.

### PERF-06: Edit to `crates/anofox-forecast-js/src/laplace_playground.rs`

**Line 15** — remove `RecipeKind` from the use statement:
```rust
// BEFORE (laplace_playground.rs line 14–16):
use anofox_forecast::models::laplace::{
    recipe_for, recommended_for, DistributionalForecaster, LaplaceForecaster, MultiScaleLaplace,
    RecipeKind,
};

// AFTER:
use anofox_forecast::models::laplace::{
    recipe_for, recommended_for, DistributionalForecaster, LaplaceForecaster, MultiScaleLaplace,
};
```

---

## Shared Patterns

### Criterion `harness = false` bench registration

**Source:** `Cargo.toml` lines 74–109
**Apply to:** Every new `[[bench]]` entry in root `Cargo.toml` and `crates/anofox-bench-harness/Cargo.toml`
```toml
[[bench]]
name = "baseline_suite"
harness = false
```

### GitHub Actions Rust toolchain + cache

**Source:** `.github/workflows/ci.yml` lines 22–28
**Apply to:** `bench.yml`, `wasm-size.yml`
```yaml
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
```

### wasm-pack build + package.json restore

**Source:** `.github/workflows/npm.yml` lines 33–37
**Apply to:** `wasm-size.yml`
```yaml
      - name: Build WASM package
        run: wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js
      - name: Restore custom package.json and README
        run: git checkout js/package.json js/README.md
```

### LCG seeded series generator

**Source:** `benches/ets_benchmark.rs` lines 15–26
**Apply to:** `crates/anofox-bench-harness/src/fixtures.rs`, inlined copy in `benches/baseline_suite.rs`

LCG multiplier: `6364136223846793005u64`. Shift: `>> 33`. Seed drives `rng_state` initial value.

### `Forecaster` fit + predict call convention

**Source:** `benches/ets_benchmark.rs` lines 98–103
**Apply to:** All bench functions in `baseline_suite.rs` and `iai_suite.rs`
```rust
let mut model = AutoETS::with_period(12);
model.fit(&ts).unwrap();
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `scripts/update_criterion.sh` | utility | batch | No shell scripts in codebase; Python-only scripts directory |
| `scripts/update_iai.sh` | utility | batch | Same — no shell precedent |
| `scripts/update_dhat.sh` | utility | batch | Same — no shell precedent |
| `scripts/update_wasm_size.sh` | utility | batch | Same — no shell precedent |
| `crates/anofox-bench-harness/benches/iai_suite.rs` | utility | batch | iai-callgrind not yet in codebase; use RESEARCH.md Pattern 1 |
| `.planning/baselines/*.json` | data | — | Committed data files; no code analog; use D-02 schema from baseline.rs |

For these files, the planner should reference the RESEARCH.md patterns directly (Pattern 1–6).

---

## Metadata

**Analog search scope:** `benches/`, `tests/`, `.github/workflows/`, `crates/`, `scripts/`
**Files scanned:** 10 (ets_benchmark.rs, ensemble_benchmark.rs, batch_validation.rs, ci.yml, deploy-playground.yml, npm.yml, root Cargo.toml, anofox-forecast-js/Cargo.toml, forecaster.rs:380–440, forecaster.rs:2135–2145, laplace_playground.rs:1–25)
**Pattern extraction date:** 2026-08-09
