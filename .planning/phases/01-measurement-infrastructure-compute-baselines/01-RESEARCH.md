# Phase 01: Measurement Infrastructure & Compute Baselines — Research

**Researched:** 2026-08-09
**Domain:** Rust benchmarking toolchain (criterion, iai-callgrind, dhat), WASM size tracking, dead-code removal, CI gate design
**Confidence:** MEDIUM

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** One JSON file per dimension under `.planning/baselines/` — `criterion.json`, `iai.json`, `dhat.json`, `wasm_size.json` — each self-contained.
- **D-02:** Each baseline record carries a full provenance fingerprint: git SHA, ISO timestamp, rustc version, host CPU/OS, active feature flags, plus metric value(s).
- **D-03:** criterion baselines store median + MAD (robust to wall-clock noise); informational only in CI — CI reports drift but never fails on criterion. iai-callgrind is the hard compute gate.
- **D-04:** New workspace harness crate (e.g. `crates/anofox-bench-harness`) owns baseline serde structs, read/compare logic, and the dhat + iai bins. `benches/` and `scripts/*.sh` call into it.
- **D-05:** Maintainer entrypoint is `scripts/update_*.sh` per dimension — `update_criterion.sh`, `update_iai.sh`, `update_dhat.sh`, `update_wasm_size.sh`.
- **D-06:** One new dedicated baseline bench (`benches/baseline_suite.rs`) as the single source of truth for committed criterion baselines. Existing 8 benches stay as-is.
- **D-07:** Tracked matrix = one representative model per family — AutoARIMA, AutoETS, AutoTheta, Naive, Croston, AutoEnsemble, Laplace — × {single-series fit+predict, batch-100}.
- **D-08:** Tracked benches pull from shared seeded fixtures in the harness crate — deterministic series at fixed lengths (n=100, n=1000) with a fixed seed.
- **D-09:** Native-parallel and WASM/single-thread (no-Rayon) profiles reported in separate sections of the criterion output/baseline.
- **D-10:** iai-callgrind gate: run only on pinned stable Rust; fail CI if instruction count rises > 1% vs baseline. Beta/nightly build+test but do not gate instructions.
- **D-11:** WASM size gate: fail if compiled release `.wasm` grows > 1% relative to committed baseline.
- **D-12:** dhat peak-memory gate: hard assert peak stays under baseline × 1.15 (15% headroom) for major model families.

### Claude's Discretion

- **iai hot-path selection:** AutoETS fit, AutoARIMA fit, batch-100 as the initial 3 critical hot paths. Room to add 1–2 more.
- **Dead-code cleanup scope (PERF-06):** Remove named unused `inner()` methods + unused imports in `crates/anofox-forecast-js/`; run cargo-machete sweep; confirm npm package still builds before locking WASM size baseline.
- **`bench.yml` CI scope:** iai-callgrind instruction gate only; criterion is local-capture.
- **dhat harness form:** native bin/test in harness crate using dhat's allocator (NOT `wee_alloc` — banned). Family selection = PERF-01 representative set.

### Deferred Ideas (OUT OF SCOPE)

- Broader within-family bench coverage (multiple intermittent/ETS/HW variants) — revisit if Phase 4 backlog points at a specific family.
- Vendored real-series fixtures — deferred; ties bench timing to Phase-2 dataset/loader work.
- WASM runtime memory profiling — explicitly a v2 requirement (XROB-02).

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MEAS-01 | Committed baseline store at `.planning/baselines/` with JSON per dimension; CI reads but never writes | Baseline schema (D-02) defined; one file per tool (D-01) |
| MEAS-02 | Maintainer can refresh baselines via `scripts/update_*.sh` on a quiet local machine | `update_criterion.sh`, `update_iai.sh`, `update_dhat.sh`, `update_wasm_size.sh` scripts documented |
| MEAS-03 | CI workflows `bench.yml` and `wasm-size.yml` exist; `accuracy.yml` is workflow_dispatch only | bench.yml structure (iai gate + valgrind install); wasm-size.yml structure documented |
| MEAS-04 | All measurement code in `benches/`, `tests/`, `scripts/`, harness crate — nothing new in `src/` | New harness crate `crates/anofox-bench-harness` owns all tooling code |
| PERF-01 | criterion suite covers fit+predict across 7 model families, single+batch, committed baselines | `benches/baseline_suite.rs` + seeded fixtures; `target/criterion/*/new/estimates.json` → `criterion.json` |
| PERF-02 | iai-callgrind instruction-count gates on 3–5 hot paths in CI | iai-callgrind 0.16.1; `soft_limits([(EventKind::Ir, 1.0f64)])` for 1% gate; 3 initial hot paths |
| PERF-03 | Native-parallel and WASM/single-thread profiles measured separately | Separate criterion sections in baseline; WASM target forbids `parallel` feature |
| PERF-04 | dhat peak-memory measurement asserts bounds for major model families | `dhat::Profiler::builder().testing().build()` + `HeapStats::get().max_bytes < baseline * 115 / 100` |
| PERF-05 | Compiled release WASM binary size tracked against committed baseline with CI delta | `js/anofox_forecast_js_bg.wasm` size via `stat`; 1% threshold gate in `wasm-size.yml` |
| PERF-06 | Known WASM dead code removed before WASM size baseline locked | 10 compiler warnings confirmed: 8 unused `inner()` methods in `forecaster.rs` + 1 unused import + 1 unnecessary `mut`; removal verified via `cargo build -p anofox-forecast-js` |

</phase_requirements>

---

## Summary

Phase 1 stands up the measurement backbone for a Rust library hardening cycle. The primary challenge is integrating four distinct measurement dimensions — criterion wall-clock benchmarks, iai-callgrind instruction-count gates, dhat peak-memory bounds, and WASM binary size tracking — into a coherent CI+local workflow with a committed baseline store that CI reads but never overwrites.

The toolchain is concrete: `criterion 0.5` is already a dev-dependency (no new cost); `iai-callgrind 0.16.1` requires valgrind to be installed on the CI runner (one `apt-get install` step, available on ubuntu 24.04 which is now `ubuntu-latest`); `dhat 0.3.3` uses a swap-in global allocator for test-mode heap profiling; and WASM size is measured by `wc -c` / `stat` on the `wasm-pack` release output. All four are established, legitimate crates with multi-year histories and high download counts.

The key sequencing constraint is **PERF-06 before PERF-05**: the compiler already emits 10 warnings against `crates/anofox-forecast-js` (8 unused `inner()` methods, 1 unused import, 1 unnecessary `mut`) that inflate the WASM binary. These must be removed before the WASM size baseline is committed, or the baseline will overstate the true post-cleanup size.

**Primary recommendation:** Add `iai-callgrind = "0.16.1"` and `dhat = "0.3.3"` as dev-dependencies in the new harness crate; use `soft_limits([(EventKind::Ir, 1.0f64)])` for the CI instruction-count gate; use `dhat::Profiler::builder().testing().build()` + `HeapStats::get().max_bytes` for the memory gate; measure `js/anofox_forecast_js_bg.wasm` size for the WASM gate. Remove the 10 dead-code warnings in `crates/anofox-forecast-js/` before capturing the WASM baseline.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Criterion baselines capture | Local machine (scripts) | — | Wall-clock noise on CI is unacceptable; D-03 locks this |
| iai-callgrind instruction gate | CI (bench.yml) | Local (update_iai.sh) | Callgrind instruction counts are stable across virtualized environments |
| dhat peak-memory gate | Native test in harness crate | CI (bench.yml or separate job) | Runs as a Rust test via `cargo test -p anofox-bench-harness` |
| WASM size gate | CI (wasm-size.yml) | Local (update_wasm_size.sh) | Production build artifact; gate must run on a consistent environment |
| Baseline store (read) | CI (all workflows) | — | Committed JSON read at CI start; never written |
| Baseline store (write) | Local (scripts/update_*.sh) | — | Deliberate human action; never automated |
| Dead-code cleanup | Library source (`crates/anofox-forecast-js/`) | — | Removes 10 compiler warnings before WASM baseline lock |
| Harness crate (shared code) | `crates/anofox-bench-harness` | — | Single typed owner of provenance schema + compare logic |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| criterion | 0.5 | Wall-clock benchmarks, criterion baselines | Already in dev-dependencies; industry standard for Rust |
| iai-callgrind | 0.16.1 | Instruction-count benchmarks + CI gate | Only stable callgrind harness on crates.io; 26K weekly downloads [VERIFIED: crates.io] |
| iai-callgrind-runner | 0.16.1 | Companion binary that runs the callgrind tool | Must match iai-callgrind version exactly |
| dhat | 0.3.3 | Heap peak-memory measurement in tests | Written by the Rust perf-book author; 268K weekly downloads [VERIFIED: crates.io] |
| serde / serde_json | 1.0 | Baseline JSON serialization | Already in dev-dependencies |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cargo-machete | 0.9.2 (CLI tool) | Scan for unused Cargo dependencies | PERF-06 sweep on `crates/anofox-forecast-js/` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| iai-callgrind 0.16.1 | gungraun 0.19.4 | gungraun is the renamed upstream (same project) but NOT yet on crates.io as a stable release; use 0.16.1 from crates.io until gungraun stabilizes |
| dhat (Rust-side) | heaptrack, massif | heaptrack/massif are external tools requiring separate profiling runs; dhat integrates directly into Rust test harness with `HeapStats::get()` — required for programmatic assertion |
| wee_alloc | dhat::Alloc | `wee_alloc` is **banned** — archived Aug 2025 with known memory leaks [VERIFIED: STATE.md] |
| criterion JSON parsing | cargo-criterion | `cargo-criterion` has a separate install; raw `target/criterion/*/new/estimates.json` parsing in a shell script is simpler and has no extra dependency |

**Installation (harness crate dev-dependencies):**
```toml
[dev-dependencies]
iai-callgrind = "0.16.1"
dhat = "0.3.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
criterion = "0.5"  # inherited from workspace
```

**CI tooling install (bench.yml):**
```yaml
- name: Install Valgrind
  run: sudo apt-get update && sudo apt-get install -y valgrind

- name: Install iai-callgrind-runner
  run: cargo install iai-callgrind-runner --version 0.16.1 --locked
```

---

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| iai-callgrind | crates.io | 3+ yrs (2023-03-08) | 26,108/wk | github.com/iai-callgrind/iai-callgrind | OK | Approved |
| iai-callgrind-runner | crates.io | 3+ yrs | companion | github.com/iai-callgrind/iai-callgrind | OK | Approved |
| dhat | crates.io | 5+ yrs (2020-12-08) | 268,074/wk | github.com/nnethercote/dhat-rs | OK | Approved |
| cargo-machete | crates.io | 4+ yrs (2021-11-13) | 35,585/wk | github.com/bnjbvr/cargo-machete | OK | Approved |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

**Note on gungraun:** The GitHub repository at `github.com/iai-callgrind/iai-callgrind` has been renamed to `gungraun/gungraun` and the project published to crates.io as `gungraun 0.19.4`. However, `iai-callgrind 0.16.1` remains the stable crates.io version, is not deprecated, and is the correct version to pin. Do NOT switch to gungraun 0.19.4 without a deliberate migration decision — the crate name changed and the API may have breaking changes.

---

## Architecture Patterns

### System Architecture Diagram

```
Local machine (update_*.sh)          GitHub Actions CI
─────────────────────────────        ──────────────────────────────────────

  cargo bench --bench baseline_suite  bench.yml:
  → target/criterion/.../estimates.json  install valgrind
  → scripts/update_criterion.sh         cargo install iai-callgrind-runner
  → .planning/baselines/criterion.json  cargo bench --bench iai_suite
                                        read iai.json baseline
  cargo test -p anofox-bench-harness    compare instruction counts
  (dhat tests)                          fail if Ir rises > 1%
  → scripts/update_dhat.sh
  → .planning/baselines/dhat.json    wasm-size.yml:
                                        wasm-pack build --release
  cargo bench --bench iai_suite         stat js/anofox_forecast_js_bg.wasm
  → scripts/update_iai.sh               read wasm_size.json baseline
  → .planning/baselines/iai.json        fail if size grows > 1%

  wasm-pack build --release
  → scripts/update_wasm_size.sh      bench.yml reads dhat.json (informational)
  → .planning/baselines/wasm_size.json bench.yml reads criterion.json (informational)

All baselines committed in git. CI always reads; never writes.
```

### Recommended Project Structure

```
crates/
└── anofox-bench-harness/         # New workspace member (D-04)
    ├── Cargo.toml                # dev-deps: iai-callgrind, dhat, serde_json
    ├── src/
    │   ├── lib.rs                # pub mod baseline; pub mod fixtures;
    │   ├── baseline.rs           # BaselineRecord serde struct (D-02 schema)
    │   └── fixtures.rs           # make_seeded_series(n, seed) → TimeSeries
    ├── benches/
    │   └── iai_suite.rs          # [[bench]] harness=false; iai hot paths
    └── tests/
        └── dhat_peak.rs          # #[global_allocator] dhat::Alloc; peak tests

benches/
└── baseline_suite.rs             # NEW: criterion tracked baselines (D-06)
                                  # (existing 8 benches stay, not baseline-tracked)

scripts/
├── update_criterion.sh           # NEW: capture → .planning/baselines/criterion.json
├── update_iai.sh                 # NEW: capture → .planning/baselines/iai.json
├── update_dhat.sh                # NEW: capture → .planning/baselines/dhat.json
└── update_wasm_size.sh           # NEW: build + measure → .planning/baselines/wasm_size.json

.planning/baselines/
├── criterion.json                # Wall-clock (median+MAD, local only)
├── iai.json                      # Instruction counts (CI gate)
├── dhat.json                     # Peak memory bytes (CI gate)
└── wasm_size.json                # WASM binary bytes (CI gate)

.github/workflows/
├── ci.yml                        # Existing — unchanged
├── bench.yml                     # NEW: iai gate + dhat informational
└── wasm-size.yml                 # NEW: WASM size gate
```

### Pattern 1: iai-callgrind Instruction-Count Benchmark

**What:** Callgrind-instrumented one-shot bench measuring CPU instruction count for a single function invocation.
**When to use:** Any hot path where regressions should block CI (D-10).

```rust
// crates/anofox-bench-harness/benches/iai_suite.rs
// Source: docs.rs/iai-callgrind [CITED: docs.rs/iai-callgrind/0.16.1/iai_callgrind/]

use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use iai_callgrind::{Callgrind, EventKind};
use std::hint::black_box;
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
use anofox_bench_harness::fixtures::make_seeded_series;

fn setup_auto_ets() -> (TimeSeries, AutoETS) {
    let ts = make_seeded_series(200, 42);
    let mut model = AutoETS::with_period(12);
    (ts, model)
}

#[library_benchmark]
#[bench::n200(setup_auto_ets())]
fn bench_auto_ets_fit((ts, mut model): (TimeSeries, AutoETS)) {
    black_box(model.fit(black_box(&ts)).unwrap());
}

library_benchmark_group!(
    name = compute_hot_paths;
    config = LibraryBenchmarkConfig::default()
        .callgrind(Callgrind::default()
            .soft_limits([(EventKind::Ir, 1.0f64)])  // fail CI if Ir rises > 1%
        );
    benchmarks = bench_auto_ets_fit
);

main!(library_benchmark_groups = compute_hot_paths);
```

**Cargo.toml for harness crate:**
```toml
[[bench]]
name = "iai_suite"
harness = false
```

### Pattern 2: dhat Peak-Memory Test

**What:** Rust integration test using dhat as global allocator, asserting peak bytes < baseline × 1.15.
**When to use:** Major model families per D-12.

```rust
// crates/anofox-bench-harness/tests/dhat_peak.rs
// Source: docs.rs/dhat [CITED: docs.rs/dhat/latest/dhat/]

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
fn auto_ets_peak_memory_within_baseline() {
    let _profiler = dhat::Profiler::builder().testing().build();
    
    let ts = make_seeded_series(1000, 42);
    let mut model = AutoETS::with_period(12);
    model.fit(&ts).unwrap();
    let _forecast = model.predict(12).unwrap();
    
    let stats = dhat::HeapStats::get();
    // Baseline from dhat.json × 1.15 headroom (D-12)
    let baseline_bytes: u64 = load_dhat_baseline("auto_ets_n1000");
    assert!(
        stats.max_bytes <= (baseline_bytes as f64 * 1.15) as usize,
        "AutoETS peak memory {} bytes exceeds baseline {} × 1.15",
        stats.max_bytes,
        baseline_bytes
    );
}
```

**Note:** `dhat::Alloc` replaces the system allocator for the entire test binary. Only one `#[global_allocator]` is allowed per test binary — the dhat tests must live in their own test file.

### Pattern 3: Criterion Baseline Capture and JSON Extraction

**What:** Run criterion benchmark suite locally, parse `target/criterion/*/new/estimates.json`, write provenance-stamped `criterion.json`.
**When to use:** `scripts/update_criterion.sh` only; never in CI.

```bash
#!/usr/bin/env bash
# scripts/update_criterion.sh
set -euo pipefail

# Capture baselines — native parallel profile
cargo bench --bench baseline_suite --features parallel -- --save-baseline committed

# Capture WASM/no-rayon profile
cargo bench --bench baseline_suite -- --save-baseline committed_no_parallel

# Parse estimates and build criterion.json with provenance
RUST_VERSION=$(rustc --version)
GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CPU_INFO=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)

python3 scripts/_build_criterion_json.py \
  --baseline-dir target/criterion \
  --git-sha "$GIT_SHA" \
  --rustc "$RUST_VERSION" \
  --timestamp "$TIMESTAMP" \
  --cpu "$CPU_INFO" \
  --output .planning/baselines/criterion.json
```

**`target/criterion/<bench_name>/new/estimates.json` structure (criterion 0.5):**
```json
{
  "mean":   { "confidence_interval": {...}, "point_estimate": 123456.7, "standard_error": 456.0 },
  "median": { "confidence_interval": {...}, "point_estimate": 122000.0, "standard_error": 0.0 },
  "median_abs_dev": { "confidence_interval": {...}, "point_estimate": 800.0, "standard_error": 0.0 },
  "slope":  { ... },
  "std_dev": { ... }
}
```
Extract `median.point_estimate` and `median_abs_dev.point_estimate` for the baseline.

### Pattern 4: WASM Size Gate Workflow

**What:** CI workflow that builds release WASM, compares size to committed baseline, fails if > 1% growth.
**When to use:** Every PR merge attempt.

```yaml
# .github/workflows/wasm-size.yml
name: WASM Size Gate

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

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
        run: |
          wasm-pack build crates/anofox-forecast-js --target web \
            --out-dir ../../js --out-name anofox_forecast_js --release
      - name: Restore custom package.json and README
        run: git checkout js/package.json js/README.md
      - name: Check WASM size gate
        run: |
          CURRENT=$(stat --format=%s js/anofox_forecast_js_bg.wasm)
          BASELINE=$(python3 -c "import json; d=json.load(open('.planning/baselines/wasm_size.json')); print(d['bytes'])")
          DELTA=$(python3 -c "print(($CURRENT - $BASELINE) / $BASELINE * 100)")
          echo "WASM size: current=${CURRENT}B baseline=${BASELINE}B delta=${DELTA}%"
          python3 -c "
          delta = ($CURRENT - $BASELINE) / $BASELINE * 100
          if delta > 1.0:
              raise SystemExit(f'WASM size grew {delta:.2f}% (threshold: 1%)')
          "
```

### Pattern 5: Baseline Provenance Schema (D-02)

**What:** Serde struct used in the harness crate for all four baseline files.

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
pub struct CriterionBaseline {
    pub provenance: ProvenanceFingerprint,
    pub benchmarks: Vec<CriterionEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CriterionEntry {
    pub name: String,             // e.g. "auto_ets_fit_n200_parallel"
    pub profile: String,          // "parallel" | "no_parallel"
    pub median_ns: f64,
    pub mad_ns: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IaiBaseline {
    pub provenance: ProvenanceFingerprint,
    pub benchmarks: Vec<IaiEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IaiEntry {
    pub name: String,             // e.g. "bench_auto_ets_fit::n200"
    pub instruction_count: u64,   // Ir from Callgrind
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DhatBaseline {
    pub provenance: ProvenanceFingerprint,
    pub benchmarks: Vec<DhatEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DhatEntry {
    pub name: String,             // e.g. "auto_ets_n1000"
    pub peak_bytes: u64,          // HeapStats::get().max_bytes
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WasmSizeBaseline {
    pub provenance: ProvenanceFingerprint,
    pub filename: String,         // "anofox_forecast_js_bg.wasm"
    pub bytes: u64,
}
```

### Pattern 6: Workspace Harness Crate Registration

**What:** Add `crates/anofox-bench-harness` as a workspace member.
**When to use:** D-04 requires a single typed owner for baseline schema.

```toml
# Root Cargo.toml (root package section unchanged)
[workspace]
members = [".", "crates/anofox-forecast-js", "crates/anofox-bench-harness"]
resolver = "2"
```

```toml
# crates/anofox-bench-harness/Cargo.toml
[package]
name = "anofox-bench-harness"
version = "0.1.0"
edition = "2021"
publish = false  # internal tooling only

[lib]
name = "anofox_bench_harness"

[dependencies]
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

[[bench]]
name = "baseline_suite"
harness = false
```

### Anti-Patterns to Avoid

- **Capturing criterion baselines on GitHub Actions:** Wall-clock timing noise on virtual runners makes numbers non-reproducible (D-03 forbids it). Use `--save-baseline` only in `update_criterion.sh` on a quiet local machine.
- **Using `wee_alloc` as the global allocator:** Banned — archived August 2025 with known memory leaks. Use `dhat::Alloc` for profiling tests, standard allocator otherwise [VERIFIED: STATE.md].
- **One `#[global_allocator]` per test binary (dhat):** `dhat::Alloc` must be the only `#[global_allocator]` in a binary. Put dhat tests in a dedicated file (e.g., `tests/dhat_peak.rs`), not mixed with regular integration tests.
- **Pinning beta/nightly for iai-callgrind gate:** D-10 locks the iai gate to **stable Rust only**. Beta/nightly drift in instruction counts is expected; gating on them causes false positives.
- **CI writing baselines:** Never let any CI step write to `.planning/baselines/`. CI always reads; only `scripts/update_*.sh` on a developer machine writes.
- **Committing WASM size baseline before PERF-06:** The 10 dead-code compiler warnings indicate unused methods that contribute to binary size. Remove them first, then capture the baseline.
- **Using cargo-machete to detect dead Rust functions:** cargo-machete only scans `[dependencies]` in `Cargo.toml` for unused package imports. Dead Rust *methods* (like the 8 unused `inner()` methods) are caught by the `rustc` dead_code lint, not by machete. Use `cargo build -p anofox-forecast-js 2>&1 | grep "method.*is never used"` to confirm.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Instruction-count measurement | Custom valgrind wrapper | iai-callgrind 0.16.1 | Handles callgrind instrumentation, result parsing, regression comparison, CI integration |
| Heap peak-memory assertion | Custom allocator wrapper | dhat 0.3.3 | dhat::Alloc + HeapStats::get() is the standard approach; handles thread-safety and drop semantics |
| WASM binary optimization | Custom wasm-opt invocation | wasm-pack profile (already configured in Cargo.toml) | `[package.metadata.wasm-pack.profile.release]` with `wasm-opt = ["-O3"]` already in `crates/anofox-forecast-js/Cargo.toml` |
| Bench fixture generation | Random number generator per bench | Shared `make_seeded_series(n, seed)` in harness crate | LCG-based generators already in existing benches; formalize into one place (D-08) |
| Baseline JSON schema | Ad-hoc per-script structs | `ProvenanceFingerprint` + typed structs in harness crate | Prevents schema drift across 4 dimensions; single owner |

**Key insight:** The hardest part of this phase is not the tooling setup — it's the sequencing discipline. Each tool has a clearly documented setup; the risk is letting CI capture wall-clock baselines or committing WASM size before dead code is removed.

---

## Dead Code Inventory (PERF-06)

The following dead code was **confirmed by a live build** (`cargo build -p anofox-forecast-js --target wasm32-unknown-unknown 2>&1`) [VERIFIED: live build run this session]:

**File: `crates/anofox-forecast-js/src/forecaster.rs`**

8 unused `inner()` methods (pub(crate), never called across the crate):
- Line 384: `impl SESForecaster { pub(crate) fn inner(&self) -> &SimpleExponentialSmoothing }`
- Line 434: `impl HoltForecaster { pub(crate) fn inner(&self) -> &HoltLinearTrend }`
- Line 517: `impl HoltWintersForecaster { pub(crate) fn inner(&self) -> &HoltWinters }`
- Line 1209: `impl CrostonForecaster { pub(crate) fn inner(&self) -> &Croston }`
- Line 1261: `impl TSBForecaster { pub(crate) fn inner(&self) -> &TSB }`
- Line 1315: `impl ADIDAForecaster { pub(crate) fn inner(&self) -> &ADIDA }`
- Line 1369: `impl IMAPAForecaster { pub(crate) fn inner(&self) -> &IMAPA }`
- Line 1633: `impl GARCHForecaster { pub(crate) fn inner(&self) -> &GARCH }`

**File: `crates/anofox-forecast-js/src/laplace_playground.rs`**
- Line 15: `unused import: RecipeKind`

**File: `crates/anofox-forecast-js/src/forecaster.rs`**
- Line 2141: `variable does not need to be mutable` (let mut kf)

**Note:** The `inner()` methods in `time_series.rs` (line 144), `calendar.rs` (line 212), and `postprocess.rs` (line 67) are **used** — do not remove them.

**Removal approach:**
1. Delete the 8 unused `fn inner()` method blocks (bodies + impl sections if the `inner` method is the only one in that `impl` block).
2. Remove `RecipeKind` from the import in `laplace_playground.rs`.
3. Remove `mut` from the KalmanFilter binding in `forecaster.rs:2141`.
4. Verify: `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown 2>&1 | grep "warning:" | wc -l` should output 0.
5. Verify: `wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js` succeeds and the npm package still builds.
6. Then capture WASM size baseline.

---

## Common Pitfalls

### Pitfall 1: iai-callgrind Runner Version Mismatch

**What goes wrong:** The iai-callgrind runner binary (`iai-callgrind-runner`) must be the **exact same version** as the `iai-callgrind` crate. Mismatched versions cause the bench runner to exit with an error before any measurements are taken.
**Why it happens:** The library and runner use a versioned protocol; the runner validates the client version on startup.
**How to avoid:** In CI, always pin `cargo install iai-callgrind-runner --version 0.16.1 --locked`. In local setup docs, state the version explicitly.
**Warning signs:** Bench crashes immediately with a "version mismatch" error rather than a Callgrind measurement error.

### Pitfall 2: Valgrind Version Too Old on CI

**What goes wrong:** iai-callgrind requires valgrind >= 3.20.0. Ubuntu 22.04 ships valgrind 3.18, which is too old. Ubuntu 24.04 (now `ubuntu-latest`) ships 3.22 — sufficient.
**Why it happens:** GitHub Actions `ubuntu-latest` migrated to Ubuntu 24.04 in January 2025. Older pinned `ubuntu-22.04` would fail.
**How to avoid:** Use `runs-on: ubuntu-latest` (= Ubuntu 24.04) for bench.yml. If locking to `ubuntu-22.04` for any reason, install valgrind from source or a PPA.
**Warning signs:** iai-callgrind prints "Unsupported valgrind version" or similar at bench startup.

### Pitfall 3: dhat Global Allocator Conflicts

**What goes wrong:** A Rust test binary can have exactly one `#[global_allocator]`. If dhat tests are placed in the same test file as other integration tests that also declare a global allocator (or if another crate in the binary does), compilation fails.
**Why it happens:** Rust enforces single global allocator at the linker level.
**How to avoid:** All dhat tests live in a dedicated file (`tests/dhat_peak.rs`) with the `#[global_allocator]` declaration at the file root. No other integration test file in the harness crate should declare a global allocator.
**Warning signs:** Compiler error "cannot have more than one `#[global_allocator]`".

### Pitfall 4: Criterion Wall-Clock Noise in CI

**What goes wrong:** If `update_criterion.sh` is accidentally called in CI (e.g., a CI job runs `cargo bench`), criterion writes to `target/criterion/` which is ephemeral. The committed `.planning/baselines/criterion.json` would show drift unrelated to code changes.
**Why it happens:** Criterion benchmarks run normally under `cargo bench` with no special flag.
**How to avoid:** `bench.yml` must NOT run `cargo bench --bench baseline_suite`. Only the iai bench runs in CI. The script `update_criterion.sh` documentation explicitly states "run on a quiet local machine."
**Warning signs:** CI shows >5% criterion drift between runs on the same commit.

### Pitfall 5: WASM Baseline Captured Before Dead-Code Removal

**What goes wrong:** If `update_wasm_size.sh` is run before removing the 8 unused `inner()` methods, the baseline overstates the actual production binary size. Future improvements that remove more dead code will appear to be "size regressions" because the baseline was too large.
**Why it happens:** Forgetting the PERF-06 → PERF-05 sequencing constraint.
**How to avoid:** `update_wasm_size.sh` must check that `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown 2>&1 | grep "warning:" | wc -l` returns 0 before capturing size. Add this check as the first step of the script.
**Warning signs:** `cargo build -p anofox-forecast-js` emits "method `inner` is never used" warnings.

### Pitfall 6: Native-Parallel vs WASM Profile Confusion

**What goes wrong:** The `parallel` feature activates Rayon for multi-threaded batch fitting. WASM targets **cannot** use `parallel` — it is Rayon-backed and panics in WASM. If criterion baselines mix parallel and single-thread timings, the two profiles are incomparable.
**Why it happens:** The existing criterion benches do not explicitly separate parallel vs non-parallel runs.
**How to avoid:** D-09 requires separate baseline sections: run `cargo bench --bench baseline_suite --features parallel` for native-parallel, and `cargo bench --bench baseline_suite` (no Rayon) for the single-thread / WASM-comparable profile. The `criterion.json` schema uses a `"profile"` field (`"parallel"` | `"no_parallel"`).
**Warning signs:** Batch-100 timings look unrealistically fast in one profile vs another.

---

## Code Examples

### Seeded Fixture Generation (Harness Crate)

```rust
// crates/anofox-bench-harness/src/fixtures.rs
use anofox_forecast::core::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

/// Creates a deterministic synthetic time series with trend + seasonality + noise.
/// Uses an LCG with the provided seed — matches the pattern in existing benches.
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
            trend + seasonal + noise + 20.0  // +20 keeps values positive for multiplicative models
        })
        .collect();

    TimeSeries::univariate(timestamps, values).unwrap()
}
```

### iai-callgrind Bench File Skeleton

```rust
// crates/anofox-bench-harness/benches/iai_suite.rs
use iai_callgrind::{
    library_benchmark, library_benchmark_group, main,
    LibraryBenchmarkConfig, Callgrind, EventKind,
};
use std::hint::black_box;
use anofox_bench_harness::fixtures::make_seeded_series;
use anofox_forecast::models::{
    arima::AutoARIMA,
    exponential::AutoETS,
    Forecaster,
};
use anofox_forecast::batch;

fn setup_ts_n200() -> anofox_forecast::core::TimeSeries {
    make_seeded_series(200, 42)
}

#[library_benchmark]
#[bench::n200(setup_ts_n200())]
fn bench_auto_ets_fit(ts: anofox_forecast::core::TimeSeries) {
    let mut model = AutoETS::with_period(12);
    black_box(model.fit(black_box(&ts)).unwrap());
}

#[library_benchmark]
#[bench::n200(setup_ts_n200())]
fn bench_auto_arima_fit(ts: anofox_forecast::core::TimeSeries) {
    let mut model = AutoARIMA::new();
    black_box(model.fit(black_box(&ts)).unwrap());
}

fn setup_batch_100() -> Vec<anofox_forecast::core::TimeSeries> {
    (0..100).map(|seed| make_seeded_series(100, seed as u64)).collect()
}

#[library_benchmark]
#[bench::batch100(setup_batch_100())]
fn bench_batch_100(series: Vec<anofox_forecast::core::TimeSeries>) {
    black_box(batch::auto_ets(black_box(&series), 12).unwrap());
}

library_benchmark_group!(
    name = hot_paths;
    config = LibraryBenchmarkConfig::default()
        .callgrind(Callgrind::default()
            .soft_limits([(EventKind::Ir, 1.0f64)]));
    benchmarks = bench_auto_ets_fit, bench_auto_arima_fit, bench_batch_100
);

main!(library_benchmark_groups = hot_paths);
```

### Criterion Baseline Suite Skeleton

```rust
// benches/baseline_suite.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use anofox_forecast::models::{
    arima::AutoARIMA,
    exponential::AutoETS,
    theta::AutoTheta,
    baseline::Naive,
    intermittent::Croston,
    ensemble::AutoEnsemble,
    laplace::LaplaceForecaster,
    Forecaster,
};
use anofox_forecast::core::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

fn make_seeded_series(n: usize, seed: u64) -> TimeSeries {
    // Same logic as harness crate fixture — copy for now, share via harness crate later
    // (benches/ cannot depend on a crate in crates/ that has harness crate as a sibling;
    //  alternatively, add anofox-bench-harness as a dev-dependency of the root crate)
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n)
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    let mut rng: u64 = seed;
    let values: Vec<f64> = (0..n).map(|i| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
        5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() + 0.05 * i as f64 + noise + 20.0
    }).collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn bench_auto_ets(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("auto_ets_fit_n200", |b| b.iter(|| {
        let mut m = AutoETS::with_period(12);
        m.fit(black_box(&ts)).unwrap();
    }));
}

// ... repeat for all 7 families × {n=100, n=1000} × {single, batch-100}

criterion_group!(
    name = baselines;
    config = Criterion::default().sample_size(20);  // standardized sample size
    targets = bench_auto_ets /*, bench_auto_arima, ... */
);
criterion_main!(baselines);
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `wee_alloc` global allocator | System allocator (or `dhat::Alloc` for profiling) | Aug 2025 (wee_alloc archived) | wee_alloc banned — do not use |
| `iai` (original bheisler crate) | `iai-callgrind` (clockworklabs fork → iai-callgrind org) | 2023+ | iai-callgrind has active maintenance; original `iai` is unmaintained |
| `iai-callgrind` crate name | `gungraun` crate name | 2026 (renaming) | On crates.io, `iai-callgrind 0.16.1` is the stable version; `gungraun 0.19.4` is the new name — do not switch without deliberate migration |
| Criterion CSV output | Criterion JSON in `target/criterion/*/new/estimates.json` | criterion 0.4+ | CSV removed; parse JSON for baselines |
| `ubuntu-latest` = Ubuntu 22.04 | `ubuntu-latest` = Ubuntu 24.04 | Jan 2025 | Valgrind 3.22 (sufficient for iai-callgrind ≥3.20 requirement) now available via `apt` |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | cargo test (built-in) + criterion 0.5 (benches) + iai-callgrind 0.16.1 (instruction benches) |
| Config file | None (Cargo.toml [[bench]] entries) |
| Quick run command | `cargo test -p anofox-bench-harness` |
| Full suite command | `cargo bench --bench baseline_suite && cargo bench --bench iai_suite` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MEAS-01 | `.planning/baselines/*.json` files exist and are valid JSON | smoke | `python3 -c "import json; [json.load(open(f)) for f in ('.planning/baselines/criterion.json', '.planning/baselines/iai.json', '.planning/baselines/dhat.json', '.planning/baselines/wasm_size.json')]"` | ❌ Wave 0 |
| MEAS-02 | `scripts/update_*.sh` scripts are executable and documented | smoke | `ls -la scripts/update_*.sh` | ❌ Wave 0 |
| MEAS-03 | bench.yml and wasm-size.yml are green | CI gate | GitHub Actions check | ❌ Wave 0 |
| MEAS-04 | No new files in `src/` | smoke | `git diff --name-only main | grep '^src/'` | N/A |
| PERF-01 | criterion baseline_suite bench runs for all 7 families | criterion bench | `cargo bench --bench baseline_suite` | ❌ Wave 0 |
| PERF-02 | iai gate green; Ir within 1% | CI gate | CI bench.yml | ❌ Wave 0 |
| PERF-03 | criterion.json has "parallel" and "no_parallel" sections | unit | `python3 -c "import json; d=json.load(open('.planning/baselines/criterion.json')); profiles=set(e['profile'] for e in d['benchmarks']); assert 'parallel' in profiles and 'no_parallel' in profiles"` | ❌ Wave 0 |
| PERF-04 | dhat peak-memory tests pass | integration | `cargo test -p anofox-bench-harness --test dhat_peak` | ❌ Wave 0 |
| PERF-05 | wasm-size.yml gate is green | CI gate | GitHub Actions check | ❌ Wave 0 |
| PERF-06 | Zero dead-code warnings in anofox-forecast-js | smoke | `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown 2>&1 \| grep "warning:" \| wc -l` (expect: 0) | pre-req |

### Wave 0 Gaps (files that must be created before Wave 1)

- [ ] `crates/anofox-bench-harness/Cargo.toml` + `src/lib.rs` + `src/baseline.rs` + `src/fixtures.rs`
- [ ] `crates/anofox-bench-harness/benches/iai_suite.rs`
- [ ] `crates/anofox-bench-harness/tests/dhat_peak.rs`
- [ ] `benches/baseline_suite.rs`
- [ ] `scripts/update_criterion.sh`, `update_iai.sh`, `update_dhat.sh`, `update_wasm_size.sh`
- [ ] `.github/workflows/bench.yml`
- [ ] `.github/workflows/wasm-size.yml`
- [ ] `.planning/baselines/` directory (needs to exist before scripts run)

### Sampling Rate

- **Per task commit:** `cargo build -p anofox-bench-harness` (compile check)
- **Per wave merge:** `cargo test -p anofox-bench-harness` + `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown 2>&1 | grep "warning:" | wc -l` (expect 0)
- **Phase gate:** All 4 baseline JSON files committed + CI workflows green

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All | ✓ | 1.97.0 | — |
| cargo | All | ✓ | 1.97.0 | — |
| wasm-pack | PERF-05, PERF-06 | ✗ (auto-installs via Makefile) | — | `curl ... \| sh` (Makefile already does this) |
| valgrind | PERF-02 (CI) | ✗ (local) | — | CI: `sudo apt-get install -y valgrind` on ubuntu-latest (3.22 available) |
| wasm32-unknown-unknown target | WASM builds | ✓ (via dtolnay/rust-toolchain in CI) | — | `rustup target add wasm32-unknown-unknown` |
| python3 | scripts/update_criterion.sh baseline JSON parsing | ✓ | — | Rewrite in bash with jq if unavailable |
| jq (optional) | Baseline parsing alternative | ✗ | — | Use python3 |
| Chrome (headless) | WASM tests (existing ci.yml) | CI only | — | Already handled in existing ci.yml |

**Missing dependencies with no fallback:**
- None that block phase execution. valgrind is absent locally but not needed for local development; the CI installs it.

**Missing dependencies with fallback:**
- `wasm-pack`: auto-installed by `install-wasm-pack` Makefile target.
- `valgrind`: not available locally; `apt-get install valgrind` on CI.

---

## Security Domain

`security_enforcement` is enabled (absent in config = enabled). This phase adds only dev-tooling and shell scripts — no production code paths, no user input handling.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (scripts parse JSON) | `python3 json.load()` raises on malformed JSON; fail-fast |
| V6 Cryptography | no | — |

### Known Threat Patterns for Shell Scripts + CI Workflows

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Arbitrary code via `cargo install` with network access | Tampering | Pin `--version 0.16.1 --locked` for iai-callgrind-runner |
| Baseline file injection (CI writes baselines) | Tampering | CI is read-only; only `scripts/update_*.sh` on local machines write baselines |
| WASM binary substitution via artifact upload | Tampering | wasm-size.yml measures the build output directly; no artifact upload/download |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `ubuntu-latest` = Ubuntu 24.04 (valgrind 3.22 via apt) | Environment Availability | If GitHub reverts to 22.04, valgrind 3.18 is too old; add `ubuntu-24.04` explicit pin to bench.yml |
| A2 | criterion 0.5 writes `target/criterion/<name>/new/estimates.json` with `median` and `median_abs_dev` fields | Pattern 3 / Code Examples | If field names changed in 0.5, the parsing script produces wrong values; verify by running `cargo bench --bench baseline_suite` locally and inspecting the JSON |
| A3 | `iai-callgrind 0.16.1` (crates.io stable) and `gungraun 0.19.4` (GitHub, new name) are compatible migration paths — the 0.16.1 API as documented works | Standard Stack | If 0.16.1 has regressions fixed in 0.19.x, need to either migrate or work around; check changelog before pinning |
| A4 | `soft_limits([(EventKind::Ir, 1.0f64)])` means 1% threshold on instruction count | Pattern 1 / iai-callgrind gate | If the API signature changed between 0.16.0 and 0.16.1, the code won't compile; verify by checking docs.rs for 0.16.1 |
| A5 | `batch::auto_ets()` accepts a `&[TimeSeries]` or `Vec<TimeSeries>` signature for batch-100 benchmark | Pattern 1 / Code Examples | Actual API may differ; verify via `cargo doc -p anofox-forecast` |
| A6 | The `anofox-bench-harness` crate can add `anofox-forecast` as a `[dependencies]` entry (not just dev-dep) for the `src/fixtures.rs` module | Pattern 6 | If workspace dependency resolution conflicts, fixtures may need to be inlined into each test/bench file |

---

## Open Questions

1. **batch::auto_ets() API signature**
   - What we know: `src/batch.rs` exports `auto_ets()`, `auto_arima()`, `auto_theta()` for parallel series
   - What's unclear: Exact function signature (input type, seasonal period arg, return type)
   - Recommendation: Read `src/batch.rs` before writing the iai batch-100 bench; the planner should include a task to verify the signature

2. **iai instruction count baseline capture — first-run behavior**
   - What we know: iai-callgrind on first run has no baseline to compare against; it prints results without regression failure
   - What's unclear: Does iai-callgrind write the baseline automatically on first run (to a local directory), or does the `update_iai.sh` script need to parse stdout and produce the JSON?
   - Recommendation: The planner should include a task to investigate iai-callgrind's `--save-baseline` equivalent or stdout parsing; the harness crate may need to parse iai output rather than relying on a built-in save mechanism

3. **Criterion baseline_suite dev-dependency on anofox-bench-harness**
   - What we know: `benches/baseline_suite.rs` is a bench of the root crate; `crates/anofox-bench-harness` is a sibling workspace crate
   - What's unclear: Can the root crate's `[dev-dependencies]` include a sibling workspace crate (`anofox-bench-harness = { path = "crates/anofox-bench-harness" }`) without circular dependency? The harness crate depends on `anofox-forecast` (the root crate).
   - Recommendation: To avoid the circular dependency, `baseline_suite.rs` should inline the fixture generation (the seeded LCG pattern is 10 lines), or fixtures.rs should be moved to a separate tiny crate with no upward dependency

---

## Sources

### Primary (MEDIUM confidence)
- [docs.rs/iai-callgrind/0.16.1](https://docs.rs/iai-callgrind/0.16.1/iai_callgrind/) — library_benchmark macros, soft_limits API, EventKind::Ir, setup patterns
- [docs.rs/dhat/latest](https://docs.rs/dhat/latest/dhat/) — Profiler::builder().testing().build(), HeapStats::get(), max_bytes field
- [crates.io package legitimacy tool](https://crates.io) — iai-callgrind 0.16.1, dhat 0.3.3, cargo-machete 0.9.2 all verified OK

### Secondary (MEDIUM confidence)
- [bheisler.github.io/criterion.rs](https://bheisler.github.io/criterion.rs/book/user_guide/command_line_options.html) — --save-baseline, --baseline, --load-baseline flags
- [github.com/iai-callgrind/iai-callgrind/releases](https://github.com/iai-callgrind/iai-callgrind/releases) — latest release info, gungraun rename confirmed
- Live build run: `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown` — confirmed 10 dead-code warnings with exact file:line references

### Tertiary (LOW confidence)
- WebSearch results on valgrind ubuntu 24.04, wasm-pack size measurement, GitHub Actions ubuntu-latest migration timeline
- Training knowledge on criterion JSON file structure and path conventions

---

## Metadata

**Confidence breakdown:**
- Standard stack (iai-callgrind, dhat, cargo-machete): MEDIUM — packages verified on crates.io via legitimacy gate; API cross-checked via docs.rs
- Architecture (patterns, workflow structure): MEDIUM — based on official docs + codebase read
- Dead code inventory: HIGH — confirmed via live build with exact line numbers
- Valgrind CI availability: MEDIUM — confirmed ubuntu-latest = 24.04 via web search; valgrind 3.22 availability on 24.04 from known package repos
- Criterion JSON structure: LOW — described in docs but not directly inspected this session (estimates.json path and field names from training + web)

**Research date:** 2026-08-09
**Valid until:** 2026-11-09 (stable toolchain, 90 days)
