# Architecture Research

**Domain:** In-repo measurement and benchmark harness for a mature Rust forecasting library
**Researched:** 2026-08-09
**Confidence:** MEDIUM (Rust tooling patterns well-established; WASM-specific measurement is LOW)

## Standard Architecture

### System Overview

The harness adds a measurement layer that sits alongside the library, not inside it. All
measurement code is strictly outside `src/` — in `benches/`, `tests/`, `harness/`, and a new
`.planning/baselines/` artifact store. The library crate itself is never modified except to expose
accessors needed by the harness (e.g., returning allocation counts from a fit call, which is
zero library-API surface change under a dev-dependency feature gate).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         External Inputs                                      │
│  validation/data/*.tsf *.csv (gitignored, 1.4 GB, already present)          │
│  CI environment / local developer machine                                   │
└──────────────┬──────────────────────────────────┬───────────────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐      ┌───────────────────────────────────────────┐
│   Speed / Instruction    │      │           Accuracy Harness                │
│   Benchmark Layer        │      │   harness/accuracy/  (feature-gated)      │
│                          │      │                                           │
│  benches/*.rs            │      │  Dataset loader (TSF/CSV parser)          │
│  (existing Criterion)    │      │  → TimeSeries::univariate()               │
│                          │      │  → model.fit() / model.predict()          │
│  + iai-callgrind benches │      │  → metrics: MASE, RMSE, sMAPE             │
│    for hot-path gates    │      │  → compare to baselines/accuracy.json     │
└──────────┬───────────────┘      └───────────────────────┬───────────────────┘
           │                                              │
           ▼                                              ▼
┌──────────────────────────┐      ┌───────────────────────────────────────────┐
│   Memory Measurement     │      │         Coverage Measurement              │
│   Layer                  │      │                                           │
│                          │      │  cargo llvm-cov (already in CI)           │
│  dhat::Alloc (native)    │      │  → lcov → Codecov (trend tracking)        │
│  tests/memory_*.rs       │      │  → JSON summary → baselines/coverage.json │
│  assert peak_bytes < N   │      │                                           │
└──────────┬───────────────┘      └───────────────────────┬───────────────────┘
           │                                              │
           ▼                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Baseline Artifact Store                                │
│                     .planning/baselines/                                     │
│                                                                             │
│  criterion/      ← critcmp --export JSON snapshots (one per bench group)    │
│  accuracy.json   ← per-model, per-dataset MASE/RMSE/sMAPE                  │
│  coverage.json   ← line/branch percent totals                               │
│  wasm_size.json  ← compiled .wasm byte count                                │
│  memory.json     ← peak_bytes per model family                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ read by CI regression scripts
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CI Integration Layer                                │
│                    .github/workflows/                                        │
│                                                                             │
│  ci.yml (existing) — standard test/clippy/audit gates, unchanged            │
│  bench.yml (new)   — criterion + iai-callgrind, scheduled + PR label        │
│  wasm-size.yml (new) — wasm-pack build, size delta check vs baseline        │
│  accuracy.yml (new)  — workflow_dispatch only; needs dataset dir env var    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Boundary | Responsibility | Notes |
|-----------|----------|----------------|-------|
| `benches/*.rs` (existing Criterion) | Harness | Wall-clock speed for fit+predict per model family, single + batch | Already exists; extend, don't replace |
| `benches/iai_*.rs` (new) | Harness | Instruction-count regression gate for 3-5 hot paths (AutoETS fit, ARIMA fit, batch 100-series) | Requires Valgrind in CI runner |
| `tests/memory_*.rs` (new) | Harness | Peak heap assertions using dhat::Alloc; native only, not WASM | dev-dependency: dhat |
| `harness/accuracy/` or `tests/accuracy_*.rs` (new) | Harness | Load TSF/CSV, fit models, compute MASE/RMSE/sMAPE, compare to .planning/baselines/accuracy.json | Gated behind env var ANOFOX_DATASET_DIR |
| `.planning/baselines/` (new) | Artifact store | Committed JSON snapshots for all baseline dimensions | Updated by maintainer after intentional improvement; never auto-committed by CI |
| `scripts/update_baselines.sh` (new) | Tooling | Re-run each measurement, write to .planning/baselines/, require explicit `git commit` | Maintainer-run, not CI |
| `.github/workflows/bench.yml` (new) | CI | Run criterion + iai-callgrind on schedule or PR label; fail if iai delta > threshold | Does NOT gate every PR (too slow/noisy) |
| `.github/workflows/wasm-size.yml` (new or extend existing) | CI | wasm-pack build --release; capture size; compare to baselines/wasm_size.json | Fail if +5% vs baseline |
| `.github/workflows/accuracy.yml` (new) | CI | workflow_dispatch only; requires dataset volume mounted; write accuracy report | Never blocks PR merges |
| `crates/anofox-forecast-js/` | Library (existing) | WASM bindings — no changes needed; size is measured from its build output | Unchanged |
| `src/` (library) | Library | Zero new measurement code here; harness imports the library as a black-box | Boundary must not be crossed |

## Recommended Project Structure

The additions slot into the existing layout without disrupting it:

```
anofox-forecast/
├── src/                          # Library — unchanged; harness is a consumer
│
├── benches/                      # Extended, not replaced
│   ├── (existing: arima_benchmark.rs, ets_benchmark.rs, ...)
│   ├── iai_hot_paths.rs          # NEW: iai-callgrind instruction-count gates
│   └── hot_paths.rs              # EXISTING: already present; verify it covers fit+predict
│
├── tests/                        # Extended with memory + robustness suites
│   ├── (existing integration tests)
│   ├── memory_native.rs          # NEW: dhat::Alloc peak-bytes assertions (native only)
│   ├── robustness_edge.rs        # NEW: NaN/Inf/singular/convergence edge cases
│   └── accuracy_harness.rs       # NEW: dataset-driven accuracy; gated on env var
│
├── harness/                      # NEW top-level crate or module (see note below)
│   └── (optional: if accuracy harness needs its own Cargo.toml for deps)
│
├── validation/data/              # EXISTING: dataset files (gitignored, ~1.4 GB)
│   └── (m3_monthly.tsf, m4_*.tsf, tourism_*.tsf, etc.)
│
├── .planning/baselines/          # NEW: committed baseline artifact store
│   ├── criterion/
│   │   ├── arima_fit.json        # critcmp --export snapshot
│   │   ├── ets_pool_search.json
│   │   └── batch_100_series.json
│   ├── iai/
│   │   └── hot_paths.json        # iai-callgrind instruction count snapshot
│   ├── accuracy.json             # {model, dataset, horizon, MASE, RMSE, sMAPE}
│   ├── coverage.json             # {lines_percent, branches_percent, ...}
│   ├── wasm_size.json            # {bytes: N, build_profile: "release", date: "..."}
│   └── memory.json               # {model, fit_peak_bytes, predict_peak_bytes}
│
├── scripts/
│   ├── (existing)
│   ├── update_criterion_baselines.sh   # NEW: runs cargo bench, exports via critcmp
│   ├── update_accuracy_baseline.sh     # NEW: runs accuracy harness, writes JSON
│   └── update_wasm_size_baseline.sh    # NEW: wasm-pack build, writes wasm_size.json
│
└── .github/workflows/
    ├── ci.yml                    # EXISTING: unchanged (test/clippy/audit/coverage)
    ├── deploy-playground.yml     # EXISTING: unchanged
    ├── npm.yml                   # EXISTING: unchanged
    ├── bench.yml                 # NEW: criterion + iai-callgrind (scheduled + label)
    ├── wasm-size.yml             # NEW: WASM size delta check on every PR
    └── accuracy.yml              # NEW: workflow_dispatch only; large dataset run
```

**Structure rationale:**

- `benches/` extension: Criterion is already there and working; adding iai-callgrind benches
  alongside keeps all speed measurement in one place. iai requires Valgrind (Linux-only), so
  the new bench.yml runs only on `ubuntu-latest`.

- `tests/` for memory and robustness: memory assertions via dhat fit naturally as `#[test]`
  functions — they assert on `HeapStats::max_bytes`, not on timing. Robustness tests similarly
  are `#[test]` functions, run in standard `cargo test`. No new test framework needed.

- `.planning/baselines/`: Committed JSON sidesteps the "target/ is gitignored" problem.
  critcmp can export Criterion baselines to a named JSON file placed anywhere. These committed
  files are the single source of truth that CI scripts compare against. Maintainers update them
  deliberately (via scripts/update_*.sh) after a proven improvement; a diff in these files is
  always intentional and reviewable.

- No harness/ sub-crate unless needed: If the accuracy harness requires additional dependencies
  (e.g., a TSF parser), and those deps are unwanted in `[dev-dependencies]` of the main crate,
  create a `crates/anofox-forecast-harness/` workspace member. Otherwise, keep accuracy tests
  in `tests/accuracy_harness.rs` with the dependency added to `[dev-dependencies]`.

## Architectural Patterns

### Pattern 1: Committed JSON Baselines for All Measurement Dimensions

**What:** Every measurable dimension (speed, memory, accuracy, coverage, WASM size) has a
corresponding JSON file in `.planning/baselines/`. CI reads the file, runs the measurement,
and fails if the delta exceeds a per-dimension threshold. Baselines are only updated by
maintainers running `scripts/update_*.sh` locally and committing the result.

**When to use:** Any dimension where "did this PR regress X?" needs a clear yes/no answer.

**Trade-offs:** Git history shows every intentional baseline update (good for audit trail);
baseline drift requires explicit human action (prevents silent regression acceptance). CI never
auto-commits baselines, which prevents feedback loops.

**Example schema for `.planning/baselines/accuracy.json`:**

```json
{
  "schema_version": 1,
  "generated": "2026-08-09",
  "results": [
    {
      "model": "AutoETS",
      "dataset": "m3_monthly",
      "horizon": 18,
      "n_series": 1428,
      "metrics": { "MASE": 1.12, "RMSE": 42.3, "sMAPE": 13.7 },
      "threshold": { "MASE_max": 1.25, "sMAPE_max": 15.0 }
    }
  ]
}
```

**Example schema for `.planning/baselines/wasm_size.json`:**

```json
{
  "schema_version": 1,
  "generated": "2026-08-09",
  "build_profile": "release",
  "features": ["js", "postprocess", "distributional", "anomaly"],
  "bytes": 2457600,
  "threshold_percent_increase": 5
}
```

### Pattern 2: Dual-Layer Speed Measurement (Criterion + iai-callgrind)

**What:** Criterion benches measure wall-clock time and are the primary developer-facing tool
for latency understanding. iai-callgrind benches measure instruction counts and are the CI
regression gate — instruction counts are deterministic regardless of runner hardware noise.
The two layers target different use cases and run on different schedules.

**When to use:** Projects where both "how fast is it really?" (Criterion) and "did this PR
change the hot path?" (iai-callgrind) matter. Criterion answers the first; iai answers the
second reliably in CI.

**Trade-offs:** iai-callgrind requires Valgrind (Linux only; not available in macOS runners).
Instruction counts do not perfectly predict wall-clock latency (cache effects differ). The
dual approach covers both needs without forcing a choice.

**iai-callgrind bench skeleton for `benches/iai_hot_paths.rs`:**

```rust
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
// ... TimeSeries construction

#[library_benchmark]
#[bench::m3_monthly_length_144(setup_m3_monthly())]
fn bench_auto_ets_fit(ts: TimeSeries) {
    let mut model = AutoETS::default();
    // iai runs this once; instruction count is the metric
    model.fit(&ts).unwrap();
}

library_benchmark_group!(name = hot_paths; benchmarks = bench_auto_ets_fit);
main!(library_benchmark_groups = hot_paths);
```

**CI bench.yml job structure:**

```yaml
bench:
  runs-on: ubuntu-latest
  if: contains(github.event.pull_request.labels.*.name, 'bench') || github.event_name == 'schedule'
  steps:
    - run: cargo bench --bench iai_hot_paths 2>&1 | tee iai_output.txt
    - run: python3 scripts/check_iai_regression.py iai_output.txt .planning/baselines/iai/hot_paths.json
```

### Pattern 3: Env-Var-Gated Accuracy Harness

**What:** The accuracy harness tests check for `ANOFOX_DATASET_DIR` at runtime. If the var is
absent, they call `return` (or use the `#[ignore]` attribute with `cargo test -- --ignored`).
If present, they load datasets from the path, run all registered model/dataset combinations,
compute metrics, and compare to `.planning/baselines/accuracy.json`.

**When to use:** Any test that requires large external data that cannot be committed to the
repo and is too slow for every-PR CI.

**Trade-offs:** Silent skip (no env var) means CI doesn't detect accuracy regressions on every
PR — only on scheduled/manual runs. This is the right trade-off: accuracy is expected to
change intentionally (a fix improves it, or a refactor should not change it). The accuracy
workflow is workflow_dispatch so maintainers run it deliberately before releases.

**Test skeleton for `tests/accuracy_harness.rs`:**

```rust
#[test]
fn auto_ets_m3_monthly_accuracy() {
    let data_dir = match std::env::var("ANOFOX_DATASET_DIR") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => return,  // skip silently when dataset not available
    };
    let series_list = load_tsf(data_dir.join("m3_monthly.tsf")).unwrap();
    let results = run_accuracy_sweep::<AutoETS>(series_list, 18);
    let baseline = load_baseline(".planning/baselines/accuracy.json", "AutoETS", "m3_monthly");
    assert!(
        results.mase <= baseline.threshold.mase_max,
        "AutoETS M3 monthly MASE {:.3} exceeds threshold {:.3}",
        results.mase, baseline.threshold.mase_max
    );
}
```

### Pattern 4: Native vs WASM Measurement Split

**What:** Native and WASM profiles are measured separately because they have different
capabilities and constraints:

- Native (parallel feature enabled): Criterion wall-clock, iai-callgrind instruction count,
  dhat peak-memory assertions. These use the full `--all-features` build.
- Native (no parallel): A second criterion benchmark group with `parallel` feature disabled,
  to simulate WASM-equivalent single-threaded behavior on native hardware.
- WASM (wasm32-unknown-unknown): Binary size tracking only. Runtime accuracy/speed measurement
  inside WASM requires a headless browser driver (wasm-pack test --headless --chrome), which is
  already wired in ci.yml; extending it for timing is impractical in CI (browser overhead
  dominates). WASM speed is instead inferred from the native no-parallel profile.

**When to use:** Any project with a WASM target where parallel behavior differs from native.
The key insight is: measure WASM what you can (size), and infer the rest from the equivalent
native profile.

**WASM size CI step (extend existing wasm.yml):**

```bash
wasm-pack build crates/anofox-forecast-js --target web --release
ACTUAL=$(wc -c < js/anofox_forecast_js_bg.wasm)
python3 scripts/check_wasm_size.py "$ACTUAL" .planning/baselines/wasm_size.json
```

## Data Flow

### Accuracy Harness Flow

```
ANOFOX_DATASET_DIR env var
    ↓
validation/data/{m3_monthly,m4_yearly,...}.tsf
    ↓ (TSF/CSV loader in tests/accuracy_harness.rs)
Vec<TimeSeries>  (one per competition series)
    ↓ (cross-validation via existing cross_validate())
Vec<CVResults>   (MASE, RMSE, sMAPE per fold)
    ↓ (aggregate across series)
AccuracyReport   {model, dataset, horizon, mean_MASE, mean_RMSE, mean_sMAPE}
    ↓ (compare to)
.planning/baselines/accuracy.json  →  pass / fail with diff
```

### Speed Benchmark Flow

```
cargo bench --bench {bench_name} [-- --save-baseline main]
    ↓ (Criterion statistical analysis)
target/criterion/{bench}/{fn}/main/   (Criterion internal storage)
    ↓ (critcmp --export main)
.planning/baselines/criterion/{bench_name}.json   (committed snapshot)
    ↓ (CI: critcmp --load baseline.json --load current.json)
Regression report  →  pass (within threshold) / fail
```

### Memory Measurement Flow

```
cargo test --test memory_native  (native, no wasm32 target)
    ↓ (#[global_allocator] static ALLOC: dhat::Alloc)
dhat::Profiler::new_heap()
    ↓ model.fit(&ts) + model.predict(h)
dhat::HeapStats::get().max_bytes
    ↓ (assert! vs threshold from .planning/baselines/memory.json)
pass / fail with delta message
```

### WASM Size Flow

```
wasm-pack build crates/anofox-forecast-js --target web --release
    ↓
js/anofox_forecast_js_bg.wasm   (binary)
    ↓ wc -c (bytes) + twiggy top (top contributors)
{bytes: N}
    ↓ (compare to .planning/baselines/wasm_size.json)
pass if delta <= +threshold_percent_increase
```

### Coverage Flow (already partially wired)

```
cargo llvm-cov --all-features --lcov --output-path lcov.info   (existing)
    ↓ (add) cargo llvm-cov --all-features --json --summary-only
coverage_summary.json
    ↓ extract lines_percent
compare to .planning/baselines/coverage.json  →  fail if decreased
```

## Anti-Patterns

### Anti-Pattern 1: Committing Baseline Snapshots Automatically from CI

**What people do:** CI job runs benchmarks, detects no regression, then commits an updated
baseline file back to the repo (using a "GitHub bot" commit).

**Why it's wrong:** Baselines drift silently. A series of small regressions each within
threshold accumulates into a large regression. The baseline becomes meaningless. Also causes
CI/CD loop instability and confusing git history.

**Do this instead:** CI only reads baselines — never writes them. Maintainers update baselines
deliberately by running `scripts/update_*.sh` locally after a proven improvement and committing
the diff as a separate PR. The PR review process for the baseline update is the human check.

### Anti-Pattern 2: Gating Every PR on Criterion Wall-Clock Benchmarks

**What people do:** CI blocks every PR merge if Criterion detects a >5% slowdown on any
benchmark.

**Why it's wrong:** Criterion measures wall-clock time, which is noisy on shared CI runners
(GitHub Actions ubuntu-latest has variable CPU frequency, co-tenant noise, etc.). This causes
flaky CI — PRs fail not because of regressions but because of runner variance. Developer trust
in CI erodes.

**Do this instead:** Use iai-callgrind (instruction counts, deterministic) for hard CI gates
on 3-5 critical hot paths. Use Criterion on a scheduled job or PR label ("bench") for
informational latency measurement without blocking merge.

### Anti-Pattern 3: Measuring WASM Performance via wasm-bindgen-test Timers

**What people do:** Write `wasm_bindgen_test` functions that call `js_sys::Date::now()` before
and after a model fit, assert the duration is under N ms.

**Why it's wrong:** Browser-driven test timing is dominated by WASM instantiation overhead,
GC pauses, and browser scheduling jitter. The measurement is far more variable than native
timing and not comparable across browsers or CI runs. Thresholds must be so loose as to be
useless.

**Do this instead:** Measure WASM speed by running the equivalent no-parallel native build
(same single-threaded code path, same data structures, no Rayon). WASM size is measurable
deterministically from the build output. Runtime performance inside WASM is inferred from the
native profile, not directly measured in CI.

### Anti-Pattern 4: Mixing Measurement Code into Library Source

**What people do:** Add `#[cfg(feature = "measure")]` blocks inside `src/models/*/` that
track counters, emit timings, or instrument allocations directly in the library code.

**Why it's wrong:** Instrumentation changes the code under measurement (observer effect).
It also pollutes the library API, complicates feature matrix, and risks shipping measurement
artifacts to end users.

**Do this instead:** Keep all measurement code strictly outside `src/`. The harness imports
the library as a consumer. For memory measurement, dhat::Alloc is a global allocator in the
test binary only — it does not modify library source. For instruction counting, iai-callgrind
wraps the binary from outside. The library is a black box to the harness.

### Anti-Pattern 5: Storing Baseline Artifacts in target/

**What people do:** Rely on Criterion's built-in `target/criterion/` storage for baseline
comparison in CI.

**Why it's wrong:** `target/` is gitignored and ephemeral in CI (even with Swatinem/rust-cache,
cache misses mean a cold runner has no baseline). The comparison has nothing to compare against
and either trivially passes or errors. This gives false confidence.

**Do this instead:** Use critcmp to export Criterion baselines to `.planning/baselines/criterion/`
(committed JSON). CI loads the committed file as the reference. This is reproducible regardless
of cache state.

## Integration Points

### Existing CI Touchpoints

| Workflow | Current State | Harness Addition |
|----------|--------------|-----------------|
| `ci.yml` test job | `cargo test --all-features` | No change; robustness tests (`tests/robustness_edge.rs`) run here automatically |
| `ci.yml` coverage job | `cargo llvm-cov ... --lcov` | Add `--json --summary-only` output; compare to `baselines/coverage.json` |
| `ci.yml` wasm job | `cargo build --target wasm32` | No change |
| `ci.yml` wasm-test job | `wasm-pack test --headless` | No change |
| `npm.yml` | wasm-pack build + publish | Add size capture step; compare to `baselines/wasm_size.json` |
| **bench.yml (new)** | — | Criterion + iai-callgrind; scheduled weekly + `bench` label |
| **accuracy.yml (new)** | — | workflow_dispatch; requires dataset env var; writes report |

### Harness-to-Library Boundary Rules

| Boundary | Rule | Enforcement |
|----------|------|-------------|
| Harness imports library | Via `use anofox_forecast::...` in `[dev-dependencies]` | Standard Cargo workspace |
| Library has no harness deps | `dhat`, `iai-callgrind` only in `[dev-dependencies]` | `cargo publish` dry-run check |
| Measurement code location | Only in `benches/`, `tests/`, `harness/`, `scripts/` | Code review convention |
| WASM target excludes dhat | `dhat` behind `#[cfg(not(target_arch = "wasm32"))]` in test | Compiler enforced |

## Build Order Implications

The harness must be built before improvements — you cannot prove a before/after delta without
a before measurement. The recommended phase sequence:

1. **Harness infrastructure first** — establish `.planning/baselines/` layout, wire CI jobs,
   confirm each measurement produces stable output with existing code. This is the foundation
   that makes all subsequent improvement phases trustworthy.

2. **Baseline capture** — run each harness with the current codebase; commit initial baselines
   to `.planning/baselines/`. These become the "before" numbers for every improvement.

3. **Improvement phases** — each improvement phase runs the relevant harness subset at
   completion to produce a documented delta. The updated baseline is committed as part of
   the improvement PR.

4. **Regression gates in CI are activated after baselines are committed** — enabling the
   iai-callgrind gate or WASM size check before baselines exist would immediately fail CI.
   Stage the gate enablement: add it to bench.yml only after the first baseline JSON is
   committed.

## Sources

- Criterion.rs command-line documentation (bheisler.github.io/criterion.rs)
- critcmp (github.com/BurntSushi/critcmp) — baseline export to JSON
- iai-callgrind (crates.io/crates/iai-callgrind) — Valgrind instruction-count benchmarking
- dhat (docs.rs/dhat) — HeapStats.max_bytes for peak allocation testing
- The Rust Performance Book (nnethercote.github.io/perf-book) — benchmarking, heap, profiling
- Rust and WebAssembly Book (rustwasm.github.io) — code size, twiggy, wasm-opt
- Existing project: .planning/codebase/ARCHITECTURE.md, TESTING.md, STRUCTURE.md (2026-08-09)

---
*Architecture research for: anofox-forecast measurement/benchmark harness*
*Researched: 2026-08-09*
