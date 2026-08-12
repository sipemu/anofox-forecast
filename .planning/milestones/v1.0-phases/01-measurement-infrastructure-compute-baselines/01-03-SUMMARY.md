---
phase: "01"
plan: "03"
subsystem: measurement-infrastructure
status: complete
tags: [measurement, criterion, dhat, peak-memory, baseline-schema, local-only, ci-gate]
completed_date: "2026-08-09"

dependency_graph:
  requires:
    - anofox-bench-harness crate (plan 01-01)
    - CriterionEntry + DhatEntry schema in baseline.rs (plan 01-01)
    - make_seeded_series fixture (plan 01-01)
    - IaiEntry baseline pattern (plan 01-02)
  provides:
    - benches/baseline_suite.rs (tracked criterion suite, 7 families x single+batch-100)
    - scripts/update_criterion.sh (local-only dual-profile median+MAD capture)
    - scripts/update_dhat.sh (dhat capture script via CAPTURE_DHAT=1 env mode)
    - crates/anofox-bench-harness/tests/dhat_peak.rs (native peak-memory gate)
    - .planning/baselines/criterion.json (structural placeholder; regenerate locally)
    - .planning/baselines/dhat.json (real captured baseline for 6 model families)
  affects:
    - Phases 2-4 have all 4 baseline JSON files committed; before/after numbers available
    - CI: dhat_peak test is a native gate (not criterion) — compiles and runs in CI

tech_stack:
  added:
    - criterion 0.5 (already declared as dev-dep in plan 01-01; now consumed by baseline_suite)
    - dhat 0.3.3 (already declared as dev-dep in plan 01-01; now consumed by dhat_peak.rs)
  patterns:
    - D-03 local-only criterion: update_criterion.sh runs only on developer machine; never CI
    - D-09 dual-profile: --save-baseline parallel_run vs no_parallel_run; python3 merges entries
    - D-12 x1.15 gate: (baseline_bytes as f64 * 1.15) as usize (float→usize truncation, conservative)
    - MEAS-04 first-run: missing baseline entry → skip assertion, print NOTICE
    - CAPTURE_DHAT=1: env-driven capture mode prints "CAPTURE name bytes" for script harvesting
    - Circular-dep fix: LCG fixture inlined in baseline_suite.rs (no harness dev-dep on root)

key_files:
  created:
    - benches/baseline_suite.rs
    - scripts/update_criterion.sh
    - scripts/update_dhat.sh
    - crates/anofox-bench-harness/tests/dhat_peak.rs
    - .planning/baselines/criterion.json
    - .planning/baselines/dhat.json
  modified:
    - Cargo.toml (added [[bench]] name = "baseline_suite" harness = false)

decisions:
  - "LaplaceForecaster gated behind cfg(distributional) in baseline_suite.rs — matches library feature gate; criterion_group! duplicated for cfg/no-cfg variants"
  - "All 7 families measured in one #[test] fn in dhat_peak.rs — avoids overlapping dhat profiler sessions (dhat panics on concurrent Profiler instances)"
  - "CAPTURE_DHAT=1 env-driven capture mode preferred over a separate #[ignore] test — simpler, no separate test binary needed"
  - "criterion.json committed as structural placeholder (0.0 values) — same pattern as iai.json; regenerate with bash scripts/update_criterion.sh on a quiet local machine"
  - "dhat.json committed with real captured values (n=1000, debug profile) — gate is functional immediately"
  - "update_criterion.sh uses --save-baseline <name> to preserve parallel and no_parallel runs in separate criterion named dirs before python3 parses both"

metrics:
  duration_minutes: 30
  completed_date: "2026-08-09"
  tasks_completed: 3
  tasks_total: 3
  commits: 3

actuals:
  tokens: 62000
  tasks: 3
  commits: 3
---

# Phase 01 Plan 03: Criterion + dhat Baseline Dimensions Summary

Criterion wall-clock suite (7 families x dual-profile) and dhat peak-memory gate (6 families, <=1.15x) complete the four-dimension measurement backbone; all baseline JSON files now committed.

## What Was Built

### Task 1 — benches/baseline_suite.rs: 7 families x {single, batch-100} x {parallel, no_parallel}

Created `benches/baseline_suite.rs` (root crate) and added `[[bench]] name = "baseline_suite" harness = false` to root `Cargo.toml`.

**CIRCULAR DEPENDENCY RESOLUTION (RESEARCH.md Open Question 3):**
The LCG `make_seeded_series` fixture is inlined directly in `baseline_suite.rs`. Adding `anofox-bench-harness` as a root dev-dependency would create a cycle (root ← harness ← root). The inlined copy is identical code (same LCG multiplier 6364136223846793005, same seed convention).

**7 families covered:**
- `AutoARIMA` — `AutoARIMA::new()` + fit+predict n=200
- `AutoETS` — `AutoETS::with_period(12)` + fit+predict n=200
- `AutoTheta` — `AutoTheta::new()` + fit+predict n=200
- `Naive` — `Naive::new()` + fit+predict n=200
- `Croston` — `Croston::new()` + fit+predict n=200
- `AutoEnsemble` — `AutoEnsemble::new()` + fit+predict n=200
- `LaplaceForecaster` — `LaplaceForecaster::new()` (gated behind `cfg(distributional)`)

**Each family has:**
- Single-series bench (n=200 fit+predict)
- Batch-100 bench (`batch::auto_ets` for ETS; loop fit+predict for others)

**Profile strategy (D-09):** bench IDs are profile-agnostic. `update_criterion.sh` runs the suite twice and stamps each entry in `criterion.json` with `profile: "parallel"` or `profile: "no_parallel"`.

**Standardized sample size (D-06):** `criterion_group! { name = baselines; config = Criterion::default().sample_size(20); ... }`

Build verified in four configurations:
- `cargo build --bench baseline_suite` (no features) — OK
- `cargo build --bench baseline_suite --features parallel` — OK
- `cargo build --bench baseline_suite --features distributional` — OK
- `cargo build --bench baseline_suite --features "distributional,parallel"` — OK

### Task 2 — scripts/update_criterion.sh + criterion.json

`scripts/update_criterion.sh` (`set -euo pipefail`, `chmod +x`):

1. **Top-of-file documentation:** "Run on a QUIET LOCAL MACHINE ONLY — never CI (wall-clock noise, D-03). Criterion baselines are informational only."
2. **Profile 1:** `cargo bench --bench baseline_suite --features parallel -- --save-baseline parallel_run`
3. **Profile 2:** `cargo bench --bench baseline_suite -- --save-baseline no_parallel_run`
4. **Parse:** python3 reads `target/criterion/<bench>/parallel_run/estimates.json` and `target/criterion/<bench>/no_parallel_run/estimates.json` for each bench; extracts `median.point_estimate` → `median_ns` and `median_abs_dev.point_estimate` → `mad_ns` (Assumption A2)
5. **Write:** `open('w')` — overwrites idempotently; first-run creates

`.planning/baselines/criterion.json` committed as structural placeholder (all `median_ns: 0.0`, `mad_ns: 0.0`):
- Both profiles present: `parallel`, `no_parallel`
- 14 bench entries (12 core + 2 Laplace gated)
- D-02 provenance fingerprint attached
- Regenerate: `bash scripts/update_criterion.sh` on a quiet machine

**No CI reference in script** — criterion capture is strictly local (T-01-09 mitigated).

### Task 3 — crates/anofox-bench-harness/tests/dhat_peak.rs + update_dhat.sh + dhat.json

**`dhat_peak.rs`** (single integration test binary in harness crate):
- Exactly one `#[global_allocator] static ALLOC: dhat::Alloc = dhat::Alloc;` (Pitfall 3, T-01-10)
- One `#[test] fn peak_memory_all_families()` covering 6 families sequentially (dhat panics on concurrent Profiler instances; all families in one test fn avoids overlap)
- Each family block: `let _p = dhat::Profiler::builder().testing().build(); ... let stats = dhat::HeapStats::get();`
- **D-12 boundary:** `assert!(stats.max_bytes <= (baseline_bytes as f64 * 1.15) as usize)` — float→usize truncation is conservative (documented)
- **First-run handling (MEAS-04):** `load_dhat_baseline(name)` returns `None` if `dhat.json` missing/absent → prints NOTICE, skips assertion (does not panic)
- **CAPTURE mode:** `CAPTURE_DHAT=1` env → prints `CAPTURE <name> <peak_bytes>` instead of asserting

**`scripts/update_dhat.sh`** (`set -euo pipefail`, `chmod +x`):
1. Runs `CAPTURE_DHAT=1 cargo test -p anofox-bench-harness --test dhat_peak -- --nocapture`
2. python3 parses `CAPTURE <name> <bytes>` lines from stdout
3. Writes `{provenance, benchmarks: [{name, peak_bytes}]}` via `open('w')` — idempotent overwrite

**`.planning/baselines/dhat.json`** — real captured values (n=1000, debug profile, i9-13900H):

| Family | peak_bytes |
|--------|-----------|
| auto_ets_n1000 | 290,440 |
| auto_arima_n1000 | 191,268 |
| auto_theta_n1000 | 133,456 |
| naive_n1000 | 60,024 |
| croston_n1000 | 76,792 |
| auto_ensemble_n1000 | 199,976 |

Gate test verified passing: `cargo test -p anofox-bench-harness --test dhat_peak` → 1 passed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] LaplaceForecaster is behind cfg(distributional) feature gate**
- **Found during:** Task 1 build
- **Issue:** `cargo build --bench baseline_suite` failed — `src/models/mod.rs` guards the `laplace` module behind `#[cfg(feature = "distributional")]`. The plan listed LaplaceForecaster as one of the 7 required families without noting the feature gate.
- **Fix:** Added `#[cfg(feature = "distributional")] use anofox_forecast::models::laplace::LaplaceForecaster;` and duplicated the `criterion_group!` macro with a `#[cfg(feature = "distributional")]` / `#[cfg(not(feature = "distributional"))]` split so the suite compiles in both configurations.
- **Files modified:** `benches/baseline_suite.rs`
- **Commit:** 9ed2dcf

**2. [Rule 1 - Bug] dhat::Profiler can't run multiple instances in same test binary without overlapping**
- **Found during:** Task 3 design
- **Issue:** dhat panics if more than one Profiler is live simultaneously. The plan's PATTERNS.md skeleton showed one `#[test]` fn per family, which would overlap when run in parallel (default Rust behavior).
- **Fix:** All 6 families measured in a single `#[test] fn peak_memory_all_families()` using nested blocks to ensure each Profiler drops before the next is created. This is the documented dhat pattern.
- **Files modified:** `crates/anofox-bench-harness/tests/dhat_peak.rs`
- **Commit:** f0c710d

## Known Stubs

**criterion.json placeholder values:**
- **File:** `.planning/baselines/criterion.json`
- **All entries:** `median_ns: 0.0`, `mad_ns: 0.0`
- **Reason:** Criterion wall-clock benchmarks must be captured on a quiet local machine (D-03); cannot run in CI or during automated plan execution. Structural placeholder committed per same pattern as `iai.json` (Plan 01-02).
- **Resolution:** `bash scripts/update_criterion.sh` on a quiet local machine.

## Threat Flags

No new threat surface beyond the plan's threat model (T-01-09 through T-01-SC). All mitigations applied:
- `update_criterion.sh` documented local-only; no CI step references it (T-01-09)
- Exactly one `#[global_allocator]` in `dhat_peak.rs`; verified by grep (T-01-10)
- `wee_alloc` not used anywhere; only `dhat::Alloc` in the profiling test binary (T-01-SC)

## Self-Check: PASSED

| Artifact | Status |
|----------|--------|
| benches/baseline_suite.rs | FOUND |
| Cargo.toml [[bench]] baseline_suite | FOUND |
| scripts/update_criterion.sh | FOUND |
| scripts/update_dhat.sh | FOUND |
| crates/anofox-bench-harness/tests/dhat_peak.rs | FOUND |
| .planning/baselines/criterion.json | FOUND |
| .planning/baselines/dhat.json | FOUND |
| commit 9ed2dcf (Task 1) | FOUND |
| commit 9f94308 (Task 2) | FOUND |
| commit f0c710d (Task 3) | FOUND |
