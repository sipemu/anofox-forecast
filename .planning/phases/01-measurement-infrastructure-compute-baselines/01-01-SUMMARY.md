---
phase: "01"
plan: "01"
subsystem: measurement-infrastructure
status: complete
tags: [measurement, wasm, harness, dead-code, ci-gate, baseline-schema]
completed_date: "2026-08-09"

dependency_graph:
  requires: []
  provides:
    - anofox-bench-harness crate (D-04 workspace member)
    - D-02 ProvenanceFingerprint schema (baseline.rs)
    - seeded fixture generator make_seeded_series (D-08)
    - PERF-06 dead-code removal (zero WASM warnings)
    - .planning/baselines/wasm_size.json (post-cleanup baseline)
    - scripts/update_wasm_size.sh (D-05 maintainer entrypoint)
    - .github/workflows/wasm-size.yml (D-11 CI gate)
  affects:
    - Plans 01-02 and 01-03 (iai, criterion, dhat dimensions replicate this schema+pattern)
    - crates/anofox-forecast-js (PERF-06 removals)
    - Root Cargo.toml workspace members

tech_stack:
  added:
    - anofox-bench-harness 0.1.0 (publish=false harness crate)
    - iai-callgrind 0.16.1 (dev-dep, declared for Plans 02-03)
    - dhat 0.3.3 (dev-dep, declared for Plans 02-03)
    - serde + serde_json + chrono (harness crate deps)
  patterns:
    - D-02 ProvenanceFingerprint: 6-field JSON schema attached to every committed baseline
    - D-05 update_*.sh: bash entrypoint with PERF-06 guard + python3 JSON write (open w, idempotent)
    - D-11 CI gate: python3 inline size comparison, fail on delta > 1.0%, read-only permissions

key_files:
  created:
    - crates/anofox-bench-harness/Cargo.toml
    - crates/anofox-bench-harness/src/lib.rs
    - crates/anofox-bench-harness/src/baseline.rs
    - crates/anofox-bench-harness/src/fixtures.rs
    - scripts/update_wasm_size.sh
    - .github/workflows/wasm-size.yml
    - .planning/baselines/wasm_size.json
  modified:
    - Cargo.toml (workspace members: added crates/anofox-bench-harness)
    - crates/anofox-forecast-js/src/forecaster.rs (PERF-06: -8 inner() blocks, -1 unnecessary mut)
    - crates/anofox-forecast-js/src/laplace_playground.rs (PERF-06: -RecipeKind import)

decisions:
  - "No [[bench]] entries in harness Cargo.toml for Plans 02-03 targets yet — avoid compile errors until bench source files exist"
  - "wasm-size.yml uses quoted 'on' key to avoid YAML boolean coercion of bare 'on'"
  - "RecipeKind doc-comment references in laplace_playground.rs retained — they describe the type, not import it; zero compiler warnings confirmed"
  - "wasm-pack version 0.14.0 (local) vs 0.15.0 (latest) — update not in scope of this plan; no impact on baseline correctness"

metrics:
  duration_minutes: 9
  completed_date: "2026-08-09"
  tasks_completed: 4
  tasks_total: 4
  commits: 4

actuals:
  tokens: 18000
  tasks: 4
  commits: 4
---

# Phase 01 Plan 01: WASM-Size Measurement Tracer Summary

WASM-size baseline backbone end-to-end: harness crate with D-02 serde schema, PERF-06 dead-code removal (zero WASM warnings), provenance-stamped idempotent capture script, and read-only >1% CI gate.

## What Was Built

### Task 1 — Harness crate skeleton + D-02 baseline schema + seeded fixtures

Created `crates/anofox-bench-harness` as a new workspace member (`publish = false`, edition 2021).

- `src/baseline.rs`: Five `#[derive(Serialize, Deserialize, Debug, Clone)]` structs implementing the D-02 schema:
  - `ProvenanceFingerprint` — 6 fields: `git_sha`, `timestamp_iso`, `rustc_version`, `host_cpu`, `host_os`, `active_features: Vec<String>`
  - `CriterionEntry` — `name`, `profile`, `median_ns: f64`, `mad_ns: f64`
  - `IaiEntry` — `name`, `instruction_count: u64`
  - `DhatEntry` — `name`, `peak_bytes: u64`
  - `WasmSizeBaseline` — `provenance: ProvenanceFingerprint`, `filename: String`, `bytes: u64`

- `src/fixtures.rs`: `pub fn make_seeded_series(n: usize, seed: u64) -> TimeSeries` using the same LCG as `benches/ets_benchmark.rs` (multiplier 6364136223846793005, `>> 33` shift), seed-parameterized per D-08.

- `src/lib.rs`: `pub mod baseline; pub mod fixtures;`

- `dev-dependencies` declared now for Plans 02-03: `iai-callgrind = "0.16.1"`, `dhat = "0.3.3"`, `criterion = "0.5"`.

### Task 2 — PERF-06 dead-code removal (zero WASM warnings)

Removed all 10 confirmed dead-code items from `crates/anofox-forecast-js/`:

- 8 unused `pub(crate) fn inner()` impl blocks in `forecaster.rs`:
  `SESForecaster`, `HoltForecaster`, `HoltWintersForecaster`, `CrostonForecaster`, `TSBForecaster`, `ADIDAForecaster`, `IMAPAForecaster`, `GARCHForecaster`
- `RecipeKind` from the `use` statement in `laplace_playground.rs`
- `mut` from the `KalmanFilter` binding in `log_likelihood()` in `forecaster.rs`

Verified: `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown` → 0 warnings.
Verified: `cargo build -p anofox-forecast-js` (native) → clean build, no breakage.

USED `inner()` methods in `time_series.rs:144`, `calendar.rs:212`, `postprocess.rs:67` were NOT touched.

### Task 3 — scripts/update_wasm_size.sh + captured baseline

`scripts/update_wasm_size.sh` (`set -euo pipefail`, `chmod +x`) steps:

1. PERF-06 guard: `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown | grep -c 'warning:'` — exits 1 if non-zero (sequencing enforced)
2. `wasm-pack build --release` + `git checkout js/package.json js/README.md`
3. `stat --format=%s js/anofox_forecast_js_bg.wasm`
4. Gathers provenance: `git rev-parse HEAD`, ISO-8601 UTC timestamp, `rustc --version`, `/proc/cpuinfo`, `uname -sr`
5. `python3` writes JSON via `open('w')` — OVERWRITES (never appends, idempotent)

Baseline captured: `.planning/baselines/wasm_size.json`
- `bytes`: 2838958
- `rustc_version`: rustc 1.97.0 (2d8144b78 2026-07-07)
- `host_cpu`: 13th Gen Intel(R) Core(TM) i9-13900H
- `active_features`: []

Idempotency verified: two consecutive runs produced byte-identical `filename` and `bytes`.

### Task 4 — .github/workflows/wasm-size.yml

- Triggers: `push` and `pull_request` on `[main, master]`
- `permissions: contents: read` (least-privilege, T-01-02, T-01-03)
- Steps: checkout, `dtolnay/rust-toolchain@stable` + `wasm32-unknown-unknown` target, `Swatinem/rust-cache@v2`, wasm-pack install (HTTPS curl), release build, restore `js/package.json`
- Size gate: reads baseline via python3, computes `(current - baseline) / baseline * 100`, fails only when `delta > 1.0` (strict greater-than per D-11)
- ZERO steps that write, `git add`, or commit under `.planning/baselines/` (MEAS-01)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] wasm-size.yml YAML `on:` boolean coercion**
- **Found during:** Task 4 verification
- **Issue:** PyYAML interprets bare `on:` as boolean `True`, making the trigger block inaccessible
- **Fix:** Quoted the key as `"on":` — standard YAML workaround for this reserved word; other workflows in this repo use bare `on:` which works in GitHub Actions' own parser but fails PyYAML-based verification
- **Files modified:** `.github/workflows/wasm-size.yml`
- **Commit:** 6cb455e

**2. [Rule 3 - Blocking] `[[bench]]` entries omitted from harness Cargo.toml**
- **Found during:** Task 1 design
- **Issue:** Plan notes "Do NOT register any `[[bench]]` yet — later plans add those alongside their bench files so the crate keeps compiling"; PATTERNS.md skeleton included a `[[bench]]` entry that would cause a compile error since `iai_suite.rs` doesn't exist yet
- **Fix:** Omitted `[[bench]]` entirely from harness Cargo.toml per plan instruction; Plans 02-03 will add them with their bench source files
- **Impact:** None — harness crate builds cleanly; dev-deps are declared and ready

## Known Stubs

None — all artifacts are fully functional. The harness crate compiles; the script captures real production data; the CI gate reads real committed data; dead code removed with zero remaining warnings.

## Threat Flags

No new threat surface introduced beyond what was already in the plan's threat model (T-01-01 through T-01-04, T-01-SC). All mitigations applied:
- `permissions: contents: read` on wasm-size job (T-01-02)
- No GITHUB_TOKEN secrets consumed (T-01-03)
- wasm-pack curl installer documented inline (T-01-04)
- iai-callgrind 0.16.1 / dhat 0.3.3 pre-verified in RESEARCH Package Legitimacy Audit (T-01-SC)

## Self-Check: PASSED

| Artifact | Status |
|----------|--------|
| crates/anofox-bench-harness/Cargo.toml | FOUND |
| crates/anofox-bench-harness/src/lib.rs | FOUND |
| crates/anofox-bench-harness/src/baseline.rs | FOUND |
| crates/anofox-bench-harness/src/fixtures.rs | FOUND |
| scripts/update_wasm_size.sh | FOUND |
| .github/workflows/wasm-size.yml | FOUND |
| .planning/baselines/wasm_size.json | FOUND |
| commit bb6f528 (Task 1) | FOUND |
| commit e0f1617 (Task 2) | FOUND |
| commit ca1e188 (Task 3) | FOUND |
| commit 6cb455e (Task 4) | FOUND |
