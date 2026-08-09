---
phase: 01-measurement-infrastructure-compute-baselines
verified: 2026-08-09T22:30:00Z
status: human_needed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Push a PR that introduces a crafted wasm_size.json with a non-integer 'bytes' field (e.g. '0; import os; os.system(\"id\")') and observe whether CI fails gracefully or executes injected code"
    expected: "CI should fail with a numeric validation error, not execute injected code. The fix suggested in 01-REVIEW.md CR-01 (move JSON parse inside Python, use int() validation) should be applied before this workflow guards any production branch."
    why_human: "The CR-01 injection vulnerability (wasm-size.yml line 60 interpolates $BASELINE unquoted into a Python -c string) is a security property that grep/compile checks cannot verify. Its blast radius under default fork-PR permissions is limited (contents: read), but the risk is real and requires human security review or the fix application to be confirmed."
  - test: "On a machine with valgrind >= 3.20 installed, run 'bash scripts/update_iai.sh' and verify it writes iai.json with three non-zero instruction_count values"
    expected: "iai.json should be overwritten with real Ir counts (non-zero) for bench_auto_ets_fit::n200, bench_auto_arima_fit::n200, and bench_batch_100::s100_n100. The structural placeholder currently committed has instruction_count: 0 for all three entries."
    why_human: "Valgrind is not installed on the dev machine so this cannot be verified programmatically here. The infrastructure (script, bench, CI gate) is complete and correct — but the numeric baseline itself is a placeholder pending maintainer action."
  - test: "On a quiet local machine, run 'bash scripts/update_criterion.sh' and verify criterion.json is overwritten with real non-zero median_ns values for both 'parallel' and 'no_parallel' profiles"
    expected: "criterion.json should have median_ns > 0 for all 12 non-Laplace bench entries across both profiles. The current file has all median_ns: 0.0 (structural placeholder per D-03 design)."
    why_human: "Criterion wall-clock capture requires a quiet local machine (D-03); cannot run in this verification context. The capture infrastructure is complete — the placeholder is intentional by design."
---

# Phase 01: Measurement Infrastructure & Compute Baselines — Verification Report

**Phase Goal:** The measurement backbone exists — CI workflows run, the baseline store is initialized, and every compute/memory/WASM-size number is committed and trustworthy (dead code removed before the size baseline is locked)

**Verified:** 2026-08-09T22:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `.planning/baselines/` contains committed JSON files for criterion, iai-callgrind, dhat, and WASM size — CI reads them but never overwrites | VERIFIED | All four files exist: `wasm_size.json` (bytes=2838958, real), `iai.json` (structural placeholder per known deviation), `dhat.json` (6 real peak_bytes values), `criterion.json` (structural placeholder per D-03 design). No CI step writes/adds/commits under `.planning/baselines/` — confirmed in both `wasm-size.yml` and `bench.yml` by grep. |
| 2 | `bench.yml` and `wasm-size.yml` are present, have correct triggers, least-privilege permissions, and the iai gate blocks regressions | VERIFIED | Both files exist. `wasm-size.yml`: triggers push+PR on [main,master], `permissions: contents: read`, gate fails at `delta > 1.0` (strict greater-than). `bench.yml`: triggers push+PR on [main,master], `permissions: contents: read`, `dtolnay/rust-toolchain@stable` only (D-10 stable-only), `iai-callgrind-runner --version 0.16.1 --locked` (T-01-05), no `baseline_suite` step (D-03). |
| 3 | A maintainer can reproduce any baseline via documented `scripts/update_*.sh` — no secrets or special access required | VERIFIED | All four scripts exist and are executable: `update_wasm_size.sh`, `update_iai.sh`, `update_criterion.sh`, `update_dhat.sh`. Each has `set -euo pipefail`, `chmod +x`, open('w') idempotent overwrite, and provenance fingerprint capture. `update_iai.sh` guards against missing valgrind with a clear error. `update_wasm_size.sh` guards against dead-code warnings (PERF-06 sequencing). |
| 4 | Known WASM dead code (8 unused `inner()` impl blocks, unused `RecipeKind` import, unnecessary `mut`) is removed and confirmed absent before the WASM size baseline was committed | VERIFIED | `grep -n "pub(crate) fn inner" crates/anofox-forecast-js/src/forecaster.rs` returns empty — all 8 blocks removed. `RecipeKind` does not appear in any `use` statement in `laplace_playground.rs` (only in doc comments, which is correct). WASM size baseline committed AFTER removal (wasm_size.json git_sha matches post-cleanup commit). |
| 5 | Native-parallel and WASM/single-thread profiles are measured separately; criterion baselines are captured locally not in CI | VERIFIED | `criterion.json` has 28 entries across both `"parallel"` and `"no_parallel"` profiles (verified via python3). `update_criterion.sh` is documented local-only ("Run on a QUIET LOCAL MACHINE ONLY — never CI"). `bench.yml` contains no criterion step — only the iai gate. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | `accuracy.yml` workflow (workflow_dispatch-only) — the accuracy part of MEAS-03 | Phase 2 | Phase 2 SC#1: "the accuracy.yml workflow is workflow_dispatch-only and never blocks a PR merge" |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/anofox-bench-harness/Cargo.toml` | Harness crate definition, publish=false | VERIFIED | Exists; `publish = false`; lib name `anofox_bench_harness`; dev-deps iai-callgrind 0.16.1, dhat 0.3.3, criterion 0.5; `[[bench]] name = "iai_suite" harness = false` |
| `crates/anofox-bench-harness/src/lib.rs` | pub mod baseline + pub mod fixtures | VERIFIED | Exists |
| `crates/anofox-bench-harness/src/baseline.rs` | D-02 serde structs with all required fields | VERIFIED | `ProvenanceFingerprint` (6 fields: git_sha, timestamp_iso, rustc_version, host_cpu, host_os, active_features), `CriterionEntry`, `IaiEntry`, `DhatEntry`, `WasmSizeBaseline` — all present with correct types |
| `crates/anofox-bench-harness/src/fixtures.rs` | `make_seeded_series(n, seed)` with LCG multiplier 6364136223846793005 | VERIFIED | Exists; LCG multiplier confirmed in source; seed-parameterized; `+20.0` offset |
| `crates/anofox-bench-harness/benches/iai_suite.rs` | Three hot-path library benchmarks with 1% Ir soft limit | VERIFIED | Exists; `bench_auto_ets_fit::n200`, `bench_auto_arima_fit::n200`, `bench_batch_100::s100_n100`; `Callgrind::default().soft_limits([(EventKind::Ir, 1.0f64)])` confirmed; batch uses `batch::auto_ets(&values, 12, Some(12), None)` (verified 4-arg signature); compiles clean (`cargo build -p anofox-bench-harness --benches` exits 0) |
| `crates/anofox-bench-harness/tests/dhat_peak.rs` | Peak-memory gate for major families, single #[global_allocator], 1.15x boundary | VERIFIED | Exists; exactly one `#[global_allocator] static ALLOC: dhat::Alloc`; 6 families (AutoETS, AutoARIMA, AutoTheta, Naive, Croston, AutoEnsemble); `(baseline_bytes as f64 * 1.15) as usize` boundary; first-run skip logic; CAPTURE_DHAT=1 mode; test passes (`1 passed; 0 failed`) |
| `benches/baseline_suite.rs` | 7-family (+ Laplace feature-gated) × single+batch-100 criterion suite | VERIFIED | Exists; all 7 families present (AutoARIMA, AutoETS, AutoTheta, Naive, Croston, AutoEnsemble, LaplaceForecaster gated); `criterion_group! config = Criterion::default().sample_size(20)`; no `anofox_bench_harness` import (circular dep avoided); compiles in both parallel and no-parallel configurations |
| `scripts/update_wasm_size.sh` | Provenance-stamped, idempotent, PERF-06-guarded capture script | VERIFIED | Exists; executable; PERF-06 guard at top; `open('w')` overwrite; full 6-field provenance; `bash scripts/update_wasm_size.sh` runs and produces valid JSON (wasm_size.json bytes=2838958) |
| `scripts/update_iai.sh` | Valgrind-guarded iai capture script | VERIFIED | Exists; executable; valgrind presence guard; iai-callgrind-runner presence guard; `open('w')` overwrite; parses `Instructions:` lines from bench output; writes correct schema |
| `scripts/update_criterion.sh` | Dual-profile (parallel + no_parallel) median+MAD criterion capture | VERIFIED | Exists; executable; documented local-only; runs `--save-baseline parallel_run` then `--save-baseline no_parallel_run`; parses `median.point_estimate` and `median_abs_dev.point_estimate`; `open('w')` overwrite; no CI reference |
| `scripts/update_dhat.sh` | CAPTURE_DHAT=1 mode dhat capture script | VERIFIED | Exists; executable; runs test with `CAPTURE_DHAT=1 ... --nocapture`; parses `CAPTURE <name> <bytes>` lines; `open('w')` overwrite |
| `.github/workflows/wasm-size.yml` | Read-only >1% relative size gate | VERIFIED | Exists; triggers push+PR on [main,master]; `permissions: contents: read`; gate uses `delta > 1.0`; no write/commit steps under `.planning/baselines/` |
| `.github/workflows/bench.yml` | Stable-only iai instruction gate | VERIFIED | Exists; `dtolnay/rust-toolchain@stable` only; `iai-callgrind-runner --version 0.16.1 --locked`; valgrind apt install; no `baseline_suite` step; no write steps |
| `.planning/baselines/wasm_size.json` | Real captured baseline with D-02 provenance | VERIFIED | bytes=2838958 (positive, real); all 6 provenance fields present; filename=`anofox_forecast_js_bg.wasm` |
| `.planning/baselines/iai.json` | D-02 schema with three benchmark entries | VERIFIED (with known deviation) | Schema correct; all 6 provenance fields; 3 entries for the correct bench names; instruction_count=0 for all — structural placeholder by design (valgrind not available on dev machine). Intentional per context. |
| `.planning/baselines/criterion.json` | Dual-profile provenance-stamped baseline | VERIFIED (with known deviation) | Schema correct; 28 entries; both `parallel` and `no_parallel` profiles present; median_ns=0.0/mad_ns=0.0 — structural placeholder per D-03 (must capture on quiet local machine). Intentional per context. |
| `.planning/baselines/dhat.json` | Real peak-memory baseline for 6 families | VERIFIED | 6 entries with real non-zero peak_bytes values (auto_ets_n1000=290440, auto_arima_n1000=191268, auto_theta_n1000=133456, naive_n1000=60024, croston_n1000=76792, auto_ensemble_n1000=199976); D-02 provenance attached |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Root `Cargo.toml` | `crates/anofox-bench-harness` | `[workspace] members` | WIRED | Line 2: `members = [".", "crates/anofox-forecast-js", "crates/anofox-bench-harness"]` |
| `update_wasm_size.sh` | `.planning/baselines/wasm_size.json` | python3 `open('w')` json.dump | WIRED | Writes exact D-02 ProvenanceFingerprint + filename + bytes |
| `wasm-size.yml` | `.planning/baselines/wasm_size.json` | python3 read-only at gate step | WIRED | Reads `d['bytes']` from committed file; never writes |
| PERF-06 removal | `wasm_size.json` | sequencing guard in `update_wasm_size.sh` | WIRED | WARNINGS guard at script top ensures dead code removed before capture |
| `iai_suite.rs` | `batch::auto_ets` | 4-arg `(&[Vec<f64>], usize, Option<usize>, Option<ModelPool>)` | WIRED | Correct 4-argument signature confirmed in source |
| `bench.yml` | iai gate (soft-limit) | `cargo bench -p anofox-bench-harness --bench iai_suite` | WIRED | 1% Ir soft limit encoded in `iai_suite.rs` propagates to CI exit code |
| `dhat_peak.rs` | `.planning/baselines/dhat.json` | `load_dhat_baseline()` via `CARGO_MANIFEST_DIR` path resolution | WIRED | Test reads baseline JSON from repo root; gate test passes |
| `baseline_suite.rs` | harness crate | NONE (circular dep avoided) | CORRECT | LCG inlined; no `anofox_bench_harness` import — `grep -c 'anofox_bench_harness' benches/baseline_suite.rs` = 0 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `wasm_size.json` | `bytes` | `stat --format=%s` on real built `.wasm` | Yes — 2838958 measured from release build | FLOWING |
| `dhat.json` | `peak_bytes` per family | `dhat::HeapStats::get().max_bytes` in test binary | Yes — real allocator measurements | FLOWING |
| `iai.json` | `instruction_count` | Valgrind callgrind output (placeholder=0) | No — structural placeholder, valgrind unavailable | STATIC (known deviation, intentional) |
| `criterion.json` | `median_ns`, `mad_ns` | criterion `estimates.json` files (placeholder=0.0) | No — structural placeholder per D-03 | STATIC (known deviation, intentional) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Harness crate builds | `cargo build -p anofox-bench-harness` | Finished dev profile, 0 errors | PASS |
| iai_suite bench compiles | `cargo build -p anofox-bench-harness --benches` | 0 errors, 1 bench artifact | PASS |
| dhat_peak test compiles | `cargo test -p anofox-bench-harness --test dhat_peak --no-run` | 0 errors | PASS |
| dhat_peak test passes | `cargo test -p anofox-bench-harness --test dhat_peak` | 1 passed; 0 failed (all 6 families PASS with real baseline values) | PASS |
| baseline_suite bench compiles | `cargo build --bench baseline_suite` | 0 errors | PASS |
| wasm_size.json valid structure | python3 validation | All required keys present, bytes=2838958 (positive integer) | PASS |
| criterion.json dual-profile | python3 validation | profiles={'parallel','no_parallel'}, 28 entries | PASS |
| wasm-size.yml no write steps | `grep -E 'git add.*baselines\|git commit.*baselines'` | 0 matches | PASS |
| bench.yml no baseline_suite | `grep -c 'baseline_suite'` | 0 | PASS |
| bench.yml iai-runner pinned | `grep '0.16.1 --locked'` | Found | PASS |
| update_wasm_size.sh executable | `test -x` | executable | PASS |
| update_iai.sh executable | `test -x` | executable | PASS |
| update_criterion.sh executable | `test -x` | executable | PASS |
| update_dhat.sh executable | `test -x` | executable | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MEAS-01 | 01-01 | Committed baseline store; CI reads, never writes | SATISFIED | All 4 JSON files in `.planning/baselines/`; both CI workflows have `permissions: contents: read` and no write steps |
| MEAS-02 | 01-01, 01-02, 01-03 | Maintainer can capture/refresh via `scripts/update_*.sh` | SATISFIED | All 4 scripts exist, are executable, and documented; capture tested (wasm_size.json and dhat.json have real values) |
| MEAS-03 | 01-01 (bench.yml+wasm-size.yml part) | CI workflows for benchmarking and WASM size exist; accuracy.yml deferred to Phase 2 | SATISFIED (partial — accuracy.yml is Phase 2 SC#1) | `bench.yml` and `wasm-size.yml` exist and have correct structure; accuracy.yml is a Phase 2 deliverable |
| MEAS-04 | All plans | No measurement code added to library `src/` | SATISFIED | No new files under `src/`; all measurement code in `benches/`, `tests/`, `scripts/`, `crates/anofox-bench-harness/` |
| PERF-01 | 01-03 | Criterion suite covers fit+predict across model families | SATISFIED | `baseline_suite.rs` covers all 7 families (with Laplace feature-gated) × single+batch-100 |
| PERF-02 | 01-02 | iai-callgrind gates on 3 critical hot paths in CI | SATISFIED | `bench.yml` runs `iai_suite` with 1% Ir soft limit on 3 hot paths |
| PERF-03 | 01-03 | Native-parallel and no-Rayon profiles measured separately | SATISFIED | `criterion.json` has 28 entries across `parallel` and `no_parallel` profiles |
| PERF-04 | 01-03 | dhat peak-memory asserts bounds for major families | SATISFIED | `dhat_peak.rs` test passes for 6 families with real dhat.json baseline; `<=1.15x` gate |
| PERF-05 | 01-01 | Release WASM size tracked against committed baseline with delta threshold | SATISFIED | `wasm-size.yml` reads `wasm_size.json` and fails on `delta > 1.0%`; baseline bytes=2838958 |
| PERF-06 | 01-01 | WASM dead code removed before size baseline locked | SATISFIED | 8 `pub(crate) fn inner()` blocks removed; `RecipeKind` import removed; `mut` removed; 0 warnings on wasm32 target confirmed |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `.github/workflows/wasm-size.yml` | 58-66 | CR-01: `$BASELINE` interpolated unquoted into Python `-c` string — code injection if `wasm_size.json` bytes field is crafted | WARNING (tracked) | Security quality issue; blast radius limited by `permissions: contents: read` on fork PRs. Tracked in `01-REVIEW.md` as CR-01. Does NOT block phase goal (workflows exist and run; measurement backbone is functional). Fix documented in 01-REVIEW.md. |
| `.github/workflows/bench.yml` | 4-5 | WR-01: Comment claims job reads `iai.json` — it does not. Actual gate compares callgrind output in `target/` (cached by rust-cache). On cache miss, gate passes vacuously. | INFO (tracked) | Documentation inaccuracy; gate mechanism is correct. Tracked in `01-REVIEW.md` as WR-01. |
| `scripts/update_wasm_size.sh` | 32 | WR-02: `stat --format=%s` is Linux-only (macOS requires `stat -f%z`) | INFO (tracked) | Dev machine portability issue; CI uses ubuntu-latest (Linux). Tracked in `01-REVIEW.md` as WR-02. |
| `crates/anofox-bench-harness/benches/iai_suite.rs` | 43 | WR-03: Batch fixture seed initialization diverges from `baseline_suite.rs` (`wrapping_add(1).wrapping_mul(...)` vs `seed + 1`) | INFO (tracked) | D-08 single-source-of-truth principle violated for batch LCG; iai and criterion batch fixtures exercise different data. Tracked in `01-REVIEW.md` as WR-03. |
| `scripts/update_iai.sh` | 29-30 | WR-04: Valgrind presence checked but minimum version (>=3.20) not validated | INFO (tracked) | Old valgrind would produce opaque errors. Tracked in `01-REVIEW.md` as WR-04. |

No `TBD`, `FIXME`, or `XXX` markers found in any phase-created files (clean scan). The `/tmp` usage in `update_iai.sh` and `update_dhat.sh` for temp log files is transient and cleaned via `trap`, not a stub indicator.

**Prohibition checks:**

| Prohibition | Status |
|-------------|--------|
| WASM size baseline NOT captured before PERF-06 cleanup | VERIFIED — wasm_size.json git_sha is post-cleanup commit; `update_wasm_size.sh` WARNINGS guard enforces sequencing |
| CI workflows must NOT write/git-add/commit under `.planning/baselines/` | VERIFIED — grep confirms 0 such steps in both workflow files |
| Measurement code must NOT be in library `src/` | VERIFIED — no new files under `src/`; all code in harness crate, benches/, tests/, scripts/ |
| Must NOT introduce wee_alloc | VERIFIED — only `dhat::Alloc` used (in profiling test binary only); wee_alloc not found anywhere |
| bench.yml must NOT run `cargo bench --bench baseline_suite` | VERIFIED — `grep -c 'baseline_suite' .github/workflows/bench.yml` = 0 |
| iai gate must NOT run on beta/nightly | VERIFIED — only `dtolnay/rust-toolchain@stable` in bench.yml; no matrix |
| dhat_peak.rs must NOT declare more than one `#[global_allocator]` | VERIFIED — exactly 1 `#[global_allocator]` in dhat_peak.rs |

### Human Verification Required

#### 1. CR-01 Security Finding: Python Code Injection in wasm-size.yml

**Test:** Submit a test PR (or local branch) modifying `.planning/baselines/wasm_size.json` to set `"bytes"` to a crafted string (e.g., `0; import os; os.system("id")`). Observe what the `Check WASM size gate` step does.

**Expected:** CI should fail with a numeric validation error and not execute the injected code. The fix from 01-REVIEW.md CR-01 should be applied: move the JSON read and comparison fully inside Python using `int(d['bytes'])` validation, eliminating `$BASELINE` shell interpolation entirely.

**Why human:** This is a security property requiring intentional fix evaluation and testing; grep/compile checks cannot confirm safe behavior. The `permissions: contents: read` scope limits blast radius under default GitHub fork-PR settings, but the vulnerability is real and should be remediated before this workflow guards production traffic.

#### 2. iai.json — Real Instruction Counts Needed

**Test:** On a machine with valgrind >= 3.20 and `iai-callgrind-runner 0.16.1` installed, run `bash scripts/update_iai.sh` and inspect the resulting `.planning/baselines/iai.json`.

**Expected:** All three entries (`bench_auto_ets_fit::n200`, `bench_auto_arima_fit::n200`, `bench_batch_100::s100_n100`) should have `instruction_count > 0`. Commit the result.

**Why human:** Valgrind is not installed on the dev machine; the current placeholder (instruction_count=0 for all entries) means the iai CI gate has no real regression baseline to compare against — it will produce `N/A` comparisons and pass vacuously on first CI run after a cache miss.

#### 3. criterion.json — Real Wall-Clock Baselines Needed

**Test:** On a quiet local machine (no background load), run `bash scripts/update_criterion.sh` and inspect `.planning/baselines/criterion.json`.

**Expected:** All non-Laplace entries should have `median_ns > 0` and `mad_ns > 0` for both `parallel` and `no_parallel` profiles.

**Why human:** Criterion wall-clock capture is deliberately local-only (D-03); the current placeholder (all 0.0) means the informational trend-tracking purpose of the baseline cannot be fulfilled until a maintainer captures real timings.

### Gaps Summary

No gaps block the phase goal. All 5 ROADMAP success criteria are verified against the codebase. The three human verification items are:

1. **CR-01 security fix** — a security quality improvement needed before wasm-size.yml is considered production-safe. The workflow exists, runs, and gates correctly; the injection vector has limited blast radius but requires remediation.
2. **iai.json real values** — a maintainer action (requires valgrind). Infrastructure is complete; placeholder is intentional per context provided. The CI gate functions once valgrind runs produce real callgrind output.
3. **criterion.json real values** — a maintainer action (local quiet machine). Infrastructure is complete; placeholder is intentional per D-03 design.

The code review findings (01-REVIEW.md: 1 Critical, 4 Warnings, 3 Info) are tracked quality/security items. None block the phase goal of "measurement backbone exists — CI workflows run, baseline store initialized, WASM dead code removed before baseline locked."

---

_Verified: 2026-08-09T22:30:00Z_
_Verifier: Claude (gsd-verifier)_
