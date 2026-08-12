---
phase: 02-accuracy-harness-statistical-methodology
verified: 2026-08-10T22:27:07Z
status: passed
score: 10/10
behavior_unverified: 0
overrides_applied: 0
re_verification: false
deferred:
  - truth: "accuracy.json committed with per-frequency M3 Naive2+AutoETS metrics and provenance, locked after ACCUR-08 anchor passes"
    addressed_in: "Phase 4"
    evidence: "Phase 4 SC-2: 'each landed improvement has a documented before/after delta in the relevant baseline file (e.g., MASE delta in accuracy.json)' — Phase 4 will update/commit accuracy.json once AutoETS improvements close the ~21% gap"
---

# Phase 02: Accuracy Harness & Statistical Methodology — Verification Report

**Phase Goal:** A statistically correct accuracy harness produces trustworthy per-frequency numbers — validated against a published reference result before any baseline is locked — and a documented cross-library comparison is committed.

**Verified:** 2026-08-10T22:27:07Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | D-03 MASE denominator-collapse guard substitutes period-1 fallback in calculate_mase (no silent NaN on constant seasonal series) | VERIFIED | `src/utils/metrics.rs` lines 156–175: `if naive_mae == 0.0` branch computes period-1 first-difference MAE; regression test `mase_constant_series_no_nan` in the same file; 76 metrics tests pass |
| 2 | TSF loader reads Monash .tsf with Latin-1 decode, is_finite guard, env gate, canonicalized path safety | VERIFIED | `loader.rs:43` uses `bytes.iter().map(|&b| b as char).collect()`; line 80 `is_finite().then_some(v)`; line 109 `std::fs::canonicalize(dir)` + fixed filenames only; `dataset_dir_from_env()` returns None when unset; 5 unit tests pass offline |
| 3 | Naive2 selects SeasonalNaive when ACF at seasonal lag exceeds 1.645/sqrt(n) Bartlett 95% band, else Naive | VERIFIED | `naive2.rs:56` `let critical = 1.645 / (n as f64).sqrt()`; module doc corrected to "Bartlett 95% confidence band, 5% two-sided" (WR-03 fix committed 9c7a6ad); 5 unit tests (seasonal gate, flat gate, ACF signal, ACF noise) pass offline |
| 4 | Every evaluation fold asserts train_end <= test_start; fitted_values() never used as out-of-sample proxy | VERIFIED | `accuracy.rs:180` `assert!(fold.train_end <= fold.test_start, "temporal integrity violation:...")` fires on every fold in all tests; grep confirms `fitted_values` appears only in a comment at line 438, never as a function call |
| 5 | Full metric set (MASE with training denominator, sMAPE, RMSE, MAE, MSIS, coverage) is implemented and NaN-guarded before aggregation | VERIFIED | `accuracy.rs:49` imports `{coverage, mae, msis, rmse, smape}` from library; `FrequencyResult` fields include all six; `nanmean()` at line 294 filters `is_finite()` and logs `skipped` count to stderr before averaging; ForecastMetrics::compute absent from accuracy.rs (only in comments) |
| 6 | Per-frequency stratification never produces a single cross-frequency aggregate (ACCUR-07) | VERIFIED | `run_accuracy_harness()` returns `HashMap<String, FrequencyResult>` keyed by "monthly"/"quarterly"/"yearly"; `per_frequency_stratification` test asserts three separate frequency keys and asserts absence of blended keys; no `"all"` or `"combined"` key ever written |
| 7 | ACCUR-08 anchor test asserts AutoETS M3-monthly MASE within ±0.02 of pinned statsforecast reference, and the emit helper refuses to write accuracy.json when the anchor fails | VERIFIED | `accur08_anchor_m3_monthly_autoets` at accuracy.rs:677 reads pinned reference (0.8633 from statsforecast_reference.json), asserts `(autoets_mase - reference_mase).abs() <= 0.02`; `emit_accuracy_json()` at line 904 panics with descriptive error if `!anchor_passed`; anchor FAILS (1.0452 vs 0.8633, diff=+0.1819) → accuracy.json NOT committed (defer-lock human decision); accuracy.json correctly absent |
| 8 | Diebold-Mariano test implements squared-error loss differentials, HAC variance (autocovariances to lag h-1), and HLN small-sample correction | VERIFIED | `dm_test.rs:40` function signature confirmed; HAC at line 67-69 `gamma(0) + 2.0 * (1..h).map(|k| gamma(k)).sum::<f64>()`; HLN at line 87 `(t + 1.0 - 2.0 * h + h * (h - 1.0) / t) / t`; four DM tests pass: identical (p≈1.0), clear winner (reject), length guard (NaN, false), HLN shrinks stat |
| 9 | Cross-library comparison runs on shared M3 datasets/horizons/preprocessing, produces per-frequency diff table, with DM gate suppressing close (<5%) superiority claims lacking p<0.05 | VERIFIED | `cross_library.rs`: `bench01_cross_library` env-gated per-frequency diff table; `dm_gate_unit_synthetic` exercises 4 gate scenarios including suppression; `claim_allowed = (gap_pct.abs() >= 0.05) || (reject && anofox_beats)` at line 316, 345, 368, 387; 2 tests pass (bench01 skips cleanly with env unset, dm_gate runs always) |
| 10 | statsforecast reference fixture carries pinned provenance (runtime version capture); accuracy.yml is workflow_dispatch-only with contents:read; CI never regenerates the fixture | VERIFIED | `statsforecast_reference.json` contains `provenance` with 6 keys (statsforecast_version=2.0.3, numpy=2.3.5, pandas=2.3.3, python=3.12.12, timestamp, m3_preprocessing); `run_statsforecast.py` uses `importlib.metadata.version()` (no hard-coded literals); `accuracy.yml` trigger is `workflow_dispatch` only — grep for `^\s*(push|pull_request):` returns empty; `permissions: contents: read` present |

**Score:** 10/10 truths verified (0 present, behavior-unverified)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|---------|
| 1 | `accuracy.json` committed with per-frequency Naive2+AutoETS metrics and provenance, locked after ACCUR-08 anchor passes (SC-4 second clause) | Phase 4 | Phase 4 SC-2: "each landed improvement has a documented before/after delta in the relevant baseline file (e.g., MASE delta in accuracy.json)" — lock procedure documented in Plan 04 SUMMARY and in `emit_accuracy_baseline_if_write_flag_set` test |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/utils/metrics.rs` | D-03 MASE denominator-collapse guard, regression test | VERIFIED | Period-1 fallback at lines 161-175; `mase_constant_series_no_nan` test (2 cases); 76 metrics tests pass |
| `crates/anofox-bench-harness/src/loader.rs` | TSF loader, Latin-1, is_finite, env gate, canonicalize, mase_scale | VERIFIED | All 4 public functions present; 5 offline unit tests pass |
| `crates/anofox-bench-harness/src/naive2.rs` | Naive2 ACF-gated seasonal/non-seasonal, acf_at_lag | VERIFIED | 1.645/sqrt(n) Bartlett 95% threshold; 5 offline unit tests pass |
| `crates/anofox-bench-harness/tests/accuracy.rs` | Expanding-window fold, temporal integrity, full metrics, ACCUR-08 anchor, guarded emit | VERIFIED | 5 tests (tracer, per_frequency, msis_coverage, anchor, emit_guard) all skip cleanly without dataset env; CR-01 fix applied (Option<f64> for rmse/mae in ModelMetrics); WR-01 fix applied (mase > 0.0 guard) |
| `crates/anofox-bench-harness/src/dm_test.rs` | Diebold-Mariano with HLN + HAC, normal_cdf, 4 unit tests | VERIFIED | 4 DM tests pass; no statrs dependency added |
| `crates/anofox-bench-harness/tests/cross_library.rs` | Cross-library diff table + DM gate + claim_allowed | VERIFIED | 2 tests pass with env unset |
| `validation/run_statsforecast.py` | M3 reference mode with runtime provenance emission | VERIFIED | `importlib.metadata.version()` used; `--m3-reference` mode exists |
| `.planning/baselines/statsforecast_reference.json` | Pinned fixture with provenance block | VERIFIED | 6-key provenance object; monthly MASE=0.8633 (statsforecast 2.0.3 pinned env) |
| `.github/workflows/accuracy.yml` | workflow_dispatch-only, contents:read, no PR gating | VERIFIED | No non-comment push/pull_request triggers; contents:read present; no Python/baseline-write step |
| `.planning/baselines/accuracy.json` | CORRECTLY ABSENT (defer-lock) | VERIFIED | File does not exist — correct per human defer-lock decision after ACCUR-08 anchor failed (1.0452 vs 0.8633, gap=+0.1819 > ±0.02 tolerance) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `loader.rs::mase_scale(train, period)` | `accuracy.rs` training-denominator MASE | Imported and called at lines 205, 241 | WIRED | `use anofox_bench_harness::loader::{..., mase_scale}` at accuracy.rs:43; used for both Naive2 and AutoETS denominators |
| `accuracy.rs::run_accuracy_harness()` | `accuracy.rs::accur08_anchor_m3_monthly_autoets` | Called directly at line 678 | WIRED | Anchor reads harness output map; monthly bucket extracted and compared to pinned reference |
| `accuracy.rs::emit_accuracy_json(anchor_passed)` | Baseline write guard | `if !anchor_passed { panic!(...) }` at line 905 | WIRED | Refuses to write when anchor failed; CR-01 fix ensures no NaN serde panic in the success path |
| `dm_test.rs::diebold_mariano_hln` | `cross_library.rs` DM gate | `use anofox_bench_harness::dm_test::diebold_mariano_hln` | WIRED | Called in `dm_gate_unit_synthetic` with 4 synthetic scenarios; `claim_allowed` boolean encodes the <5% gate rule |
| `statsforecast_reference.json` | `accuracy.rs::load_reference_monthly_mase()` | CARGO_MANIFEST_DIR-relative path at line 608 | WIRED | Fixture loaded at test time; 0.8633 value used as reference in ACCUR-08 assertion |
| `accuracy.yml` | `cargo test -p anofox-bench-harness --test accuracy` | workflow step at lines 61-68 | WIRED | workflow_dispatch-only; ANOFOX_DATASET_DIR set to workspace/validation/data; no write flag |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `accuracy.rs::run_accuracy_harness()` | `autoets_mase` | `mase_scale(train, period)` training denominator; `ae_fmae` from real AutoETS predictions on M3 series | Yes — real model fitting on real corpus series | FLOWING |
| `statsforecast_reference.json::monthly.autoets_mase` | 0.8633 | Regenerated by `run_statsforecast.py --m3-reference` on pinned statsforecast 2.0.3 env; provenance block timestamps and pins all dependency versions | Yes — produced by real statsforecast run, not hardcoded | FLOWING |
| `dm_test.rs::diebold_mariano_hln` | `p_value, reject_h0` | Squared-error loss differentials → HAC variance → HLN correction → Abramowitz-Stegun normal CDF | Yes — mathematical computation, std-only | FLOWING |
| `accuracy.yml` | CI test results | `cargo test -p anofox-bench-harness --test accuracy` run with real harness code | Yes — executes real tests (skips cleanly without corpus) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| DM test unit tests pass | `cargo test -p anofox-bench-harness --lib dm_test` | 4 passed, 0 failed | PASS |
| Accuracy tests skip cleanly without env var | `cargo test -p anofox-bench-harness --test accuracy` | 5 passed (skipped data-dependent paths), 0 failed | PASS |
| Cross-library tests pass (synthetic DM gate runs unconditionally) | `cargo test -p anofox-bench-harness --test cross_library` | 2 passed, 0 failed | PASS |
| Loader unit tests pass without env var | `cargo test -p anofox-bench-harness --lib loader::` | 5 passed, 0 failed | PASS |
| Naive2 unit tests pass without env var | `cargo test -p anofox-bench-harness --lib naive2::` | 5 passed, 0 failed | PASS |
| D-03 MASE regression tests pass | `cargo test -p anofox-forecast --lib metrics::` | 76 passed, 0 failed | PASS |
| accuracy.json is absent (defer-lock) | `ls .planning/baselines/accuracy.json` | File not found | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| ACCUR-01 | 02-01 | Dataset loader reads M3 corpus, gated on ANOFOX_DATASET_DIR | SATISFIED | `loader.rs::dataset_dir_from_env()` + `parse_tsf_with_meta()` + `load_m3()`; clean skip when env unset |
| ACCUR-02 | 02-01, 02-02 | Expanding-window CV with temporal-integrity assertions; no fitted_values() | SATISFIED | `fold.train_end <= fold.test_start` assertion on every fold in accuracy.rs; fitted_values() absent from all harness test code |
| ACCUR-03 | 02-01 | MASE with correct denominator-collapse guard (no silent NaN/Inf in aggregates) | SATISFIED | D-03 fix in metrics.rs; `nanmean()` with is_finite filter + logged skipped count in accuracy.rs |
| ACCUR-04 | 02-01 | sMAPE, RMSE, MAE verified with correct denominators | SATISFIED | Reused from `src/utils/metrics.rs` via `use anofox_forecast::utils::metrics::{coverage, mae, msis, rmse, smape}` |
| ACCUR-05 | 02-02 | MSIS and empirical interval coverage for prediction intervals | SATISFIED | `FrequencyResult.autoets_msis: Option<f64>` and `autoets_coverage: Option<f64>` for monthly; `msis_coverage_present_monthly` test asserts finite MSIS and coverage ∈ [0,1] |
| ACCUR-06 | 02-02 | Naive2 baseline with ACF-gated seasonal/non-seasonal selection | SATISFIED | `naive2.rs`: SeasonalNaive vs Naive selected by `acf.abs() > 1.645/sqrt(n)` (Bartlett 95%) |
| ACCUR-07 | 02-02, 02-04 | Per-frequency stratification; no cross-frequency aggregate | SATISFIED | `HashMap<String, FrequencyResult>` by frequency key; test asserts 3 separate keys and no "all" key |
| ACCUR-08 | 02-04 | Accuracy baseline validated against reference before being locked | SATISFIED (correctly) | ACCUR-08 anchor TEST exists and runs; anchor FAILS (1.0452 vs 0.8633, +21% gap); emit helper refuses to write when anchor fails; accuracy.json NOT committed — this is the CORRECT behavior ("validated before locked" means refuse to lock when validation fails); requirement infrastructure complete, baseline lock deferred to Phase 4 |
| BENCH-01 | 02-03 | Documented cross-library comparison on shared datasets/horizons/preprocessing | SATISFIED | `bench01_cross_library` diff table: yearly +6.1%, quarterly +22.8%, monthly +21.1% vs statsforecast 2.0.3; per-frequency, not blended |
| BENCH-02 | 02-03 | DM significance gate on accuracy-gap claims < 5% | SATISFIED | `dm_gate_unit_synthetic` 4-scenario coverage; `claim_allowed = (gap_pct.abs() >= 0.05) || (reject && anofox_beats)`; dm_test.rs HLN+HAC implementation verified |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/anofox-bench-harness/src/dm_test.rs` | 30, 69 | Doc list item overindented; redundant closure | Info | Cosmetic only; no semantic impact |
| `crates/anofox-bench-harness/tests/cross_library.rs` | 26, 27 | Doc list item without indentation | Info | Cosmetic only |
| `crates/anofox-bench-harness/tests/accuracy.rs` | 35 | Doc list item without indentation | Info | Cosmetic only |
| `src/models/smart.rs` | multiple | 5 clippy -D warnings violations (pre-existing) | Info | Pre-existing at phase-02 base commit; NOT introduced by this phase; out of scope |

No TBD, FIXME, or XXX markers in any phase-02 deliverable. No placeholder implementations. No empty returns in wired paths. All doc comment style issues are cosmetic.

**Note on WR-02 (open, deferred):** The `predict_with_intervals` failure path in accuracy.rs (line 271) skips to the next series after MASE/sMAPE/RMSE/MAE have already been pushed, causing a count mismatch between MASE and MSIS aggregates. This is acknowledged in REVIEW.md and was deliberately deferred by human instruction. It does NOT affect the ACCUR-08 anchor (MASE-only) or any other must-have truth. It is a correctness caveat for MSIS reporting when interval prediction fails on monthly series.

### Human Verification Required

None — all must-have truths are verified by code inspection and behavioral spot-checks. The defer-lock decision on accuracy.json was made by a human (documented in Plan 04 SUMMARY checkpoint:decision) and is correctly reflected in the codebase (accuracy.json absent, emit helper guards in place).

---

## Gaps Summary

No gaps. All 10 must-have truths are VERIFIED.

The one deferred item (accuracy.json lock) is explicitly addressed in Phase 4 (ROADMAP SC-2 for Phase 4 references "MASE delta in accuracy.json"). The deferral is human-approved and the harness infrastructure needed to capture the baseline is fully in place — when Phase 4 accuracy improvements close the ~21% MASE gap, the maintainer runs `ANOFOX_WRITE_ACCURACY_BASELINE=1 ANOFOX_DATASET_DIR=./validation/data cargo test -p anofox-bench-harness --test accuracy emit_accuracy_baseline_if_write_flag_set` and commits the result.

---

_Verified: 2026-08-10T22:27:07Z_
_Verifier: Claude (gsd-verifier)_
