---
phase: 02-accuracy-harness-statistical-methodology
plan: "02"
subsystem: accuracy-harness
status: complete
tags:
  - accuracy
  - naive2
  - per-frequency
  - mase
  - msis
  - coverage
  - nan-guard

dependency_graph:
  requires:
    - phase: 02-01
      provides:
        - crates/anofox-bench-harness/src/loader.rs (DatasetSeries, load_m3, mase_scale, dataset_dir_from_env)
        - src/utils/metrics.rs (D-03 period-1 MASE denominator fallback)
        - crates/anofox-bench-harness/tests/accuracy.rs (tracer, make_ts_from_slice)
  provides:
    - crates/anofox-bench-harness/src/naive2.rs (Naive2, acf_at_lag, make_ts_from_slice)
    - crates/anofox-bench-harness/tests/accuracy.rs (FrequencyResult, run_accuracy_harness, per_frequency_stratification, msis_coverage_present_monthly)
  affects:
    - Plan 03 (DM test will use FrequencyResult from this plan)
    - Plan 04 (accuracy.json will be written from FrequencyResult fields)

actuals:
  tokens: 8075
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - ACF-gated seasonal/non-seasonal selection (1.645/sqrt(n) Bartlett threshold, D-08)
    - NaN/Inf-guarded nanmean() with logged skipped count before averaging (ACCUR-03)
    - Per-frequency stratified HashMap<String, FrequencyResult> (never cross-frequency aggregate, ACCUR-07)
    - MSIS+coverage scoped to monthly only (MSIS convention A4: period-1 first-diff scale, not seasonal)
    - FrequencyResult.autoets_msis/coverage as Option<f64> (None for non-interval frequencies)

key-files:
  created:
    - crates/anofox-bench-harness/src/naive2.rs
  modified:
    - crates/anofox-bench-harness/src/lib.rs
    - crates/anofox-bench-harness/tests/accuracy.rs

key-decisions:
  - "MSIS scoped to monthly only (not all 3 frequencies) to bound runtime; monthly is the ACCUR-08 anchor frequency and has the richest seasonality for interval evaluation"
  - "Tasks 2 and 3 implemented in a single accuracy.rs file extension (no separate commit per task) because FrequencyResult fields for MSIS/coverage were architected together with the per-frequency loop; splitting would have required two partial-struct states"
  - "msis() uses period-1 first-difference scaling (src/utils/metrics.rs convention A4) — documented in module doc and in-code comment; ACCUR-08 anchor remains MASE-only"
  - "FrequencyResult.skipped_nonfinite takes the max of autoets and naive2 skip counts (they should match since each series either contributes to both or neither)"

patterns-established:
  - "Per-frequency env-gated harness: dataset_dir_from_env() → None on return empty map → caller prints skip message and returns cleanly"
  - "NaN guard pattern: filter(|v| v.is_finite()) → log WARNING with count → average finite-only"
  - "ACCUR-02 temporal integrity: assert fold.train_end <= fold.test_start on every fold before any model fitting"

requirements-completed: [ACCUR-02, ACCUR-05, ACCUR-06, ACCUR-07]

coverage:
  - id: D1
    description: "Naive2 ACF-gated seasonal reference model with 90%-confidence Bartlett threshold"
    requirement: ACCUR-06
    verification:
      - kind: unit
        ref: "crates/anofox-bench-harness/src/naive2.rs#naive2_seasonal_gate_seasonal"
        status: pass
      - kind: unit
        ref: "crates/anofox-bench-harness/src/naive2.rs#naive2_seasonal_gate_flat"
        status: pass
      - kind: unit
        ref: "crates/anofox-bench-harness/src/naive2.rs#acf_at_lag_seasonal_signal"
        status: pass
    human_judgment: false
  - id: D2
    description: "Per-frequency stratified accuracy table (monthly/quarterly/yearly) for Naive2 + AutoETS with MASE, sMAPE, RMSE, MAE"
    requirement: ACCUR-07
    verification:
      - kind: integration
        ref: "crates/anofox-bench-harness/tests/accuracy.rs#per_frequency_stratification"
        status: pass
    human_judgment: true
    rationale: "Integration test requires ANOFOX_DATASET_DIR; human must verify the numerical output (monthly MASE=1.045, quarterly MASE=1.404, yearly MASE=2.860) is plausible for M3"
  - id: D3
    description: "MSIS and empirical coverage for AutoETS M3-monthly intervals (period-1 convention)"
    requirement: ACCUR-05
    verification:
      - kind: integration
        ref: "crates/anofox-bench-harness/tests/accuracy.rs#msis_coverage_present_monthly"
        status: pass
    human_judgment: true
    rationale: "Integration test requires ANOFOX_DATASET_DIR; human should verify MSIS=25.81 and coverage=0.932 are reasonable for 95% nominal intervals on M3-monthly"

duration: 9min
completed: "2026-08-10"
estimate:
  tokens: 82000
---

# Phase 02 Plan 02: Naive2 + Per-Frequency M3 Accuracy Harness Summary

**Naive2 ACF-gated baseline and per-frequency M3 Y/Q/M stratified accuracy run producing FrequencyResult table (MASE/sMAPE/RMSE/MAE/MSIS/coverage) with NaN-guarded aggregation for Plan 04 lock**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-08-10T21:02:06Z
- **Completed:** 2026-08-10T21:11:07Z
- **Tasks:** 3 (Tasks 2+3 co-implemented)
- **Files modified:** 3

## Accomplishments

- Naive2 (ACCUR-06/D-07): ACF-gated seasonal/non-seasonal reference using `1.645/sqrt(n)` Bartlett test; 5 offline unit tests pass without dataset
- Per-frequency stratified harness (ACCUR-07): `run_accuracy_harness()` produces `HashMap<String, FrequencyResult>` for M3 Yearly/Quarterly/Monthly with separate keys — cross-frequency aggregation is architecturally impossible
- NaN/Inf-guarded aggregation (ACCUR-03): `nanmean()` helper filters `is_finite()`, logs WARNING with excluded count before averaging; verified 0 skipped series on M3
- MSIS + interval coverage (ACCUR-05): `msis()` and `coverage()` from `src/utils/metrics.rs` (D-01), scoped to monthly with A4 convention documented in code and module doc
- All 3 accuracy tests pass without env var (clean skip, exit 0); all pass with `ANOFOX_DATASET_DIR`

## Verification Results

| Check | Result |
|-------|--------|
| `cargo test -p anofox-bench-harness naive2::` (offline) | 5/5 pass |
| `cargo test -p anofox-bench-harness --test accuracy` (no env) | 3/3 pass (clean skip) |
| M3-monthly: AutoETS MASE / Naive2 MASE / skipped | 1.0452 / 1.1605 / 0 |
| M3-quarterly: AutoETS MASE / Naive2 MASE / skipped | 1.4039 / 1.4197 / 0 |
| M3-yearly: AutoETS MASE / Naive2 MASE / skipped | 2.8596 / 3.1717 / 0 |
| Monthly MSIS / coverage (95% nominal) | 25.81 / 0.932 |
| `1.645` present in naive2.rs | Confirmed |
| `is_finite()` in accuracy.rs aggregation | Confirmed |
| `ForecastMetrics::compute` absent from accuracy.rs | Confirmed (comment-only) |
| `fitted_values()` absent from accuracy.rs | Confirmed |
| `fold.train_end <= fold.test_start` asserted per fold | Confirmed |
| `msis`/`coverage` imported from metrics.rs, not reimplemented | Confirmed |
| No cross-frequency aggregate key | Confirmed |

## Task Commits

1. **Task 1: Naive2 ACF-gated model** — `83e1538` (feat(02-02))
2. **Tasks 2+3: Per-frequency harness + MSIS/coverage** — `a05f0c5` (feat(02-02))

## Files Created/Modified

- `crates/anofox-bench-harness/src/naive2.rs` — Naive2 struct, acf_at_lag helper, make_ts_from_slice, 5 unit tests
- `crates/anofox-bench-harness/src/lib.rs` — Added `pub mod naive2` declaration with doc-comment update
- `crates/anofox-bench-harness/tests/accuracy.rs` — FrequencyResult struct, run_accuracy_harness(), per_frequency_stratification, msis_coverage_present_monthly tests; tracer from Plan 01 intact

## Decisions Made

- MSIS scoped to monthly only (not all 3 frequencies) to bound runtime; monthly is the ACCUR-08 anchor and has the richest seasonality for interval evaluation. Other frequencies carry `None` for `autoets_msis`/`autoets_coverage`.
- Tasks 2 and 3 implemented together in accuracy.rs: `FrequencyResult` fields for MSIS/coverage were architected simultaneously with the per-frequency loop — splitting into two partial-struct commits would have been artificial.
- `msis()` convention is period-1 first-difference scaling (src/utils/metrics.rs, A4/Pitfall 4), NOT seasonal-lag as in M4 competition. Documented in module doc and in-code comment. ACCUR-08 anchor remains MASE-only.

## Deviations from Plan

### Minor Implementation Notes

1. **Tasks 2 and 3 co-committed**: The plan lists separate tasks but both modify the same `accuracy.rs` file with tightly coupled types (`FrequencyResult.autoets_msis` is needed by both the per-frequency loop and the msis test). Implemented together in one commit. No functional deviation — all acceptance criteria from both tasks are met.

None — plan executed exactly as specified for all functional requirements.

## Known Stubs

None — all outputs use real data and produce real values.

## Threat Flags

No new threat surface introduced. T-02-NAN2 (NaN in aggregates) mitigated via `is_finite()` filter in `nanmean()`. T-02-XFREQ (cross-frequency aggregate) mitigated — verified no "all"/"combined" key exists by test assertion.

## Self-Check: PASSED

Files:
- `crates/anofox-bench-harness/src/naive2.rs` — FOUND (5 unit tests pass)
- `crates/anofox-bench-harness/src/lib.rs` — FOUND (`pub mod naive2` declared)
- `crates/anofox-bench-harness/tests/accuracy.rs` — FOUND (3 tests pass)

Commits:
- `83e1538` — feat(02-02): Naive2 ACF-gated seasonal/non-seasonal reference model
- `a05f0c5` — feat(02-02): per-frequency M3 harness with Naive2, NaN-guard, MSIS/coverage
