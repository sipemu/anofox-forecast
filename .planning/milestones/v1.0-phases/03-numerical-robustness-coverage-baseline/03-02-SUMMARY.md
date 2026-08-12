---
phase: 03-numerical-robustness-coverage-baseline
plan: 02
subsystem: testing/property-robustness
tags: [robustness, property-tests, proptest, changepoint, mstl, cross-validation]
status: complete
requirements: [ROBUST-03]

dependency_graph:
  requires:
    - 03-01 (edge-case suite, ROBUST-01/02)
  provides:
    - tests/property_robustness.rs
    - .proptest-regressions/ (tracked corpus directory)
    - ROBUST-03 property suite (changepoint/MSTL/CV — no-panic, no-NaN, temporal integrity)
    - Proptest counterexample report for 03-03 gap inventory (none found)
  affects:
    - .proptest-regressions/.gitkeep

tech_stack:
  added: []
  patterns:
    - ProptestConfig::with_cases(N) bounds per block for CI runtime budget
    - make_bkps() helper generates valid strictly-increasing breakpoints with terminal == n
    - if let Some(r) = result { ... } handles MSTL Option<MSTLResult> without unwrap
    - if let Ok(folds) = result { ... } handles CvFoldGenerator Ok/Err without unwrap
    - Three separate proptest! blocks with different case counts (50/30/100)

key_files:
  created:
    - tests/property_robustness.rs
    - .proptest-regressions/.gitkeep
  modified: []

decisions:
  - MSTLResult field is seasonal_components (Vec<Vec<f64>>), not seasonal — RESEARCH.md example was wrong; iterated over seasonal_components
  - Three separate proptest! macro blocks used (one per subsystem) to allow different with_cases counts per block; changepoint uses 50, MSTL uses 30 (heavier fitting), CV uses 100 (cheap arithmetic)
  - make_bkps() helper avoids duplicating valid-breakpoint logic across all three changepoint properties
  - No top-level .unwrap() on any .decompose() or .generate() call; malformed-input paths left to deterministic edge_case_robustness.rs

metrics:
  duration_mins: 43
  completed_date: "2026-08-11"
  tasks_completed: 3
  tasks_total: 3
  commits: 3
  files_created: 2
  files_modified: 0

estimate:
  tokens: 58000

actuals:
  tokens: 12000
  tasks: 3
  commits: 3
---

# Phase 03 Plan 02: Property-Based Robustness Suite (ROBUST-03) Summary

**One-liner:** Six bounded proptest blocks proving changepoint metric reflexivity, MSTL no-panic/no-NaN, and CV temporal integrity across random inputs; corpus directory committed for stable CI seeds.

## What Was Built

### Task 1 — Tracer: Changepoint Self-Match Property (commit c94397e)

Created `tests/property_robustness.rs` with a single `proptest!` block (`with_cases(50)`) and one property:

- `changepoint_precision_recall_self_match_is_perfect` — generates valid strictly-increasing breakpoints with terminal `n`, calls `precision_recall(&bkps, &bkps, 0)`, asserts `precision ≈ 1.0`, `recall ≈ 1.0`, and `f1.is_finite()`.

Import path: `anofox_forecast::changepoint::{hausdorff, precision_recall, randindex}` (verified via `changepoint/mod.rs:64`). Compiles and passes end-to-end.

### Task 2 — Expand: Remaining Property Blocks (commit 8cc246f)

Expanded `tests/property_robustness.rs` to six tests across three subsystems:

**Changepoint metrics (50 cases):**
- `changepoint_hausdorff_reflexive` — `hausdorff(&bkps, &bkps) ≈ 0.0` (abs < 1e-12)
- `changepoint_randindex_self_is_one` — `randindex(&bkps, &bkps, n) ≈ 1.0` (abs < 1e-10), `is_finite()`

**MSTL decomposition (30 cases):**
- `mstl_decompose_never_panics` — random bounded f64 slice (10..100 values, period 2..8); `decompose` returns `Option`; when `Some`, all `trend`, `seasonal_components[i]`, and `remainder` values are `is_finite()`; `trend.len() == values.len()`
- `mstl_decompose_constant_series` — constant slice `[c; n]`; when `Some`, trend and remainder are finite

**CV fold generator (100 cases):**
- `cv_generate_never_panics` — random `series_len` 0..500, `horizon` 1..50, `n_folds` 1..10, `min_window` 2..50, `gap` 0..10; when `Ok(folds)`, every fold satisfies `train_end <= test_start`, `test_end <= series_len`, and `train_size() >= min_window`

All six tests pass. Zero panics. Zero NaN assertions violated. No `.decompose()/.generate().unwrap()` at top level. Suite runtime: < 0.05 s.

**Key deviation caught during implementation:** RESEARCH.md §Proptest Strategies documented the MSTL component access as `r.seasonal` but the actual `MSTLResult` field is `seasonal_components: Vec<Vec<f64>>` (verified in `src/seasonality/mstl.rs:17`). Fixed to iterate `r.seasonal_components`.

### Task 3 — Commit Proptest Regression Corpus (commit 1aa2284)

- `.proptest-regressions/` was not present and not gitignored
- Created `.proptest-regressions/.gitkeep` explaining the purpose of the directory
- `git ls-files .proptest-regressions/` confirms it is tracked
- `git check-ignore .proptest-regressions` exits non-zero (not ignored)
- No `proptest-regressions` line in `.gitignore`

## Proptest Counterexample Report (for Plan 03-03 Gap Inventory)

**No counterexamples recorded.** All six property tests passed across all randomly generated inputs during development. The `.proptest-regressions/` directory contains only the `.gitkeep` placeholder — no shrunk failing cases were produced.

Plan 03-03 gap inventory: **no P1 rows from this plan.** The three ROADMAP-flagged fragile paths (changepoint metrics, MSTL decompose, CvFoldGenerator) held their invariants under all sampled inputs.

## Observed Per-Block Runtime

| Block | Cases | Subsystem | Observed Runtime |
|-------|-------|-----------|-----------------|
| changepoint metrics (3 tests × 50 cases) | 150 total | changepoint::metrics | < 0.01 s |
| MSTL decomposition (2 tests × 30 cases) | 60 total | seasonality::MSTL | < 0.01 s |
| CV fold generator (1 test × 100 cases) | 100 total | utils::CvFoldGenerator | < 0.01 s |
| **Full suite** | **310 total** | all | **0.01–0.02 s** |

Well within the 30 s CI budget. The MSTL and CV blocks were bounded conservatively at 30 and 100 cases; both are eligible for higher case counts if deeper coverage is wanted in future phases.

## Verification Results

- `cargo test --test property_robustness --all-features`: **6 passed, 0 failed, 0 ignored**
- `cargo clippy --all-features -- -D warnings`: **clean**
- `.proptest-regressions/` is tracked in git and not gitignored
- No `.decompose()/.generate().unwrap()` at top level in the test file
- `grep -qE 'with_cases\([0-9]+\)'` succeeds (all blocks bounded)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed MSTLResult field name mismatch**
- **Found during:** Task 2, first compile attempt of MSTL block
- **Issue:** RESEARCH.md §Proptest Strategies documented access as `r.seasonal.iter().flat_map(|s| s.iter())` but `MSTLResult` struct defines the field as `seasonal_components: Vec<Vec<f64>>` (`src/seasonality/mstl.rs:17`), not `seasonal`
- **Fix:** Changed to iterate `r.seasonal_components`
- **Files modified:** `tests/property_robustness.rs`
- **Commit:** 8cc246f

### Notes

None — plan executed successfully after the field-name correction above.

## Known Stubs

None. All property blocks assert concrete invariants (no-NaN, finite components, temporal integrity). No placeholder data or unasserted properties.

## Threat Mitigation Verification

| Threat ID | Mitigation | Status |
|-----------|-----------|--------|
| T-03-04 (DoS: panic on random input) | Bounded proptest blocks assert no-panic | Mitigated — 6 tests, 310 cases, 0 panics |
| T-03-05 (NaN/Inf in metric output) | prop_assert is_finite on every MSTL component and every metric field | Mitigated |
| T-03-06 (temporal leakage in CV folds) | prop_assert train_end <= test_start and test_end <= series_len | Mitigated |
| T-03-07 (flaky CI from unstable seeds) | .proptest-regressions/ committed and not gitignored | Mitigated |

## Self-Check

### Verified: Created Files Exist

- `tests/property_robustness.rs`: EXISTS (183 lines)
- `.proptest-regressions/.gitkeep`: EXISTS

### Verified: Commits Exist

- c94397e (tracer): EXISTS
- 8cc246f (expand): EXISTS
- 1aa2284 (corpus): EXISTS

## Self-Check: PASSED
