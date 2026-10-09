---
phase: 07-erm-reconciliation-variant-ridge-solve
plan: "01"
subsystem: forecasting
tags: [rust, hierarchy, reconciliation, erm, ridge-regression, cholesky]

requires: []
provides:
  - "ReconciliationMethod::Erm { lambda: f64 } variant in src/hierarchy/mod.rs"
  - "with_erm_training() builder on HierarchyTree accepting base + leaf history HashMaps"
  - "erm_reconcile() private fn computing P = B'Ŷ(Ŷ'Ŷ+λI)⁻¹ via in-house Cholesky"
  - "Four inline tests: correctness (1e-10), missing-history error, shape mismatch, multi-horizon coherence"
affects:
  - "phase 8 (auto-lambda ERM) — can build directly on top of Erm arm and with_erm_training"

actuals:
  tokens: 3859    # 15436 chars / 4 over the realized diff
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "ERM training history stored as Option<HashMap<String,Vec<f64>>> fields mirroring actuals/residuals pattern"
    - "Ridge solve: Gram G=Ŷ'Ŷ+λI Cholesky-factored once, each P row solved via cholesky_solve_vec (m solves)"
    - "Coherent output via S·P·ŷ reusing min_trace_ols summing-matrix construction (ancestors_of + to_named_output)"

key-files:
  created: []
  modified:
    - "src/hierarchy/mod.rs — all ERM additions: enum arm, struct fields, builder, private fn, four tests"

key-decisions:
  - "Eq derive dropped from ReconciliationMethod (f64 is not Eq); PartialEq + Copy retained — no downstream breakage (no == comparisons on the enum exist)"
  - "T < n is NOT hard-blocked; lambda rescues a rank-deficient Gram — the reference test (T=2, n=3) relies on this; SingularMatrix error surfaces when lambda cannot rescue the solve"
  - "All three plan tasks implemented in a single atomic commit — the implementation was indivisible (enum + fields + init + builder + dispatch + solve + all four tests in one file)"

patterns-established:
  - "ERM solve pattern: build Ŷ_stored/B_stored in internal index order, Gram+ridge, Cholesky once, solve P rows, S·P·ŷ — mirrors min_trace_ols structural template"

requirements-completed: [ERM-01, ERM-02, ERM-03, ERM-05]

coverage:
  - id: D1
    description: "ReconciliationMethod::Erm { lambda: f64 } variant compiles and all 32 pre-existing hierarchy tests pass (ERM-01)"
    requirement: ERM-01
    verification:
      - kind: unit
        ref: "src/hierarchy/mod.rs#hierarchy::tests::* (36 tests total)"
        status: pass
    human_judgment: false
  - id: D2
    description: "with_erm_training() builder stores base+leaf history; missing history returns ForecastError with actionable hint (ERM-02)"
    requirement: ERM-02
    verification:
      - kind: unit
        ref: "src/hierarchy/mod.rs#hierarchy::tests::erm_requires_training"
        status: pass
      - kind: unit
        ref: "src/hierarchy/mod.rs#hierarchy::tests::erm_shape_mismatch"
        status: pass
    human_judgment: false
  - id: D3
    description: "Ridge solve P=B'Ŷ(Ŷ'Ŷ+λI)⁻¹ produces coherent bottom=P·ŷ and all=S·P·ŷ across all horizon steps (ERM-03)"
    requirement: ERM-03
    verification:
      - kind: unit
        ref: "src/hierarchy/mod.rs#hierarchy::tests::erm_coherent_multi_horizon"
        status: pass
    human_judgment: false
  - id: D4
    description: "Hand-computed reference values A=16/3, B=5/2, Total=47/6 match within 1e-10 and coherence Total==A+B holds (ERM-05)"
    requirement: ERM-05
    verification:
      - kind: unit
        ref: "src/hierarchy/mod.rs#hierarchy::tests::erm_correctness_reference_formula"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-09
status: complete
---

# Phase 7 Plan 01: ERM Reconciliation Variant & Ridge Solve Summary

**Fixed-λ ERM hierarchical reconciliation via Cholesky ridge solve (P=B'Ŷ(Ŷ'Ŷ+λI)⁻¹), proven correct against hand-computed reference values within 1e-10 and coherent across all horizon steps**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-09-09T (continuation from crashed executor)
- **Completed:** 2026-09-09
- **Tasks:** 3 (all completed atomically in one commit — see Deviations)
- **Files modified:** 1 (src/hierarchy/mod.rs)

## Accomplishments

- Resolved 3 pre-existing compile errors left by prior crashed executor (missing field initializers in new() + from_summing_matrix(), non-exhaustive match in reconcile())
- Added `ReconciliationMethod::Erm { lambda: f64 }` with full doc-comment referencing Ben Taieb & Koo (2019)
- Dropped `Eq` derive from `ReconciliationMethod` (f64 is not Eq) — PartialEq + Copy retained, no downstream breakage
- Added `erm_base_history` + `erm_leaf_history` fields and `with_erm_training()` builder mirroring set_actuals/set_residuals
- Implemented `erm_reconcile()`: Gram G=Ŷ'Ŷ+λI Cholesky-factored once, P solved row-by-row, bottom=P·ŷ, all=S·P·ŷ
- All 4 ERM tests pass; all 36 hierarchy tests pass (32 pre-existing + 4 new); clippy --all-features -D warnings clean

## Task Commits

All three tasks were implemented atomically in one commit (the implementation in a single file was indivisible):

1. **Tasks 1-3: Complete ERM path** - `d40e294` (feat(07-01))

**Plan metadata:** (pending — created after this summary)

## Files Created/Modified

- `src/hierarchy/mod.rs` — Erm enum arm, two new struct fields, None initializers in new() + from_summing_matrix(), with_erm_training() builder, dispatch arm in reconcile(), erm_reconcile() private fn (~325 lines net added), four inline tests

## Decisions Made

- **Eq derive dropped** — f64 is not Eq; PartialEq + Copy preserved. grep confirms no `==` comparisons on ReconciliationMethod exist anywhere in the codebase. This is a load-bearing change required for the f64 lambda field to compile.
- **T < n not hard-blocked** — The reference test deliberately uses T=2, n=3 (rank-deficient G) to demonstrate that lambda rescues the solve. Hard-blocking T<n would break the documented test case. Instead, SingularMatrix is returned when Cholesky fails (i.e., when lambda is insufficient to rescue the solve). Shape validation hard-blocks missing entries and unequal-length vectors per ERM-02.
- **All tasks in one commit** — The plan listed 3 tasks but all work is in a single file. The tracer (Task 1) + robustness (Task 2) + coherence (Task 3) implementation were indivisible at the code level; committing them atomically ensures no intermediate broken state lands in the branch.

## Deviations from Plan

**1. [Rule 1 - Process] All three tasks committed in one atomic commit rather than separately**
- **Found during:** Task 1 implementation
- **Issue:** The plan specified three separate commits (tracer, robustness, coherence), but all work lives in a single file (`src/hierarchy/mod.rs`) and the four tests were written together. Splitting would have required artificially partial states.
- **Fix:** Committed all implementation including all four tests in one atomic commit (d40e294).
- **Verification:** All 36 hierarchy tests green, clippy clean.
- **Impact:** No correctness or quality impact; the commit message documents all deliverables clearly.

---

**Total deviations:** 1 (process: atomic commit spanning all tasks)
**Impact on plan:** Zero impact on correctness or requirements coverage. All ERM-01/02/03/05 criteria met.

## Issues Encountered

- **Prior executor crash:** The previous executor left the code in a non-compiling state with 3 errors (missing field initializers and non-exhaustive match). These were resolved as the first action before any new code was added.

## Security / Threat Surface

Shape validation fully addresses T-07-01: every node/leaf presence is checked, unequal-length vectors return DimensionMismatch before any indexing. T-07-02: cholesky() returns Err (never panics); wrapped with ERM-specific SingularMatrix message. T-07-03 (NaN/Inf): accepted — cholesky detects non-positive diagonal and returns SingularMatrix.

## Known Stubs

None — all deliverables are fully implemented and tested.

## Next Phase Readiness

Phase 8 (auto-lambda ERM, ERM-04; grouped end-to-end validation, ERM-06) can build directly on top of the `Erm` arm and `with_erm_training()` API without restructuring:
- Auto-lambda: add a new variant `ErmAuto` or extend `Erm` with an `Option<f64>` lambda
- JS bindings: `parse_method()` catch-all compiles fine; add `"erm"` string arm in Phase 8 when exposing to JS

---
*Phase: 07-erm-reconciliation-variant-ridge-solve*
*Completed: 2026-09-09*
