---
phase: 07-erm-reconciliation-variant-ridge-solve
verified: 2026-09-09T08:57:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 7: ERM Reconciliation Variant & Ridge Solve Verification Report

**Phase Goal:** Add `ReconciliationMethod::Erm { lambda }` (backward-compatible), a training-history API, the ridge solve (reconciled bottom = P·ŷ, all = S·P·ŷ), proven correct against the reference formula on a small hierarchy. Auto-lambda (ERM-04) and grouped validation (ERM-06) are Phase 8 (out of scope).
**Verified:** 2026-09-09T08:57:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                                                                                                       | Status     | Evidence                                                                                                                                                                                                                                                                             |
|----|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | `ReconciliationMethod::Erm { lambda }` exists alongside existing arms; `Eq` derive dropped; all pre-existing hierarchy tests still pass (ERM-01)                                           | VERIFIED   | Line 60: `#[derive(Debug, Clone, Copy, PartialEq)]` — `Eq` absent. Line 120-124: `Erm { lambda: f64 }` arm present. `cargo test --lib hierarchy::` reports 41 passed (32 pre-existing + 9 ERM). No downstream `==` comparisons on the enum exist.                                  |
| 2  | `with_erm_training` stores base+leaf history; missing history returns `ForecastError` with actionable hint; shape mismatches (leaf AND base length) return `ForecastError` (ERM-02)         | VERIFIED   | Lines 503-510: `with_erm_training` stores both fields. Lines 810-820: `ok_or_else` returns `InvalidParameter` with hint. Tests `erm_requires_training`, `erm_shape_mismatch`, `erm_shape_mismatch_base_hist`, `erm_empty_training_history_returns_err` — all 4 pass.               |
| 3  | `erm_reconcile` reuses in-house `cholesky`/`cholesky_solve_vec` (no faer, no feature gate); reconciled bottom = P·ŷ, all = S·P·ŷ (ERM-03)                                                 | VERIFIED   | Lines 904-946: `cholesky(n, &gram)` and `cholesky_solve_vec` are module-local functions (lines 1615, 1640); no `faer` import in `src/hierarchy/mod.rs`. `erm_coherent_multi_horizon` passes, confirming coherence at every horizon step.                                            |
| 4  | Hand-computed reference values A=16/3, B=5/2, Total=47/6 match within 1e-10; coherence Total==A+B holds (ERM-05); asymmetric ordering test (`erm_correctness_asymmetric`) passes (WR-02) | VERIFIED   | Test `erm_correctness_reference_formula` (lines 2643-2676) asserts all three values within 1e-10 plus coherence. Test `erm_correctness_asymmetric` (lines 2839-3001) runs an independent Gauss-Jordan oracle on a 4-node distinct-history hierarchy and asserts agreement within 1e-9. Both tests pass. |
| 5  | Review blockers CR-01/CR-02/CR-03 guarded; tests `erm_negative_lambda_returns_err`, `erm_nan_lambda_returns_err`, `erm_empty_training_history_returns_err` pass                            | VERIFIED   | Lines 797-803: combined `!lambda.is_finite() \|\| lambda < 0.0` guard (commit d687521). Line 868-872: T=0 guard. All three tests pass in `cargo test --lib hierarchy::tests::erm` run. Review-fix report confirms status `all_fixed` for all 6 findings (3 critical, 3 warnings). |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact                                                                      | Expected                                                   | Status      | Details                                                                                                     |
|-------------------------------------------------------------------------------|------------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------------------------|
| `src/hierarchy/mod.rs` — `Erm { lambda: f64 }` enum arm                      | `ReconciliationMethod::Erm { lambda: f64 }` variant        | VERIFIED    | Lines 98-124: arm present with full doc-comment, `Eq` dropped (line 60), `PartialEq + Copy` retained       |
| `src/hierarchy/mod.rs` — `erm_base_history` + `erm_leaf_history` fields      | Fields on `HierarchyTree`                                  | VERIFIED    | Lines 143-146: both `Option<HashMap<String, Vec<f64>>>` fields present                                     |
| `src/hierarchy/mod.rs` — `with_erm_training` builder method                  | Builder storing both history maps                          | VERIFIED    | Lines 503-510: public `&mut self` method, stores both fields                                                |
| `src/hierarchy/mod.rs` — `None` initializers in `new()` and `from_summing_matrix()` | Both constructors initialize ERM fields to `None`    | VERIFIED    | Lines 240-241 (`new()`) and lines 446-447 (`from_summing_matrix()`) both set `erm_*: None`                 |
| `src/hierarchy/mod.rs` — `erm_reconcile` private fn                          | Ridge solve: Gram, Cholesky, P rows, S·P·ŷ                 | VERIFIED    | Lines 791-947: complete implementation — Gram (lines 886-894), Cholesky (line 904-911), P (lines 913-916), S·P·ŷ (lines 918-944) |
| `src/hierarchy/mod.rs` — 9 ERM inline tests                                  | All 9 tests present and passing                            | VERIFIED    | Tests at lines 2642, 2679, 2693, 2723, 2750, 2776, 2802, 2837, 3004 — all 9 pass                          |

---

### Key Link Verification

| From                                        | To                                  | Via                                                                            | Status  | Details                                               |
|---------------------------------------------|-------------------------------------|--------------------------------------------------------------------------------|---------|-------------------------------------------------------|
| `reconcile()` match                         | `erm_reconcile`                     | `ReconciliationMethod::Erm { lambda } =>` dispatch arm                         | WIRED   | Line 592: dispatch arm present and exhaustive          |
| `erm_reconcile` history access              | `name_to_idx` / `leaves()`         | `base_hist[self.nodes[i].name.as_str()]` and `leaves()` ordering for b_stored  | WIRED   | Lines 875-883: index-order lookup confirmed correct    |
| `erm_reconcile` linear algebra              | In-house `cholesky`/`cholesky_solve_vec` | Direct function calls, no feature gate                                    | WIRED   | Lines 904-916: `cholesky(n, &gram)`, `cholesky_solve_vec(n, &l, &cross[i])` |
| `erm_reconcile` S-matrix                   | `ancestors_of` + `to_named_output`  | S built via `self.ancestors_of(leaf)` (lines 918-925); output via `self.to_named_output` (line 946) | WIRED   | Mirrors `min_trace_ols` S-build structure identically |

---

### Behavioral Spot-Checks

| Behavior                                           | Command                                                              | Result          | Status  |
|----------------------------------------------------|----------------------------------------------------------------------|-----------------|---------|
| All 9 ERM tests pass                               | `cargo test --lib "hierarchy::tests::erm"`                           | 9 passed, 0 failed | PASS |
| All 41 hierarchy tests pass (32 pre-existing + 9 ERM) | `cargo test --lib "hierarchy::"`                                  | 41 passed, 0 failed | PASS |
| Clippy clean with all features and -D warnings     | `cargo clippy --all-targets --all-features -- -D warnings`           | exit 0, no warnings | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                                                                                    | Status    | Evidence                                                                                                          |
|-------------|-------------|----------------------------------------------------------------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------|
| ERM-01      | 07-01-PLAN  | `ReconciliationMethod::Erm { lambda }` exists; fully backward-compatible                                       | SATISFIED | Enum arm at line 120; `Eq` dropped at line 60; 32 pre-existing tests unchanged; 41 total tests pass             |
| ERM-02      | 07-01-PLAN  | Training-history API stores base+leaf per node; missing history returns `ForecastError`                        | SATISFIED | `with_erm_training` builder (lines 503-510); 4 error-path tests pass (`erm_requires_training`, `erm_shape_mismatch`, `erm_shape_mismatch_base_hist`, `erm_empty_training_history_returns_err`) |
| ERM-03      | 07-01-PLAN  | ERM computes `P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹`; reconciled bottom = `P·ŷ`, all = `S·P·ŷ`                             | SATISFIED | `erm_reconcile` lines 791-947; in-house Cholesky; `erm_coherent_multi_horizon` passes for h=0,1                  |
| ERM-05      | 07-01-PLAN  | Correctness verified against reference formula within tolerance                                                 | SATISFIED | `erm_correctness_reference_formula`: A=16/3, B=5/2, Total=47/6 within 1e-10 + coherence. `erm_correctness_asymmetric`: Gauss-Jordan oracle agreement within 1e-9 on 4-node distinct-history hierarchy |

ERM-04 and ERM-06 are explicitly out of scope for Phase 7 (deferred to Phase 8 per REQUIREMENTS.md and ROADMAP).

---

### Anti-Patterns Found

| File                    | Line | Pattern                    | Severity | Impact   |
|-------------------------|------|----------------------------|----------|----------|
| `src/hierarchy/mod.rs`  | (none) | No stubs, no TODOs, no panic! in ERM path | — | None |

No `TBD`, `FIXME`, `XXX`, `TODO`, or `PLACEHOLDER` markers found in the ERM code additions. No `unwrap()` or `panic!` in the production `erm_reconcile` path. No empty implementations. The review blockers (CR-01 negative-lambda, CR-02 NaN-lambda, CR-03 T=0) were all guarded in commit `d687521` and are covered by tests.

---

### Probe Execution

No probes declared in PLAN.md or SUMMARY.md. Step 7c skipped — phase has no probe scripts.

---

### Human Verification Required

None. All ERM-01/02/03/05 criteria are covered by automated unit tests. The WASM/npm build (SC5 per VALIDATION.md) was noted as a manual-only check in that file, but it is not a requirement of this phase's success criteria (ERM-01–05 are all library-level).

---

### Gaps Summary

No gaps. All 5 must-have truths are VERIFIED, all required artifacts are WIRED and SUBSTANTIVE, all key links are confirmed, the full hierarchy suite passes (41 tests), and clippy exits clean.

The code review issued 3 blockers and 3 warnings. REVIEW-FIX.md (commit `d687521`) documents all 6 as resolved:
- CR-01 + CR-02: combined lambda guard at `erm_reconcile` entry (lines 797-803)
- CR-03: T=0 guard after `t_cap` derivation (lines 866-872)
- WR-01: doc formula corrected to `P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹` (lines 100-104)
- WR-02: `erm_correctness_asymmetric` added with independent Gauss-Jordan oracle (lines 2839-3001)
- WR-03: `erm_shape_mismatch_base_hist` added (lines 2724-2747)

---

_Verified: 2026-09-09T08:57:00Z_
_Verifier: Claude (gsd-verifier)_
