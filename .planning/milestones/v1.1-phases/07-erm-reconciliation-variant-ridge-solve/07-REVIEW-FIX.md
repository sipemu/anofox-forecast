---
phase: 07-erm-reconciliation-variant-ridge-solve
fixed_at: 2026-09-09T00:00:00Z
review_path: .planning/phases/07-erm-reconciliation-variant-ridge-solve/07-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 7: Code Review Fix Report

**Fixed at:** 2026-09-09
**Source review:** `.planning/phases/07-erm-reconciliation-variant-ridge-solve/07-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (CR-01, CR-02, CR-03, WR-01, WR-02, WR-03)
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01 + CR-02: Guard invalid lambda (negative and NaN)

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** d687521
**Applied fix:** Added a guard at the entry of `erm_reconcile`, before any matrix
construction, that rejects `lambda` values that are not finite or are negative:

```rust
if !lambda.is_finite() || lambda < 0.0 {
    return Err(ForecastError::InvalidParameter(format!(
        "ERM: lambda must be finite and non-negative, got {lambda}"
    )));
}
```

This closes both the negative-lambda silent-amplification path (CR-01) and the
NaN-bypasses-Cholesky path (CR-02) with a single combined check. Two new tests
(`erm_negative_lambda_returns_err`, `erm_nan_lambda_returns_err`) cover both cases.

---

### CR-03: Guard T=0 (empty training history)

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** d687521
**Applied fix:** After deriving `t_cap = t_base`, added:

```rust
if t_cap == 0 {
    return Err(ForecastError::InvalidParameter(
        "ERM: training history must contain at least one time period (T ≥ 1)".into(),
    ));
}
```

Without this guard, a T=0 input produces a zero Gram (`+λI` makes it SPD), a zero
cross term, a zero P matrix, and all-zero reconciled forecasts returned as `Ok(...)`.
New test `erm_empty_training_history_returns_err` exercises this path.

---

### WR-01: Correct doc-comment formula

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** d687521
**Applied fix:** Both doc-comment occurrences of the ERM formula were changed from
the dimensionally inconsistent `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹` to the layout-correct
`P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹`. The enum-arm doc also now notes that the Gram is
n×n (nodes×nodes) and that this is the same ERM estimator as the reference paper,
just written for this codebase's row-major node×T layout.

---

### WR-02: Asymmetric correctness test with independent oracle

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** d687521
**Applied fix:** Added `erm_correctness_asymmetric` — a 4-node (Total→{A,B,C})
hierarchy where every node has a DISTINCT training history (no two rows of Ŷ are
identical). The test includes a self-contained Gauss-Jordan dense oracle that
computes P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹ independently (no shared code with production
`erm_reconcile`), then asserts agreement within 1e-9 and coherence (Total = A+B+C).

The asymmetric histories mean any index-transposition or ordering bug in
`y_stored`, `gram`, `cross`, `p`, or `y_hat` will cause the two paths to disagree.
The test confirmed the production code was correct — the oracle and `erm_reconcile`
agree to within numerical precision.

---

### WR-03: Base history length-mismatch test

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** d687521
**Applied fix:** Added `erm_shape_mismatch_base_hist` — sets node "B"'s base history
to length 3 while Total and A have length 2, then asserts `reconcile` returns `Err`.
This exercises the existing validation loop (lines 822–835) that checks each node's
base_history vector matches the length of the first node's vector.

---

## Verification

All verification ran in the main checkout (no isolated worktree — `workflow.use_worktrees` not configured).

- `cargo test --lib "hierarchy::tests::erm"`: 9 passed, 0 failed (was 5 before fixes)
- `cargo test --lib "hierarchy::"`: 41 passed, 0 failed (was 36 before fixes)
- `cargo clippy --all-targets --all-features -- -D warnings`: clean

---

_Fixed: 2026-09-09_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
