---
phase: 08-auto-lambda-grouped-crossed-validation
fixed_at: 2026-09-09T00:00:00Z
review_path: .planning/phases/08-auto-lambda-grouped-crossed-validation/08-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 8: Code Review Fix Report

**Fixed at:** 2026-09-09
**Source review:** `.planning/phases/08-auto-lambda-grouped-crossed-validation/08-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: `delta < 1e-30` early-return bypasses the 1e-6 degeneracy floor

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** 0785c2c
**Applied fix:** Changed `diag_ref.max(0.0)` to `diag_ref.max(1e-6)` in the `delta < 1e-30`
early-return path of `erm_auto_lambda`. Added regression test
`erm_auto_lambda_constant_forecasts_returns_floor` (constant base forecasts → λ ≥ 1e-7).
Note: the constant-forecast case for n>1 actually reaches the gamma path (delta > 0 since
off-diagonals are non-zero), but gamma=0 gives α=0, λ=0 → the 1e-15 fallback floor fires.
The 1e-6 guard on the early-return path covers the n=1 (diagonal G) degenerate case.

---

### WR-01: `λ` calibrated against centered Gram but applied to uncentered ERM Gram

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** 0785c2c
**Applied fix:** Rewrote `erm_auto_lambda` to compute `diag_ref = tr(G)/n` and `delta =
||G - diag_ref·I||_F²` from the **uncentered** Gram `G = ŶŶᵀ` (matching the matrix that
`erm_reconcile` regularizes), while retaining centered outer-product deviations for `gamma`
(noise estimation requires zero-mean residuals). The shrinkage target and the matrix being
regularized are now self-consistent. Re-captured values (post-fix, seed=42, T=20, H=5):
- ERM auto-λ RMSSE: **0.921839** (was 0.827239 with centered Gram)
- λ_auto: **0.000197** (was 1.492916 — small because T=20 >> n=9, Gram well-conditioned)
- Coherence hard-assertion: **intact** (all 9 nodes × 5 steps, tol 1e-8)

---

### WR-02: Stale doc comment misquotes the RMSSE assertion threshold

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** 0785c2c
**Applied fix:** Updated the `erm_grouped_crossed_end_to_end_accuracy` doc comment from
"Soft-asserts ERM RMSSE ≤ unreconciled RMSSE × 1.5 (tightened in Task 3)." to
"Soft-asserts ERM RMSSE ≤ unreconciled RMSSE × 1.1." — matching the actual assertion.
Also updated the inline comment to reflect the new observed ratio (0.403) after WR-01.

---

### WR-03: Accuracy-claim comparison methodologically asymmetric (informational fairness)

**Files modified:** `docs/audits/erm-grouped-validation-results.md`
**Commit:** 0785c2c
**Applied fix:** Added an information note disclosing that MinTraceStruct uses only hierarchy
structure (zero historical data) while ERM auto-λ uses T=20 training base forecasts + T=20
leaf actuals. Notes that a fairer data-parity comparison would use MinTraceShrink with the
same residuals. Also added a measurement-note indicating the headline numbers are
assertion-locked in the test to ±1%. MinTraceShrink was not added as a 4th row because it
requires residuals computed from a separate fit not already present in this test setup; the
disclosure note is the accepted mitigation per reviewer guidance.

---

### WR-04: Documented headline numbers not assertion-locked (can drift silently)

**Files modified:** `src/hierarchy/mod.rs`, `docs/audits/erm-grouped-validation-results.md`
**Commit:** 0785c2c
**Applied fix:** Added drift-lock assertions (±1% tolerance) in
`erm_grouped_crossed_end_to_end_accuracy` for all three headline RMSSE values:
- `unrec_mean`: expected 2.287392, tolerance ±0.023
- `mt_mean`: expected 2.110507, tolerance ±0.022
- `erm_mean`: expected 0.921839, tolerance ±0.010

Updated the results doc with re-captured values post WR-01 fix and the measurement note.
Note: these locked values were captured fresh with `-- --nocapture` and match the test output.

---

### IN-01: Coherence "proof" does not assert that unreconciled base forecasts are incoherent

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** 0785c2c
**Applied fix:** Added an incoherence spot-check at the start of the coherence section in
`erm_grouped_crossed_end_to_end_accuracy` that asserts `|base_total_0 - sum(base_leaves_0)| > 1e-6`
at h=0. This makes the ERM coherence proof meaningful by demonstrating that the unreconciled
base forecasts genuinely violate the constraint that ERM is enforcing.

---

### IN-02: Comment at line 1747 misdescribes the floor trigger condition

**Files modified:** `src/hierarchy/mod.rs`
**Commit:** 0785c2c
**Applied fix:** Updated the fallback comment from "all base forecasts are near-zero; use
minimal regularization" to "the centered Gram has negligible variance (either all base
forecasts are near-zero OR all are nearly constant with zero across-time variation). Use
minimal regularization to prevent a singular solve." This correctly describes both trigger
cases (near-zero absolute forecasts AND near-constant forecasts with zero variance).

---

## Skipped Issues

None — all findings were fixed.

---

## Verification

**Method:** All fixes verified via `cargo test --lib "hierarchy::tests::erm"` (13 tests pass)
and `cargo test --lib "hierarchy::"` (45 tests pass, no regressions).
**Clippy:** `cargo clippy --all-targets --all-features -- -D warnings` — clean (Finished, no errors).
**Coherence invariant:** Hard-assertion for ERM auto-λ output coherence (1e-8 tol, 9 nodes × 5 steps)
remained intact after the WR-01 λ scaling change — confirmed by test output.
**Verification ran in:** main checkout (workflow.use_worktrees not checked; edits made directly).

## Re-captured Headline Numbers (post-fix)

```
=== ERM Grouped/Crossed Validation — Mean RMSSE across 9 nodes ===
  Unreconciled:   2.287392
  MinTraceStruct: 2.110507
  ERM auto-λ:     0.921839
  Auto-λ selected: 0.000197
```

ERM vs Unreconciled: −59.7% (was −63.8% before WR-01 fix; the change reflects the
λ now being scaled to G instead of G_c — λ_auto dropped from 1.49 to 0.0002 because
G's diagonal already dominated G_c when T=20 >> n=9, making strong regularization unnecessary).

---

_Fixed: 2026-09-09_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
