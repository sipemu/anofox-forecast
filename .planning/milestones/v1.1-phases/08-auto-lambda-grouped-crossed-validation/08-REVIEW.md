---
phase: 08-auto-lambda-grouped-crossed-validation
reviewed: 2026-09-09T00:00:00Z
fixed_at: 2026-09-09T00:00:00Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - src/hierarchy/mod.rs
  - docs/audits/erm-grouped-validation-results.md
findings:
  critical: 1
  warning: 4
  info: 2
  total: 7
status: all_fixed
fix_commit: 0785c2c
---

# Phase 8: Code Review Report

**Reviewed:** 2026-09-09
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Phase 8 delivers: (1) a `lambda: Option<f64>` field on `ReconciliationMethod::Erm`, (2) the
`erm_auto_lambda` helper that computes a Ledoit-Wolf-style ridge penalty from the ERM training
Gram, (3) migration of 9 Phase-7 call sites to `Some(...)`, and (4) an end-to-end accuracy test
reporting a −63.8% mean RMSSE reduction over unreconciled base forecasts.

The signature migration is complete and mechanical. The overall test structure is sound: there
is no holdout leakage, the three methods receive identical base inputs on the holdout, and
the coherence hard-assertion is properly enforced. The headline win is plausible but the
comparison is not fully fair (informational asymmetry vs MinTraceStruct, discussed below under
the accuracy-claim audit).

Two correctness defects were found in `erm_auto_lambda`: a floor bypass in the early-return
path and a scale mismatch between where `λ` is estimated (centered Gram `G_c`) and where it
is applied (uncentered ERM Gram `G = Ŷ Ŷᵀ`). One stale doc comment misquotes the assertion
threshold. The documented comparison has a cherry-pick concern worth disclosing.

---

## Narrative Findings (AI reviewer)

---

## Critical Issues

### CR-01: `delta < 1e-30` early-return bypasses the 1e-6 degeneracy floor

**File:** `src/hierarchy/mod.rs:1722-1724`

**Issue:** `erm_auto_lambda` contains an early-return path for the case where the centered Gram
`G_c` is already close to the shrinkage target (Frobenius distance `delta < 1e-30`). That path
returns `Ok(diag_ref.max(0.0))` directly, jumping past the `1e-6` floor guard at line 1748.

When the caller supplies constant base forecasts (every node's forecast is the same value at
every training step), `G_c` is the zero matrix, `trace_g = 0`, `diag_ref = 0`, and the early
return yields `Ok(0.0)`. The zero `λ` is then added to the uncentered ERM Gram `G = Ŷ Ŷᵀ`,
which for identical constant forecasts is a rank-1 matrix; Cholesky decomposition fails with
`SingularMatrix` instead of proceeding with the intended floor regularization. The comment at
line 1747 explicitly documents the floor's purpose ("prevents λ=0 degeneracy"), confirming
that returning 0.0 from the early-return path violates the stated invariant.

No test exercises the combination `lambda: None` + constant base forecasts, so this path is
completely untested.

**Fix:**
```rust
if delta < 1e-30 {
    // G_c already diagonal — return full-shrink lambda.
    // Apply the same 1e-6 floor as the main path to prevent λ=0 when diag_ref = 0
    // (e.g., all base forecasts are constant, making G_c the zero matrix).
    return Ok(diag_ref.max(1e-6));
}
```

Add a regression test:
```rust
#[test]
fn erm_auto_lambda_constant_forecasts_returns_floor() {
    // Constant base forecasts → G_c = 0 → diag_ref = 0 → must return 1e-6, not 0.0.
    let y = vec![vec![5.0, 5.0, 5.0]; 3]; // 3 nodes, T=3, all constant
    let lam = erm_auto_lambda(&y, 3, 3).unwrap();
    assert!(lam >= 1e-7, "floor must prevent λ=0, got {}", lam);
}
```

---

## Warnings

### WR-01: `λ` is calibrated against the centered Gram `G_c` but applied to the uncentered ERM Gram `G`

**File:** `src/hierarchy/mod.rs:1682-1758` (erm_auto_lambda) and `:897-905` (erm_reconcile Gram build)

**Issue:** `erm_auto_lambda` estimates the shrinkage intensity from the **centered** Gram
`G_c = Ŷ_c Ŷ_c^T` (outer product of mean-subtracted training forecasts). The returned
`λ = α · tr(G_c)/n` is then added to the **uncentered** ERM Gram `G = Ŷ Ŷᵀ` in
`erm_reconcile` (line 905). These two matrices differ by `T · μ μᵀ` (the outer product of
per-node means scaled by T), which for demand-forecasting data with large absolute means (e.g.,
nodes averaging 50–1000 units/period) can dwarf `G_c`. In such cases, `λ` is systematically
underestimated relative to the scale of `G`, weakening the regularization.

The existing `ledoit_wolf_alpha` (line 1767) correctly calibrates against the sample covariance
`S = G_c / T`—i.e., it uses `S_ij` as the centering point inside the variance accumulation loop
(line 1800)—rather than the unscaled `G_c_ij` used at line 1731 of `erm_auto_lambda`. This
inconsistency suggests the adaptation is incomplete.

The test data (leaf levels [50, 30, 40, 20]) has large means, so this mismatch is present in
the benchmark, yet the test still passes because T=20 > n=9 makes the uncentered Gram
non-singular regardless of `λ`.

**Fix:** Either compute `λ` based on the uncentered Gram `G` directly, or scale the returned
value to account for the mean outer product:

```rust
// Option A: use G (uncentered) for shrinkage estimation
// — replace G_c computation with G = Ŷ Ŷᵀ, set diag_ref = tr(G)/n
// This is simpler and consistent with the matrix being regularized.

// Option B: apply the existing ledoit_wolf_alpha logic directly:
// Build sample_cov = G_c / (T-1), diag = [G_c[i,i]/(T-1)], then reuse ledoit_wolf_alpha.
// This would make erm_auto_lambda consistent with MinTraceShrink's formula.
```

---

### WR-02: Stale doc comment misquotes the RMSSE assertion threshold

**File:** `src/hierarchy/mod.rs:3218`

**Issue:** The `erm_grouped_crossed_end_to_end_accuracy` test's doc comment reads:

```
/// Soft-asserts ERM RMSSE ≤ unreconciled RMSSE × 1.5 (tightened in Task 3).
```

The actual assertion at line 3418 uses `× 1.1`, not `× 1.5`. The comment was not updated when
the threshold was tightened, so it misrepresents the actual gate. Any reader relying on the doc
comment to understand what the test enforces will have a false picture.

**Fix:**
```rust
/// Soft-asserts ERM RMSSE ≤ unreconciled RMSSE × 1.1.
```

---

### WR-03: Accuracy-claim comparison is methodologically asymmetric (informational fairness)

**File:** `docs/audits/erm-grouped-validation-results.md:14-16`

**Issue:** The headline −63.8% RMSSE reduction positions ERM auto-λ against MinTraceStruct as
if they are peers. They are not informationally equivalent:

- **ERM** uses T=20 periods of base-forecast history **plus** T=20 periods of true leaf
  actuals to learn the projection matrix P. It has the ability to "look back" and correct
  systematic incoherence patterns.
- **MinTraceStruct** uses only structural node counts (leaves below each ancestor). It sees
  neither historical forecasts nor actuals. It is the weakest MinTrace variant and is chosen
  precisely because it requires zero additional data.

A practitioner comparing methods at equal information cost (e.g., providing MinTraceShrink with
the same T=20 base-forecast residuals that ERM uses) would see a smaller gap. The docs do not
mention this asymmetry, making the −63.8% figure misleading as a standalone claim.

No data leakage was found: `leaf_hist_map` uses `true_all[..t_train]` (training window only),
and the holdout RMSSE uses `base_holdout_all` (identical inputs for all three methods). The
comparison is not *rigged* but is *cherry-picked* by choosing the least-data-hungry baseline.

**Fix:** Add a note to `erm-grouped-validation-results.md`:

```markdown
> **Information note:** MinTraceStruct requires only the hierarchy structure (no historical data),
> while ERM auto-λ uses T=20 training base forecasts plus T=20 leaf actuals.
> A fairer data-parity comparison would pair ERM against MinTraceShrink supplied with the
> same T=20 residuals. The −63.8% result reflects ERM's full informational advantage.
```

---

### WR-04: Documented headline numbers are not assertion-locked and can drift silently

**File:** `docs/audits/erm-grouped-validation-results.md:12-16`

**Issue:** The specific RMSSE values (2.287392, 2.110507, 0.827239, λ_auto=1.492916) are
hard-coded in the audit document but the test only asserts `erm_mean <= unrec_mean * 1.1`.
The test would pass even if the actual values drifted significantly (e.g., if the LCG output
changed on a different platform or rustc version), leaving the committed document stale. The
project's Core Value states "every claimed capability is measured" — numbers in a results doc
that are not assertion-guarded are not truly measured, they are observed once and assumed stable.

**Fix:** Add an exact-value assertion (with ±1% tolerance) for the committed numbers, or at
minimum document that the results are platform-sensitive and not pinned:

```rust
// Lock the headline numbers to ±1% to catch platform-induced drift.
assert!((unrec_mean - 2.287392).abs() < 0.023,
    "unreconciled RMSSE drifted: {}", unrec_mean);
assert!((erm_mean - 0.827239).abs() < 0.009,
    "ERM RMSSE drifted: {}", erm_mean);
```

---

## Info

### IN-01: Coherence "proof" does not assert that unreconciled base forecasts are incoherent

**File:** `src/hierarchy/mod.rs:3391-3403`

**Issue:** The test hard-asserts that ERM output is coherent (1e-8 tolerance across all 9 nodes
× 5 steps). However, it never asserts that the *unreconciled* base holdout forecasts violate
coherence, which would make the proof meaningful by demonstrating the transformation was
necessary. The test comment says "the hard coherence assertion ... would fail for unreconciled"
but this is a claim, not a checked invariant. The omission is low-risk (independent per-node
noise guarantees incoherence by construction) but weakens the evidentiary value.

**Fix:** Add a simple incoherence spot-check:
```rust
// Verify that unreconciled base forecasts ARE incoherent (making the coherence proof meaningful).
let base_total_0 = base_holdout_all[0][0];
let base_sum_leaves_0 = base_holdout_all[5][0] + base_holdout_all[6][0]
    + base_holdout_all[7][0] + base_holdout_all[8][0];
assert!(
    (base_total_0 - base_sum_leaves_0).abs() > 1e-6,
    "base forecasts must be incoherent for the coherence proof to be meaningful"
);
```

---

### IN-02: Comment at line 1747 misdescribes the floor trigger condition

**File:** `src/hierarchy/mod.rs:1747`

**Issue:** The comment reads "Fallback: if diag_ref ≈ 0, all base forecasts are near-zero; use
minimal regularization." This is inaccurate: `diag_ref = 0` (which triggers `lambda < 1e-15`)
also occurs when base forecasts are **constant but non-zero** (all variance is zero,
`G_c = 0 matrix`, `trace_g = 0`). The comment implies the floor only guards against near-zero
absolute forecasts, not near-zero variance forecasts, confusing the two distinct cases.

**Fix:**
```rust
// Fallback: if diag_ref ≈ 0, the centered Gram has negligible variance
// (either all base forecasts are near-zero OR all are nearly constant with zero
// across-time variation). Use minimal regularization to prevent a singular solve.
let resolved = if lambda < 1e-15 { 1e-6 } else { lambda };
```

---

## Accuracy-Claim Audit (Item 3 — detailed verdict)

**Is there holdout leakage?** No. `leaf_hist_map` slices `true_all[5..8][..t_train]` (T=20
training window only). The holdout `true_all[5..8][t_train..]` is never passed to ERM before
prediction. `base_erm` fed to `reconcile()` at prediction time is `base_holdout_all` (the noisy
holdout base forecasts), which is the same for all three methods. Training and test windows are
strictly disjoint.

**Do all methods get the same base inputs?** Yes. `base_holdout_all` is identical for
Unreconciled (used directly), MinTraceStruct (reconciled), and ERM (reconciled). The difference
is what each method does with those inputs, not the inputs themselves.

**Is the RMSSE scale identical across methods?** Yes. All three calls to `rmsse()` pass
`true_train[i]` as the scale denominator. This is documented in `erm-grouped-validation-results.md`
("training scale = in-sample true coherent values"). It applies consistently, so it does not
favor any one method.

**Is the data generation neutral?** Substantially yes. The LCG is seeded at 42 with
well-known Knuth constants, AR(1) noise is independent across leaves, and aggregate series are
computed by exact summation of leaf series (no privileged ERM structure in the DGP). One mild
asymmetry: the noise added to produce base forecasts (`rng() * 4.0`) comes from the same LCG
stream used to generate the true series, meaning training and holdout base-forecast noise share
the same random state. This is not expected to systematically bias results in either direction.

**Is the informational comparison fair?** Not fully. ERM uses T=20 true leaf actuals (B_stored)
plus T=20 noisy base forecasts (Ŷ_stored). MinTraceStruct uses only hierarchy structure. The
−63.8% figure quantifies ERM's advantage over a zero-data baseline (WR-03 above). The win is
real and non-trivial, but should be disclosed as reflecting an information advantage, not just
algorithmic superiority.

**Overall verdict:** The accuracy comparison is *honest* but *cherry-picked*. No leakage, no
rigging, same base inputs. However, choosing MinTraceStruct (the weakest MinTrace variant, zero
data requirement) as the sole reference inflates the apparent gap. The milestone's Core Value
requires honest numbers — the numbers themselves are accurate, but the framing omits material
context (see WR-03 for the recommended disclosure fix).

---

_Reviewed: 2026-09-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
