---
phase: 06-multiplicative-guard-bug-class-audit
reviewed: 2026-09-09T07:40:49Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - tests/mult05_guard_audit_assertions.rs
  - docs/audits/multiplicative-guard-audit.md
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-09-09T07:40:49Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

This is a documentation + guard-assertion-test phase. No production `src/` code was changed. Two
integration tests were added in `tests/mult05_guard_audit_assertions.rs` and a 19-row audit
inventory was committed to `docs/audits/multiplicative-guard-audit.md`.

The overall audit architecture and MULT-06 zero-offender conclusion are sound. All cited guard
sites in the source were verified to exist at or within one line of the cited numbers, and the
19-row scope is complete against the CONTEXT.md in-scope list. The PASS verdicts for MFLES, Theta
family, ETS family, and Laplace leaves are well-supported.

Three substantive issues were found: both guard-assertion tests are **tautological** (their
assertions pass regardless of whether the guard fires), one audit row mixes up a row-13 rationale
into row-11, and the BoxCox row-14 rationale contains two quantitative inaccuracies. None of these
invalidate the zero-offender conclusion, but the tautological tests are the most significant
defect: they provide no regression value — removing the guarded code would not cause them to fail.

---

## Warnings

### WR-01: Both guard-assertion tests are tautological — they pass whether or not the guard fires

**File:** `tests/mult05_guard_audit_assertions.rs:49–106`

**Issue:** Both tests assert `forecast_max < 2.5 * level` (= 5000). Numerical analysis of the
model behavior shows this bound is satisfied regardless of which decomposition mode is selected:

- **Theta (`theta_seasonal_factor_guard_catches_near_zero`):** Even if Rule 2 is disabled and
  multiplicative decomposition is selected, the normalised seasonal factors for the near-zero
  series lie in [0.497, 1.075]. The maximum multiplicative forecast is `2000 × 1.075 ≈ 2150`,
  well below 5000. The test passes whether the guard fires or not. The actual failure mode for
  unguarded Theta is a **severe under-forecast** at the trough month (`2000 × 0.497 ≈ 994`), not
  an over-forecast blow-up — and neither the upper-bound assertion nor `forecast_min > 0` catches
  that under-forecast.

- **AutoETS (`auto_ets_aicselection_rejects_mult_for_near_zero`):** If AIC selects multiplicative,
  the near-zero-month seasonal factor is ≈ 0.0005 and the other-month factors are ≈ 1.09.
  Maximum forecast ≈ 2180; minimum forecast ≈ 1.0 (> 0). Both assertions pass. If AIC selects
  additive, both assertions also pass. The assertions cannot distinguish the two outcomes.

The stated purpose of these tests is to prove "the guard fires and the model stays at level". But
neither test would **fail** if the guard were removed from the production code. They are
non-falsifiable with respect to their stated hypothesis.

**Fix:** Strengthen the assertions to distinguish guarded (additive) from unguarded (multiplicative)
behavior. For Theta, the meaningful distinction is the trough-month forecast: additive produces
`≈2000 + additive_factor ≈ 2000 ± 200`, while multiplicative produces `≈2000 × 0.497 ≈ 994`.
Add a lower-bound assertion so the test actually fails if multiplicative is (erroneously) selected:

```rust
// Theta test: assert the trough-month forecast is NOT severely depressed
// (additive: ~1800-2200; multiplicative without guard: ~994)
let forecast_min = preds.iter().copied().fold(f64::INFINITY, f64::min);
assert!(
    forecast_min > 0.5 * level,
    "Theta near-zero guard: forecast_min={:.0} < {:.0} (0.5× level). \
     Guard should have selected additive; multiplicative produces trough ≈ 994.",
    forecast_min,
    0.5 * level,
);
```

For AutoETS, the trough-phase forecast under multiplicative would be ≈ 1.0 (at horizon offset 11).
A meaningful assertion:

```rust
// AutoETS: if additive was selected, all predictions should stay well above 0
// Multiplicative would produce forecast ≈ 1.0 at the trough phase
assert!(
    forecast_min > 0.1 * level,
    "AutoETS near-zero: forecast_min={:.0} < {:.0} (0.1× level = 200). \
     AIC protection should reject multiplicative; trough-phase under-forecast detected.",
    forecast_min,
    0.1 * level,
);
```

---

### WR-02: Row 11 (TBATS auto-lambda) contains an incorrect supplemental claim

**File:** `docs/audits/multiplicative-guard-audit.md:52`

**Issue:** Row 11 covers standalone TBATS with `estimate_lambda()` (CoV-based optimization, not
AIC). The rationale closes with: *"AutoTBATS AIC comparison additionally rejects the
log-transform variant."* This is incorrect for row 11: `estimate_lambda()` uses CoV minimization
(variance / mean²), not AIC. AutoTBATS AIC comparison is row 13's mechanism.

More significantly, CoV analysis of the near-zero repro series shows that `lambda = 0` has the
**lowest** CoV (0.03451) compared to `lambda = 1.0` (0.03576). TBATS `estimate_lambda()` would
actually **select lambda = 0** for this series, not reject it. The row 11 PASS verdict is still
correct (TBATS Kalman filter is bounded regardless of lambda — the primary claim in that row), but
the supplemental claim about AIC rejection is factually wrong and applies to a different row.

**Fix:** Remove the sentence about AutoTBATS AIC from row 11 and clarify the actual lambda
selection outcome:

```markdown
| 11 | **TBATS** ... | **pass** | TBATS is a Kalman-like linear state-space filter (no
boosting amplification). `estimate_lambda()` (CoV-based) actually selects λ=0 for this series
(lowest CoV), but the Fourier seasonal states absorb the log-space trough accurately without
residual amplification — bounded prediction results. See row 13 for AutoTBATS AIC protection. |
```

---

### WR-03: Row 14 (BoxCox auto-lambda) rationale contains two quantitative inaccuracies

**File:** `docs/audits/multiplicative-guard-audit.md:55`

**Issue:** The row 14 rationale states *"MLE selects λ≈1 (near-identity), avoiding the log-space
crater entirely."* Two problems:

1. **Wrong lambda value:** MLE (LLF grid search over [-2, 2]) selects λ ≈ 2 for the near-zero
   repro series (LLF = −165.5), not λ ≈ 1 (LLF = −177.1). Lambda = 0 scores −229.8, confirming
   that MLE does avoid log-space; but it avoids it by going to λ ≈ 2, not λ ≈ 1.

2. **"Crater" still exists for any lambda:** `boxcox(1.0, λ) = (1^λ − 1)/λ = 0` for all λ ≠ 0,
   and `ln(1.0) = 0` for λ = 0. The value 1.0 maps to 0.0 in Box-Cox space regardless of which
   lambda is selected. The actual protection against blow-up is that `inv_boxcox(0.0, λ) = 1.0`
   for any lambda (mathematical identity), and there is no boosting amplification in the transform
   pipeline — not that MLE avoids the crater's existence.

The PASS verdict is correct; the rationale explaining it is misleading.

**Fix:** Correct the rationale to:

```markdown
**pass** | `any(x <= 0.0) → Err` blocks zero-series. For all-positive near-zero (1.0 > 0): MLE
selects λ ≈ 2 for the repro series (LLF −165 vs −230 for λ=0). Note: `boxcox(1.0, λ) = 0` for
all λ — the crater exists in transformed space — but `inv_boxcox(0.0, λ) = 1.0` (mathematical
identity), so the back-transform returns the original value without amplification. No boosting
pipeline exists to compound the crater into a blow-up.
```

---

## Info

### IN-01: RESEARCH.md Wave 0 Gaps claims "3 tests"; delivered file has 2

**File:** `.planning/phases/06-multiplicative-guard-bug-class-audit/06-RESEARCH.md:333`

**Issue:** The Wave 0 Gaps checklist says `tests/mult05_guard_audit_assertions.rs — covers
MULT-05 SC4 guard assertions (3 tests)`. The delivered file contains exactly 2 tests (confirmed
by `grep -c "^#\[test\]"`). The discrepancy is benign — the RESEARCH.md proposed code block
already showed 2 tests, and the SUMMARY.md correctly describes "two default-feature integration
tests." This is a stale planning artifact, not a gap in coverage.

**Fix:** Update the Wave 0 Gaps line to `(2 tests)` to eliminate future confusion, or leave as-is
since RESEARCH.md is a planning artifact that is superseded by SUMMARY.md.

---

### IN-02: OptimizedTheta and DynamicTheta guard assertions are undocumented as manual-verify

**File:** `docs/audits/multiplicative-guard-audit.md:47–48` (rows 9 and 10)

**Issue:** Rows 9 and 10 assert PASS verdicts for OptimizedTheta and DynamicTheta based on
"identical guard logic to Theta." The guard sites were verified in source. However, no
guard-assertion test exercises these models with the near-zero repro series — unlike Theta (row 8)
which has `theta_seasonal_factor_guard_catches_near_zero`. The VALIDATION.md requirement 6-01-02
only specifies Theta + AutoETS, so this is within scope as planned, but the audit rows do not
explicitly note that the identical-guard claim is review-backed (not test-backed).

This is a documentation gap: a future maintainer reading row 9 may not realize that only row 8
has empirical evidence; rows 9 and 10 rely on code review of the shared `determine_decomposition`
pattern.

**Fix:** Add a note to rows 9 and 10 in the Rationale column:

```markdown
**pass** | Identical seasonal-factor guard to Theta (`determine_decomposition()` shared pattern).
Verified by source inspection; no dedicated guard-assertion test (covered by structural
equivalence with row 8). |
```

---

_Reviewed: 2026-09-09T07:40:49Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
