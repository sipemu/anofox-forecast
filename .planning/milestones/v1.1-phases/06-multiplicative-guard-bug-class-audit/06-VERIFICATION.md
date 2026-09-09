---
phase: 06-multiplicative-guard-bug-class-audit
verified: 2026-09-09T10:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 6: Multiplicative-Guard Bug-Class Audit — Verification Report

**Phase Goal:** Every model with an auto-multiplicative/log selection path is inventoried and
audited for the #10/#219 too-loose-guard failure class, fix any offender with a regression test,
and record already-safe models with evidence — a complete, auditable sweep.
**Verified:** 2026-09-09T10:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `docs/audits/multiplicative-guard-audit.md` lists every one of the 19 in-scope model/paths from 06-RESEARCH.md with a pass/fail/N-A verdict and rationale, cross-referencing #10 and #219 (SC1/MULT-05) | ✓ VERIFIED | File exists; 19 rows present (grep -cE returns 23 hits, all 19 table rows confirmed by line-by-line read); #10 and #219 cross-referenced in preamble and Evidence section; verdicts match 06-RESEARCH.md exactly: 13 PASS, 6 N/A, 0 FAIL. |
| 2 | At-risk-looking models (Theta, AutoETS) are exercised with a near-zero-relative-to-level series and use TWO-SIDED bounds (max < 2.5× level AND mean within 0.5× level) — WR-01 fix from 06-REVIEW.md landed (SC2) | ✓ VERIFIED | `tests/mult05_guard_audit_assertions.rs` (commit 782261a) has both assertions per test. Theta test: `forecast_max < 2.5 * level` (line 69) AND `(mean_forecast - level).abs() < 0.5 * level` (line 78). AutoETS test: same two bounds (lines 113, 122) plus `forecast_min > 0.0` (line 131). `cargo test --test mult05_guard_audit_assertions` — 2/2 PASS. |
| 3 | The inventory explicitly records the zero-offender result as "MULT-06 satisfied by evidence" — not silently dropped; no in-scope model was skipped (SC3/MULT-06) | ✓ VERIFIED | Dedicated "MULT-06 Result: Zero New Offenders — Satisfied by Evidence" section in audit doc. Explicitly states "MULT-06 is satisfied by evidence. No additional production code change is required." All 19 in-scope paths covered; none skipped vs. CONTEXT.md scope list. |
| 4 | Already-safe models recorded with evidence AND corrected rationales for TBATS row 11 (CoV not AIC, λ≈0 selected) and BoxCox row 14 (inv_boxcox(0)=1.0, λ≈2 not λ≈1) from WR-02/WR-03 landed (SC4) | ✓ VERIFIED | Row 11: "estimate_lambda() minimises the coefficient of variation… CoV actually favours λ≈0… PASS holds on architecture, not on lambda rejection… AIC-based rejection of the log variant is an AutoTBATS property — row 13 — not standalone TBATS." Row 14: "MLE… selects λ≈2, not λ≈1… boxcox(1.0, λ) = 0 for every λ… inv_boxcox(0) = 1.0 recovers the original near-zero value." Both corrections confirmed in audit doc line 52 and 55. |
| 5 | Public Forecaster API unchanged (zero src/ production edits in phase 6 commits); clippy `--all-targets --all-features -- -D warnings` clean (SC5) | ✓ VERIFIED | `git show 0b9d3a6 d6a1f78 782261a --stat \| grep "src/"` — empty output (no src/ files in any phase 6 commit). `cargo clippy --all-targets --all-features -- -D warnings` exits clean. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/mult05_guard_audit_assertions.rs` | Two-sided guard-assertion tests for Theta and AutoETS | ✓ VERIFIED | Exists; 136 lines; two `#[test]` functions confirmed by grep; both tests pass under `cargo test --test mult05_guard_audit_assertions`. Ungated (no `#![cfg(feature = ...)]`). |
| `docs/audits/multiplicative-guard-audit.md` | 19-row inventory table, MULT-06 result, Evidence section | ✓ VERIFIED | Exists; 19 inventory rows in the table; Verdict Tally section (13 PASS, 6 N/A, 0 FAIL); MULT-06 section; Evidence section naming both guard-assertion tests and the MFLES regression test. |
| `docs/audits/` directory | Directory created | ✓ VERIFIED | Directory present; contains `multiplicative-guard-audit.md`. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/mult05_guard_audit_assertions.rs` | `tests/issue_219_mfles_multiplicative_runaway.rs` (Phase 5 repro shape) | `make_near_zero_series()` mirrors Phase 5 fixture (level ≈ 2000, one 1.0 trough, 2×period+6 length) | ✓ WIRED | Test file doc comment explicitly cross-references the #219 repro shape. Helper function matches the specified series shape exactly (lines 25–38). |
| `docs/audits/multiplicative-guard-audit.md` rows | `tests/mult05_guard_audit_assertions.rs` test names | Evidence section in audit doc names both test functions | ✓ WIRED | Evidence table in audit doc lists `theta_seasonal_factor_guard_catches_near_zero` and `auto_ets_aicselection_rejects_mult_for_near_zero` with "What it proves" column. Test function names match exactly. |
| Inventory verdicts | 06-RESEARCH.md Complete Inventory Table | One-to-one correspondence across all 19 rows | ✓ WIRED | Verdicts verified row by row during read: MFLES pass, AutoETS pass, GlobalAutoETS pass, ETS/HW/SeasonalES/GlobalETS N/A, Theta/OptimizedTheta/DynamicTheta pass, TBATS auto-lambda pass, TBATS fixed-lambda N/A, AutoTBATS pass, BoxCox pass, YeoJohnson pass, Laplace seasonal_mult/lognormal/yj_wrapper pass (distributional-gated), standardize/slow_standardize N/A. Zero deviations. |

---

### Data-Flow Trace (Level 4)

Not applicable. This phase produces documentation and integration test files only; no dynamic data is rendered to a user interface.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Theta near-zero guard-assertion passes | `cargo test --test mult05_guard_audit_assertions theta_seasonal_factor_guard_catches_near_zero` | 1 passed | ✓ PASS |
| AutoETS near-zero guard-assertion passes | `cargo test --test mult05_guard_audit_assertions auto_ets_aicselection_rejects_mult_for_near_zero` | 1 passed | ✓ PASS |
| Phase 5 MFLES regression not broken | `cargo test --test issue_219_mfles_multiplicative_runaway` | 3 passed | ✓ PASS |
| Clippy --all-features clean | `cargo clippy --all-targets --all-features -- -D warnings` | 0 errors | ✓ PASS |

---

### Probe Execution

No phase-declared probes. Step 7c: SKIPPED (doc + test phase, no probe scripts declared).

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MULT-05 | 06-01-PLAN.md | Every model with an auto-multiplicative/log selection path is inventoried and audited for the #10/#219 too-loose-guard failure class | ✓ SATISFIED | 19-row inventory at `docs/audits/multiplicative-guard-audit.md`; all paths from CONTEXT.md in-scope list present with verdicts. |
| MULT-06 | 06-01-PLAN.md | Any additional model exhibiting the failure class is fixed and guarded by a regression test | ✓ SATISFIED | Zero new offenders found; MULT-06 result section explicitly documents satisfied-by-evidence conclusion. No production code change required (correct outcome when sweep finds no failures). |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/mult05_guard_audit_assertions.rs` | — | No TBD/FIXME/XXX markers | — | Clean |
| `docs/audits/multiplicative-guard-audit.md` | — | No placeholder language | — | Clean |

No anti-patterns detected. No stub implementations, no hardcoded empty returns, no unresolved debt markers.

---

### WR-01 Fix Adequacy Assessment

The REVIEW.md finding WR-01 identified that the original one-sided bound (`max < 2.5× level`) was
tautological — it would pass even if the guard were removed, because multiplicative Theta/AutoETS
still produces forecasts well below 5000 for this series. The committed fix (commit 782261a) adds a
two-sided mean-centered bound (`|mean - level| < 0.5× level`) to both tests.

The audit doc's Evidence section correctly characterizes these as "regression guards for the failure
class" rather than "guard-firing probes" — acknowledging that for these architecturally safe models
(no ln→boosting→exp pipeline), the tests catch a future architectural regression rather than
proving the existing guard fires today. This framing is accurate and appropriate.

The `mean_forecast` bound (0.5× level = 1000 tolerance around 2000) is meaningful for catching a
collapse/blow-up: if a future refactor introduced an MFLES-style pipeline into Theta or AutoETS,
mean forecasts would deviate far outside this band. The `forecast_min > 0.0` assertion on AutoETS
adds a third dimension. Together these constitute a genuine (if not ironclad) regression guard.

**Assessment:** WR-01 fix landed correctly. Tests are materially stronger than the original
one-sided formulation. The residual epistemic gap (a tautological scenario remains possible in
theory) is acknowledged by the audit doc itself and does not affect the phase goal, which is an
auditable inventory sweep — not a proof of guard mechanism isolation.

---

### Human Verification Required

None. All success criteria are verifiable programmatically or by document inspection.

---

### Gaps Summary

No gaps. All five must-have truths are verified:

- SC1/MULT-05: 19-row inventory is complete and accurate.
- SC2: Two-sided guard-assertion tests pass (both tests, 2/2).
- SC3/MULT-06: Zero-offender conclusion explicitly documented.
- SC4: WR-02 (TBATS row 11 CoV correction) and WR-03 (BoxCox row 14 λ≈2/inv_boxcox fix) both landed in commit 782261a.
- SC5: Zero src/ changes across all phase 6 commits; clippy --all-features clean.

---

_Verified: 2026-09-09T10:00:00Z_
_Verifier: Claude (gsd-verifier)_
