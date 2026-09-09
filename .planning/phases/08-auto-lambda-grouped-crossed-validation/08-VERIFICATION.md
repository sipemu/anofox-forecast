---
phase: 08-auto-lambda-grouped-crossed-validation
verified: 2026-09-09T13:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 8: Auto-λ & Grouped/Crossed Validation — Verification Report

**Phase Goal:** ERM ships with a Ledoit-Wolf-style auto-λ default matching `MinTraceShrink`
ergonomics (fixed-λ path still available) and is validated end-to-end on a grouped/crossed
hierarchy against a MinTrace/unreconciled baseline with the accuracy before/after committed.
**Verified:** 2026-09-09T13:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `Erm { lambda: Option<f64> }` — `None` routes to Ledoit-Wolf-style auto-λ and produces a coherent reconciled forecast (ERM-04) | ✓ VERIFIED | `src/hierarchy/mod.rs:120-131` variant declared; dispatch at line 599; `erm_reconcile` accepts `Option<f64>`, resolves `None` via `erm_auto_lambda` at line 888; `erm_auto_lambda_basic` test passes (line 3167) |
| 2 | `Erm { lambda: Some(x) }` still applies the Phase-7 finite/non-negative guards (ERM-04) | ✓ VERIFIED | `erm_reconcile` lines 880–887: `Some(lam)` arm re-applies finite/non-negative guard; `erm_negative_lambda_returns_err` and `erm_nan_lambda_returns_err` tests pass |
| 3 | Auto-λ on T < 2 training periods returns a `ForecastError` rather than a silent zero/singular solve (ERM-04) | ✓ VERIFIED | `erm_auto_lambda` line 1688: `t_cap < 2` guard returns `InvalidParameter`; `erm_auto_lambda_t1_returns_err` test passes; live run `cargo test --lib "hierarchy::tests::erm"` shows 13/13 pass |
| 4 | ERM auto-λ reconciled forecasts are coherent across all 9 nodes and all H=5 horizon steps on a grouped/crossed hierarchy (ERM-06, SC4) | ✓ VERIFIED | `erm_grouped_crossed_end_to_end_accuracy` lines 3453–3464: 5 HARD coherence assertions (tol 1e-8) across all nodes × steps; IN-01 incoherence spot-check asserts base forecasts ARE incoherent (line 3445), making the proof meaningful; test passes |
| 5 | A committed before/after RMSSE table compares unreconciled, MinTraceStruct, and ERM auto-λ; numbers are honest, measured, and drift-locked (ERM-06) | ✓ VERIFIED | `docs/audits/erm-grouped-validation-results.md` exists with 3-row table (2.287392 / 2.110507 / 0.921839, −59.7%); drift-lock assertions in test at lines 3482–3496 (±1%); `--nocapture` run confirms exact match; information-asymmetry caveat disclosed (WR-03) |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/hierarchy/mod.rs` | `Erm { lambda: Option<f64> }` variant + `erm_auto_lambda` helper + inline tests | ✓ VERIFIED | Variant at lines 120–131; `erm_auto_lambda` private fn at lines 1687–1783; 3 new tests at lines 3167–3506; 9 Phase-7 call sites migrated to `Some(...)` at lines 2787, 2811, 2838, 2868, 2894, 2922, 2952, 3003, 3154 |
| `docs/audits/erm-grouped-validation-results.md` | Committed 3-row method × RMSSE table with real seeded values | ✓ VERIFIED | File exists; 3-row table present; values (2.287392 / 2.110507 / 0.921839) match live test output exactly; measurement note and information note both present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ReconciliationMethod::Erm { lambda }` | `erm_reconcile(&base_map, horizon, lambda)` | dispatch arm at line 599 | ✓ WIRED | `ReconciliationMethod::Erm { lambda } => self.erm_reconcile(&base_map, horizon, lambda)` — destructures `Option<f64>` unchanged |
| `erm_reconcile` (`lambda: None`) | `erm_auto_lambda(&y_stored, n, t_cap)?` | resolve match at lines 879–889, after `y_stored` is built at lines 874–876 | ✓ WIRED | `None => erm_auto_lambda(&y_stored, n, t_cap)?` — auto-λ computed from the same matrix that Gram regularization uses |
| `erm_auto_lambda` return value | `gram[i * n + i] += resolved_lambda` | `resolved_lambda` binding at line 879; Gram diagonal addition at line 905 | ✓ WIRED | `resolved_lambda` used at the single Gram diagonal site; no path bypasses it |
| `erm_grouped_crossed_end_to_end_accuracy` | `from_summing_matrix` 9-node crossed hierarchy | test builds via `from_summing_matrix` at lines 3392, 3407 | ✓ WIRED | Both MinTraceStruct tree and ERM tree built via `from_summing_matrix`; leaf_ancestors `[[0,1,3],[0,1,4],[0,2,3],[0,2,4]]` verified |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 45 hierarchy tests pass (no regressions) | `cargo test --lib "hierarchy::"` | `45 passed; 0 failed` | ✓ PASS |
| All 13 ERM tests pass (new + Phase-7 migrated) | `cargo test --lib "hierarchy::tests::erm"` | `13 passed; 0 failed` | ✓ PASS |
| Printed RMSSE numbers match committed doc | `cargo test --lib "hierarchy::tests::erm_grouped_crossed" -- --nocapture` | Unreconciled 2.287392 / MinTrace 2.110507 / ERM 0.921839 / λ 0.000197 — exact match | ✓ PASS |
| Clippy `-D warnings` clean | `cargo clippy --all-targets --all-features -- -D warnings` | `Finished dev profile — no errors` | ✓ PASS |

---

### Code Review Resolution (commit 0785c2c)

All 7 findings from 08-REVIEW.md addressed in a single fix commit:

| Finding | Severity | Resolution | Status |
|---------|----------|------------|--------|
| CR-01: `delta < 1e-30` early-return bypasses 1e-6 floor (λ=0 for constant forecasts → singular Gram) | Critical | `diag_ref.max(0.0)` → `diag_ref.max(1e-6)` at line 1726; regression test `erm_auto_lambda_constant_forecasts_returns_floor` added | ✓ FIXED |
| WR-01: λ calibrated against centered Gram `G_c` but applied to uncentered Gram `G` (scale mismatch for large-mean data) | Warning | `erm_auto_lambda` rewrote to use `diag_ref = tr(G)/n` and `delta = ||G - diag_ref·I||_F²` from uncentered Gram; gamma retains centered outer-product deviations for noise estimation; RMSSE updated from 0.827239 → 0.921839 | ✓ FIXED |
| WR-02: Stale doc comment misquotes assertion threshold as ×1.5 (actual is ×1.1) | Warning | Comment at line 3264 updated to ×1.1 | ✓ FIXED |
| WR-03: Comparison is informationally asymmetric (MinTraceStruct uses zero data, ERM uses T=20 history) | Warning | Information note added to `erm-grouped-validation-results.md` disclosing the asymmetry and that a fairer comparison would use MinTraceShrink | ✓ FIXED |
| WR-04: Headline numbers in committed doc not assertion-locked (can drift silently) | Warning | Drift-lock assertions (±1%) added to test at lines 3482–3496 for all three RMSSE values; measurement note added to results doc | ✓ FIXED |
| IN-01: Coherence "proof" does not assert unreconciled base forecasts ARE incoherent | Info | Incoherence spot-check added at lines 3440–3450: asserts `|base_total[0] - sum(base_leaves[0])| > 1e-6` | ✓ FIXED |
| IN-02: Comment misdescribes floor trigger (near-zero only, not near-constant) | Info | Comment at lines 1769–1771 corrected to describe both near-zero and near-constant cases | ✓ FIXED |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ERM-04 | 08-01-PLAN.md | Ledoit-Wolf-style auto-λ default with caller-supplied fixed-λ also supported | ✓ SATISFIED | `Erm { lambda: Option<f64> }` — `None` → `erm_auto_lambda`, `Some(λ)` → Phase-7 guards; all 3 auto-λ tests pass |
| ERM-06 | 08-01-PLAN.md | ERM validated end-to-end on grouped/crossed hierarchy vs MinTrace/unreconciled baseline; accuracy before/after committed | ✓ SATISFIED | `erm_grouped_crossed_end_to_end_accuracy` passes; `docs/audits/erm-grouped-validation-results.md` committed with real numbers, drift-locked |

---

### Anti-Patterns Found

No blockers, warnings, or unreferenced debt markers. No `TBD`, `FIXME`, or `XXX` in modified files. No stubs or placeholder implementations.

The SUMMARY.md reports ERM RMSSE as 0.827239 (pre-review-fix value). This is expected — the SUMMARY was written at commit `bc79526`, and the WR-01 Gram-scale fix in commit `0785c2c` correctly updated the result to 0.921839 in both the committed doc and the test's drift-lock assertions. The discrepancy is in the SUMMARY only; the codebase is internally consistent.

---

### SC5 Backward-Compatibility Check

- All 9 Phase-7 ERM call sites migrated to `Some(...)` — no bare `f64` literals remain; crate compiles and all Phase-7 ERM tests pass under the new signature
- No new crate dependencies added (verified via `git diff HEAD~5..HEAD -- Cargo.toml`: no output)
- `cargo clippy --all-targets --all-features -- -D warnings` clean
- Inline LCG used in tests (no `rand` crate dependency in test code)

---

## Gaps Summary

None. All 5 must-haves verified, all review findings resolved, tests passing, clippy clean, committed results doc matches live output.

---

_Verified: 2026-09-09T13:00:00Z_
_Verifier: Claude (gsd-verifier)_
