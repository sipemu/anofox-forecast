---
phase: 04-prioritized-improvement-backlog-top-value-fixes
reviewed: 2026-08-11T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/models/exponential/global_ets.rs
  - src/models/intermittent/global_croston.rs
  - src/models/theta/global_theta.rs
  - tests/global_theta_smoke.rs
  - tests/global_model_nan_guards.rs
  - crates/anofox-bench-harness/tests/accuracy.rs
findings:
  critical: 1
  warning: 1
  info: 1
  total: 3
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-08-11
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Six files were reviewed covering three NaN/Inf guard insertions (V-01/V-02/V-03) in
GlobalETS, GlobalCroston, and GlobalTheta; two new test files (11 total tests); and
the one-line `AutoETS::with_period(period)` change in the accuracy harness.

The three guards are correctly implemented — right predicate (`!v.is_finite()`), right
error variant (`ForecastError::InvalidParameter`), right placement, no false positives on
valid data. The harness period change is correct for all three frequencies. The test files
are rigorous: all error-path tests assert the exact error variant with `matches!()` and
use `expect_err()` / `expect()` on the happy path.

One critical defect was introduced by the Phase 4 scope boundary: the V-01 guard was
correctly added to `GlobalETS::fit`, but `GlobalAutoETS::fit` — in the same file, also
modified in this phase — calls `GlobalETS::fit` in a loop and silently discards all
`Err` returns. When NaN input is given to `GlobalAutoETS`, every inner `GlobalETS::fit`
call returns `Err` (guard fires), the loop swallows all of them with `continue`, and
`GlobalAutoETS::fit` returns `Ok(())` with poisoned zero-state. Subsequent `predict()`
calls on the poisoned model emit zero forecasts with no error signal.

---

## Critical Issues

### CR-01: `GlobalAutoETS::fit` silently returns `Ok(())` on NaN/Inf input

**File:** `src/models/exponential/global_ets.rs:637-704`

**Issue:** `GlobalAutoETS::fit` iterates over candidate ETS specs and calls
`GlobalETS::new(*spec, period).fit(all_series)` for each one (line 674). The V-01 guard
added in this phase causes `GlobalETS::fit` to return `Err(InvalidParameter(...))` when
the input contains NaN or Inf. `GlobalAutoETS::fit` handles this with `.is_err() →
continue` (line 674). When ALL candidates fail — which is the case for any NaN-containing
panel — the per-series `best_nll` vec stays at `f64::MAX`, `best_spec` stays at the
`ETSSpec::ann()` default, `best_states` stays at zeroed `SeriesState` values, and line 702
unconditionally sets `self.fitted = true` before returning `Ok(())`.

A subsequent `predict()` call returns zero forecasts for all series with no indication
that fitting failed. The `has_non_positive` check at line 650 does NOT catch NaN because
`NaN <= 0.0` evaluates to `false` in IEEE 754.

This is a direct consequence of Phase 4 adding the V-01 guard to `GlobalETS::fit` without
adding an equivalent guard to `GlobalAutoETS::fit`. Both types are in the same file that
was modified in this phase.

**Fix:** Add the same NaN guard at the top of `GlobalAutoETS::fit`, immediately after
the `is_empty()` check (line 644), matching the V-01 pattern:

```rust
// In GlobalAutoETS::fit, after the is_empty() guard:
for (i, s) in all_series.iter().enumerate() {
    if s.iter().any(|v| !v.is_finite()) {
        return Err(ForecastError::InvalidParameter(format!(
            "series {} contains NaN or Inf values",
            i
        )));
    }
}
```

A test mirroring `global_ets_nan_guard` should be added for `GlobalAutoETS`.

---

## Warnings

### WR-01: Tracer test uses `AutoETS::new()` while per-frequency loop uses `AutoETS::with_period(period)` — divergence is undocumented in code

**File:** `crates/anofox-bench-harness/tests/accuracy.rs:433`

**Issue:** The per-frequency harness at line 228 correctly uses `AutoETS::with_period(period)`,
including `period=12` for monthly. The tracer test at line 433 uses `AutoETS::new()` which
defaults to `seasonal_period = None`, suppressing seasonal candidates for the monthly series
it tests. The tracer comment at line 456-463 explains that it "proves a finite positive MASE"
only, not a tight accuracy target — so the period inconsistency is intentionally acceptable
for the tracer's purpose.

However, this divergence is not documented in the tracer's inline comments at the point
of the constructor call. A future maintainer editing the tracer to improve accuracy coverage
might not realize the period omission is deliberate, and might copy the `with_period` call
from the per-frequency loop without understanding the tracer's narrower intent.

**Fix:** Add a one-line comment on line 433:

```rust
// AutoETS::new() (no period) is intentional here — the tracer checks pipeline integrity
// only (finite positive MASE), not seasonal accuracy. See per-frequency loop for period=12.
let mut model = AutoETS::new();
```

---

## Info

### IN-01: Predicate form diverges from canonical VAR pattern (`!v.is_finite()` vs `v.is_nan() || v.is_infinite()`)

**File:** `src/models/exponential/global_ets.rs:105`, `src/models/intermittent/global_croston.rs:87`, `src/models/theta/global_theta.rs:77`

**Issue:** The three Phase 4 guards use `!v.is_finite()` (clippy-preferred shorthand). The
authoritative pattern in `src/models/var.rs:119` uses `v.is_nan() || v.is_infinite()`.
The two forms are semantically equivalent in IEEE 754 — there is no correctness difference.
The 04-01 SUMMARY documents this deliberate choice ("clippy-preferred, equivalent to
`is_nan() || is_infinite()`"). The inconsistency is harmless but creates two canonical
forms for the same guard in the codebase.

**Fix:** No immediate action required. If the codebase ever consolidates on one form,
prefer `!v.is_finite()` (clippy prefers it). If VAR is also updated to `!v.is_finite()`
at that point, the codebase will be consistent.

---

## Findings Not Raised (Scope Discipline)

The following were observed but are explicitly pre-existing and out of scope per the review mandate:

- 59 clippy `--all-targets` errors: pre-existing, not introduced by Phase 4 changes (confirmed by SUMMARY: "59 before, 59 after").
- `examples/skaters_m5_full_auto.rs` compile error: pre-existing non-exhaustive match, not touched by Phase 4.
- `GlobalAutoETS::fit` missing `n_valid` success check after the spec loop (independent of NaN — if all specs fail for non-NaN reasons, e.g. all series too short, `GlobalAutoETS::fit` also returns `Ok()` with poisoned state): this pre-dates Phase 4, but is now partially surfaced by the V-01 guard. The NaN-specific path is the Phase 4 regression; the general "all specs fail" path is pre-existing.
- `skipped_nonfinite` in `FrequencyResult` uses `sk_ae_mase.max(sk_n2_mase)` but smape/rmse/mae are pushed unconditionally (not filtered by the mase condition), creating a count mismatch: pre-existing.

---

_Reviewed: 2026-08-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
