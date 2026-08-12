---
phase: 04-prioritized-improvement-backlog-top-value-fixes
fixed_at: 2026-08-12T00:00:00Z
review_path: .planning/phases/04-prioritized-improvement-backlog-top-value-fixes/04-REVIEW.md
iteration: 1
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 4: Code Review Fix Report

**Fixed at:** 2026-08-12
**Source review:** `.planning/phases/04-prioritized-improvement-backlog-top-value-fixes/04-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 2 (CR-01 critical, WR-01 warning; IN-01 cosmetic per user directive)
- Fixed: 2
- Skipped: 0

## Fixed Issues

### CR-01: `GlobalAutoETS::fit` silently returns `Ok(())` on NaN/Inf input

**Files modified:** `src/models/exponential/global_ets.rs`, `tests/global_model_nan_guards.rs`
**Commit:** `9548c27`
**Applied fix:** Added a per-series `!v.is_finite()` guard at the top of `GlobalAutoETS::fit`,
immediately after the `is_empty()` check, before the candidate loop. The guard returns
`Err(ForecastError::InvalidParameter("series {i} contains NaN or Inf values"))`, matching
the V-01 pattern used in `GlobalETS::fit`, `GlobalCroston::fit`, and `GlobalTheta::fit`.

Added 3 regression tests to `tests/global_model_nan_guards.rs`:
- `global_auto_ets_nan_guard` — NaN in second series must return `Err(InvalidParameter)`
- `global_auto_ets_inf_guard` — Inf in first series must return `Err(InvalidParameter)`
- `global_auto_ets_valid_panel_guard_does_not_fire` — valid finite panel must return `Ok`, produce 2 forecasts

**Analogous auto-wrappers checked:** Scanned `GlobalCroston` (no auto-wrapper exists; it
has a single model class with the V-02 guard already in place) and `GlobalTheta` (same — no
auto-selection wrapper; V-03 guard is in the single `GlobalTheta::fit`). No analogous
auto-wrapper with the same swallow-and-continue pattern was found elsewhere in the codebase.

**Verification (worktree):**
- Tier 1: Re-read — guard text present, surrounding code intact
- Tier 2: `cargo test --test global_model_nan_guards --all-features` — 9/9 pass
- `cargo test --all-features -p anofox-forecast --lib` — 3174 passed; 0 failed
- Clippy: `cargo clippy --all-features -p anofox-forecast --tests -- -D warnings` — no errors
  in touched files (`global_ets.rs`, `global_model_nan_guards.rs`); 59 pre-existing errors
  in other test files confirmed pre-existing per REVIEW.md ("59 before, 59 after")

### WR-01: Tracer uses `AutoETS::new()` while per-frequency loop uses `AutoETS::with_period(period)` — undocumented

**Files modified:** `crates/anofox-bench-harness/tests/accuracy.rs`
**Commit:** `39e6cb0`
**Applied fix:** Added a 3-line inline comment at the `AutoETS::new()` call (~line 433)
explaining the deliberate omission of a seasonal period:

```rust
// AutoETS::new() (no period) is intentional here — the tracer checks pipeline integrity
// only (finite positive MASE), not seasonal accuracy. The per-frequency accuracy loop
// (above) uses AutoETS::with_period(period) for correct seasonal model selection.
```

This documents the intentional divergence at the point of decision, preventing a future
maintainer from cargo-culting `with_period` without understanding the tracer's narrower intent.

**Verification (worktree):**
- Tier 1: Re-read — comment text present, surrounding code intact
- `cargo fmt -p anofox-bench-harness` — no reformatting of changed lines

## Skipped Issues

None — both in-scope findings were fixed.

---

## IN-01 (Cosmetic — out of scope)

Per user directive: "IN-01 is cosmetic — skip." The predicate form divergence
(`!v.is_finite()` vs `v.is_nan() || v.is_infinite()`) is semantically equivalent and
requires no action.

---

_Fixed: 2026-08-12_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
_Verification: Run in isolated git worktree `gsd-reviewfix/04-434068`; changes fast-forwarded to `docs/scoring-window-attribution`._
