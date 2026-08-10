---
phase: 02-accuracy-harness-statistical-methodology
fixed_at: 2026-08-11T00:00:00Z
review_path: .planning/phases/02-accuracy-harness-statistical-methodology/02-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 4
skipped: 1
status: partial
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-08-11T00:00:00Z
**Source review:** `.planning/phases/02-accuracy-harness-statistical-methodology/02-REVIEW.md`
**Iteration:** 1
**Fix scope:** critical + warning (CR-*, WR-* only; IN-* excluded per instructions)

**Summary:**
- Findings in scope: 5 (CR-01, WR-01, WR-02, WR-03, WR-04)
- Fixed: 4 (CR-01, WR-01, WR-03, WR-04)
- Skipped: 1 (WR-02 — explicitly excluded by user instruction)

## Fixed Issues

### CR-01: emit_accuracy_json panics on NaN fields

**Files modified:** `crates/anofox-bench-harness/tests/accuracy.rs`
**Commit:** `1302b3d`
**Applied fix:** Changed `ModelMetrics.rmse` and `.mae` from `f64` to `Option<f64>` with
`#[serde(skip_serializing_if = "Option::is_none")]`, matching the existing pattern for
`msis` and `coverage`. In `make_entry`, AutoETS maps finite values to `Some(v)` and
non-finite (NaN/Inf) to `None`; Naive2 uses `None` for both (metrics not collected).
`f64::NAN` literals removed from the Naive2 branch. `serde_json::to_string_pretty` no
longer receives any NaN fields and will not return Err.

**Verification:** `cargo check -p anofox-bench-harness --tests` — clean compile.
`ANOFOX_WRITE_ACCURACY_BASELINE=1 cargo test -p anofox-bench-harness --test accuracy -- emit_accuracy_baseline_if_write_flag_set --nocapture` — test passes; without
`ANOFOX_DATASET_DIR` it skips cleanly instead of panicking.
Verification ran in the main checkout (worktree had no node_modules; `cargo check` uses
the workspace Cargo.lock which is present in the worktree).

---

### WR-01: MASE filter asymmetry between accuracy.rs and fixture/cross_library.rs

**Files modified:** `crates/anofox-bench-harness/tests/accuracy.rs`
**Commit:** `044db8a`
**Applied fix:** Added `mase_val.is_finite() && mase_val > 0.0` guards before pushing to
`autoets_mases` and `naive2_mases` accumulators (lines ~212 and ~240 in the original).
Zero-MASE series (degenerate perfect-fit / zero-denominator artifacts) are now excluded
from the aggregate, matching `cross_library.rs` (line 177: `mase > 0.0`) and the Python
fixture (`run_statsforecast.py`: `mase_val > 0`). Convention documented in inline comment
on both push sites. The `nanmean` function and `skipped_nonfinite` counter are unchanged;
zero-MASE series silently excluded pre-push (consistent with the cross_library convention).

**Verification:** `cargo check -p anofox-bench-harness --tests` — clean compile.
All 22 bench-harness tests pass (`cargo test -p anofox-bench-harness`).

---

### WR-03: Bartlett threshold mislabelled as 90% confidence

**Files modified:** `crates/anofox-bench-harness/src/naive2.rs`
**Commit:** `9c7a6ad`
**Applied fix:** Corrected three locations in the file:
1. Module-level `//!` doc: replaced "90%-confidence autocorrelation test" with
   "Bartlett 95% confidence autocorrelation test" and rewrote the ACF threshold
   description to "Bartlett 95% confidence band, |ACF| > threshold ↔ significant at
   the 5% level two-sided".
2. `Naive2` struct doc comment: replaced "(90%-confidence Bartlett test)" with
   "(Bartlett 95% confidence band, 5% two-sided)".
3. Inline comment on the `critical` variable: replaced "90%-confidence Bartlett critical
   value" with "Bartlett 95% confidence band … significant at the 5% two-sided level".
Numeric constant `1.645` unchanged.

**Verification:** `cargo check -p anofox-bench-harness --tests` — clean compile (doc
changes only; no semantic change possible). All bench-harness tests pass.

---

### WR-04: D-03 comment references wrong slice in calculate_mase

**Files modified:** `src/utils/metrics.rs`
**Commit:** `c7685dd`
**Applied fix:** Replaced the misleading "constant training window at the seasonal lag"
comment on the denominator-collapse guard with an accurate description referencing the
`actual` test slice: "degenerate test window" where "actual test values repeat exactly
at the seasonal lag". Also corrected "D-04" to "D-03" in the comment tag (the guard
is documented as D-03 throughout the codebase). Guard logic and numeric behavior
unchanged.

**Verification:** `cargo check -p anofox-bench-harness --tests` — clean compile.
67 metrics unit tests pass (`cargo test -p anofox-forecast --lib -- utils::metrics`).

---

## Skipped Issues

### WR-02: Silent series count discrepancy between MASE and MSIS aggregates

**File:** `crates/anofox-bench-harness/tests/accuracy.rs:251-269`
**Reason:** Explicitly excluded by user instruction ("Fix ONLY the four findings below —
do NOT touch WR-02"). This finding requires structural changes to the monthly interval
loop (either moving MASE pushes after interval success or adding a separate
`skipped_intervals` counter and warning) and was deferred by the requester.
**Original issue:** When `predict_with_intervals` fails on a monthly series, MASE/sMAPE
are already pushed but the series is skipped for MSIS/coverage, causing a denominator
mismatch between metric aggregates with no visible warning.

---

_Fixed: 2026-08-11T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
_Verification: ran in main checkout (worktree teardown complete). `cargo test -p anofox-bench-harness`: 22 passed, 0 failed. `cargo test -p anofox-forecast --lib -- utils::metrics`: 67 passed, 0 failed._
