---
phase: 05-mfles-multiplicative-guard-fix
reviewed: 2026-09-09T00:00:00Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - src/models/mfles.rs
  - tests/issue_219_mfles_multiplicative_runaway.rs
  - tests/laplace_component_robustness.rs
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 05: Code Review Report

**Reviewed:** 2026-09-09
**Depth:** deep
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the three commits comprising Phase 5 (MFLES multiplicative-mode guards for issue
#219): the core fix in `src/models/mfles.rs`, the regression test suite, and the
`required-features` gate added to `tests/laplace_component_robustness.rs`.

The fix is logically correct. The three guards (MULT-01 mode selector, MULT-02 log-floor
winsorisation, MULT-03 back-transform clamp) address the root cause of the runaway-forecast
bug. The boolean direction of the mode guard is correct. The `insample_max` lifecycle is
sound: it is set inside the transform block (lines 1056–1071) which runs before the
constant-series early-return path (line 1103), so the early-return inherits the correct
value. The `None` fallback in `predict_internal` (cap = `f64::INFINITY`) is inert and safe.
No new public API surface is introduced. No production logging is present.

Two items require attention before broader deployment.

---

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: New `insample_max` field breaks serde JSON/bincode round-trips for pre-existing serialized MFLES models

**File:** `src/models/mfles.rs:98`

**Issue:** `insample_max: Option<f64>` is a new struct field derived via
`#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` (line 30)
with no `#[serde(default)]` annotation. `serde_json` does not treat `Option<T>` as
implicitly absent-means-None; if the key is missing from the input JSON, deserialization
returns an error rather than `None`. Any MFLES model serialized before this PR that a
caller attempts to round-trip through `persistence::from_json` / `from_bincode` after this
PR will fail.

The field's own doc-comment (line 97) explicitly notes the None-safe intent ("additive-path
serialised model — clamp is then inert") — that intent is not reflected in the wire format.

Note: the other existing `Option<f64>` fields (`const_val`, `mean`, `std`, `penalty`) have
the same pre-existing issue, so this is not a newly introduced *category* of problem, but
this PR adds one more field with the same gap and no test covers MFLES serde round-trips.

**Fix:**
```rust
/// In-sample maximum value (original scale), stored at fit time for the
/// multiplicative back-transform clamp (issue #219).
/// `None` on the additive path (or for old deserialized models) — clamp
/// is then inert (cap = f64::INFINITY).
#[cfg_attr(feature = "serde", serde(default))]
insample_max: Option<f64>,
```

Adding `#[cfg_attr(feature = "serde", serde(default))]` (or just `#[serde(default)]`
inside the `cfg_attr` block) makes deserialization of old JSON blobs succeed and default
to `None`, exactly matching the documented fallback behaviour.

---

### WR-02: Median and min-scan computed twice for the multiplicative auto path

**File:** `src/models/mfles.rs:1042-1043` and `1057-1059`

**Issue:** When auto mode selects multiplicative, `Self::median_scalar(values)` and the
`fold(f64::INFINITY, f64::min)` min-scan are each performed once in the mode guard (lines
1042–1043) and once again in the transform block (lines 1057–1059). Both calls operate on
the same immutable `values` slice and produce identical results. For a 24-point series the
overhead is trivial, but for a caller providing a long series the extra sort (O(n log n))
and scan (O(n)) are wasted. More importantly, the duplication makes it easy for a future
edit to update one computation without updating the other, silently diverging.

**Fix:** Hoist the median and min_val out of the match arm and reuse them in the transform
block:

```rust
// Determine multiplicative mode
let series_median = Self::median_scalar(values);
let series_min = values.iter().copied().fold(f64::INFINITY, f64::min);
let use_multiplicative = match self.multiplicative {
    Some(m) => m,
    None => {
        let all_positive = self.season_length > 0 && values.iter().all(|&v| v > 0.0);
        all_positive
            && series_median > 0.0
            && (series_min / series_median) >= Self::MULT_AUTO_TAU
    }
};
self.is_multiplicative = use_multiplicative;

// Transform data
let y: Vec<f64>;
if use_multiplicative {
    let floor = Self::MULT_LOG_FLOOR_FRAC * series_median;
    self.const_val = Some(series_min);
    self.insample_max = Some(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    y = values.iter().map(|&v| v.max(floor).ln()).collect();
} else {
    // ...
}
```

(Note: `max_val` is not needed in the guard path so a single fold for it is fine inside
the transform block; only the redundant median and min need hoisting.)

---

## Info

### IN-01: Test MULT-01 and MULT-04 are functionally overlapping; MULT-04 is redundant given MULT-01

**File:** `tests/issue_219_mfles_multiplicative_runaway.rs:47-93`

**Issue:** Both `issue_219_mfles_no_multiplicative_runaway` (MULT-04, lines 47–73) and
`issue_219_mfles_auto_mode_selects_additive_for_near_zero_series` (MULT-01, lines 80–93)
fit the same series in auto mode and assert that the forecast is near level ≈ 2000.
MULT-04 checks `forecast_max < 5000`; MULT-01 checks `|mean - 2000| < 1000`. The latter
strictly implies the former (a mean within ±1000 of 2000 also keeps the max below 5000 for
sensible seasonal shapes). Both tests will fail simultaneously pre-fix and pass
simultaneously post-fix — neither provides additional regression coverage the other does not.

This is not a correctness problem. Having two overlapping assertions is a mild over-test
rather than an under-test, and the separate names help document the specific guard they
target. Leaving as-is is acceptable; the note is for future trimming.

**Fix:** No action required. If test count becomes a concern, consolidate into one test that
asserts both `forecast_max < 5000` and `|mean_forecast - 2000| < 1000`.

---

### IN-02: MULT-03 test uses a hardcoded `insample_max_approx` rather than a value derived from the same series literal

**File:** `tests/issue_219_mfles_multiplicative_runaway.rs:115-116`

**Issue:** The explicit-multiplicative test asserts `forecast_max <= 10.0 * 2400.0`. The
value 2400.0 is taken from memory (the reviewer must inspect the `values` literal and
identify the maximum). If the test series is ever modified, this assertion will silently
become incorrect — either too tight (false failures) or too loose (misses regressions).

**Fix:** Derive the cap directly from the test data to make the coupling explicit and
resilient to edits:

```rust
let values_data = vec![
    2100.0_f64, 1950.0, 2200.0, 1800.0, 2050.0, 2300.0, 1900.0,
    2150.0, 1850.0, 2400.0, 2000.0, 1.0,
    2050.0, 1950.0, 2100.0, 1800.0, 2200.0, 1900.0, 2050.0,
    2300.0, 1850.0, 2150.0, 2000.0, 2100.0,
];
let actual_insample_max = values_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
// MULT_BACK_CLAMP_K = 10.0 (matches the production constant)
let cap = 10.0 * actual_insample_max;
```

Alternatively, extract the series literal into a `const` array so it can be iterated
without a heap allocation.

---

_Reviewed: 2026-09-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
