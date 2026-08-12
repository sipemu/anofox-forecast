---
phase: 02-accuracy-harness-statistical-methodology
reviewed: 2026-08-11T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - src/utils/metrics.rs
  - crates/anofox-bench-harness/src/loader.rs
  - crates/anofox-bench-harness/src/naive2.rs
  - crates/anofox-bench-harness/src/dm_test.rs
  - crates/anofox-bench-harness/src/lib.rs
  - crates/anofox-bench-harness/tests/accuracy.rs
  - crates/anofox-bench-harness/tests/cross_library.rs
  - validation/run_statsforecast.py
  - .github/workflows/accuracy.yml
findings:
  critical: 1
  warning: 4
  info: 4
  total: 9
status: partial_fix
fixed:
  - CR-01  # emit_accuracy_json NaN panic — Option<f64> for rmse/mae (commit 1302b3d)
  - WR-01  # MASE filter asymmetry — mase > 0.0 guard added (commit 044db8a)
  - WR-03  # Bartlett label corrected to 95% confidence (commit 9c7a6ad)
  - WR-04  # D-03 comment corrected to reference test slice (commit c7685dd)
open:
  - WR-02  # silent MSIS/MASE count discrepancy — deferred
  - IN-01  # Python skip threshold differs from Rust — no M3 impact, deferred
  - IN-02  # test name mislabels guard as noise test — deferred
  - IN-03  # temp file not cleaned on panic — deferred
  - IN-04  # accuracy.yml non-existent DATASET_DIR — deferred
fixed_at: 2026-08-11T00:00:00Z
---

# Phase 02: Code Review Report

**Reviewed:** 2026-08-11T00:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 02 delivers an M3 accuracy harness (TSF loader, Naive2 baseline, per-frequency
stratification, DM/HLN significance gate, ACCUR-08 anchor, and accuracy.yml CI workflow).
The core statistical formulas — HAC variance, HLN correction, normal CDF approximation,
biased ACF, MASE D-03 denominator guard, and mase_scale training denominator — are all
implemented correctly. The workflow is correctly scoped to workflow_dispatch only with
least-privilege permissions. The temporal integrity assertion fires before every model fit.

One blocker was found: `emit_accuracy_json` is permanently broken because serde_json
cannot serialize `f64::NAN` and `.expect()` will panic whenever the write path is invoked.
Four warnings address correctness risks that could produce incorrect aggregated numbers
or mislead future maintainers working from the in-code comments.

## Structural Findings (fallow)

No structural pre-pass was provided for this review.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: `emit_accuracy_json` panics unconditionally due to `f64::NAN` in `ModelMetrics` for Naive2

**File:** `crates/anofox-bench-harness/tests/accuracy.rs:919-939`

**Issue:** `emit_accuracy_json` constructs a `ModelMetrics` for Naive2 with
`rmse: f64::NAN` and `mae: f64::NAN` (lines 919-920). The struct is then passed to
`serde_json::to_string_pretty(&doc)` at line 939, which returns `Err` for any `f64` field
containing NaN (serde_json has no built-in NaN-to-null or NaN-to-string conversion for
`#[derive(Serialize)]` on `f64` fields). The `.expect("accuracy.json serialization failed")`
then panics.

`emit_accuracy_baseline_if_write_flag_set` is the only caller of `emit_accuracy_json`,
so it too panics whenever `ANOFOX_WRITE_ACCURACY_BASELINE=1` is set. The accuracy.json
baseline lock procedure documented in the Plan 04 SUMMARY is therefore non-functional as
written.

**Fix:** Either store `Option<f64>` for Naive2 RMSE and MAE (marking them as
`None` when not computed and using `#[serde(skip_serializing_if = "Option::is_none")]`),
or use a custom serializer that maps NaN to JSON null. The cleanest fix aligns with the
existing pattern already used for MSIS and coverage:

```rust
// In ModelMetrics:
#[serde(skip_serializing_if = "Option::is_none")]
rmse: Option<f64>,
#[serde(skip_serializing_if = "Option::is_none")]
mae: Option<f64>,

// In make_entry, Naive2 branch:
naive2: ModelMetrics {
    mase: r.naive2_mase,
    smape: r.naive2_smape,
    rmse: None,   // not collected for Naive2 baseline
    mae: None,
    msis: None,
    coverage: None,
},
```

## Warnings

### WR-01: MASE filter asymmetry between `cross_library.rs` and `accuracy.rs` creates an incomparable ACCUR-08 anchor

**File:** `crates/anofox-bench-harness/tests/cross_library.rs:177` and `crates/anofox-bench-harness/tests/accuracy.rs:240`

**Issue:** `run_anofox_mase` in `cross_library.rs` applies a strict `mase > 0.0` filter
before including a series in the mean (line 177). The Python `generate_m3_reference` uses
the same filter (`mase_val > 0`, run_statsforecast.py line 407). However,
`run_accuracy_harness` in `accuracy.rs` pushes MASE unconditionally at line 240
(before `nanmean`), and `nanmean` only filters non-finite values — zero MASE values
(perfect forecasts, however rare) are kept.

The ACCUR-08 anchor test at line 684 compares `accuracy.rs`'s mean (includes zero MASE)
against the Python fixture (excludes zero MASE). If any M3 series has a perfect AutoETS
forecast, the two means are computed over numerically different sets. Additionally, the
MASE numbers produced by `bench01_cross_library` (which uses `cross_library.rs`) and
`accur08_anchor_m3_monthly_autoets` (which uses `accuracy.rs`) may differ for the same
reason, even though both purport to measure the same thing.

**Fix:** Align the filter. Either add `mase > 0.0` to `accuracy.rs`'s accumulation
(to match the Python reference and `cross_library.rs`), or drop the `> 0.0` from both
`cross_library.rs` and the Python fixture. Prefer the former, since a perfect MASE=0 is
a valid result and silent exclusion is harder to spot:

```rust
// accuracy.rs — replace unconditional push with a guard:
let mase_val = ae_fmae / ae_denom;
if mase_val.is_finite() {
    autoets_mases.push(mase_val);
}
// (Same change for naive2_mases at line 212)
```

Then update run_statsforecast.py and cross_library.rs to drop the `> 0` filter, or
document the chosen convention explicitly in all three places.

### WR-02: Silent series count discrepancy between MASE and MSIS aggregates when `predict_with_intervals` fails

**File:** `crates/anofox-bench-harness/tests/accuracy.rs:251-269`

**Issue:** Inside the monthly interval block (lines 251-270), when
`autoets.predict_with_intervals` returns `Err`, the code executes `continue` (line 256),
skipping to the next series. At that point, the series' MASE, sMAPE, RMSE, and MAE have
**already been pushed** to their respective accumulator vectors (lines 240-243). Only
`autoets_msis_vals` and `autoets_cov_vals` do not receive an entry.

Consequence: if `k` series fail interval prediction out of `n` total series:
- `autoets_mases` has `n` entries; `autoets_msis_vals` has `n - k` entries.
- `skipped_nonfinite` counts only NaN entries in the MASE vectors (none for this failure
  mode), so it reports `0`.
- `FrequencyResult.n_series` is the count of all series in the frequency, not the count
  used for MSIS.
- A reader looking at `n_series=1428, skipped_nonfinite=0` would assume MSIS was computed
  over all 1428 series, but it might be over 1420.

No warning is emitted for failed interval calls; the failure is swallowed silently.

**Fix:** Track interval failures separately, emit a warning, and either move the MASE push
below the interval block (so failures skip MASE too, giving a consistent denominator) or
track a separate `skipped_intervals` counter in `FrequencyResult`:

```rust
// Option A: push MASE only after intervals succeed (consistent sets)
// Move lines 240-243 inside the 'if is_monthly { ... }' success block

// Option B: count and warn
let mut skipped_intervals: usize = 0;
// ...inside the Err branch:
Err(_) => { skipped_intervals += 1; continue; }
// ...then log:
if skipped_intervals > 0 {
    eprintln!("WARNING: {} monthly series had interval failures (MSIS/cov excluded)", skipped_intervals);
}
```

### WR-03: Bartlett threshold documented as "90%-confidence" but the value 1.645 corresponds to 95% one-sided CI

**File:** `crates/anofox-bench-harness/src/naive2.rs:6-9` (module doc) and `line 54-55` (inline comment)

**Issue:** The module doc states:
> "90%-confidence autocorrelation test at the seasonal lag (D-08, ACCUR-06)"
> "ACF threshold is `1.645 / sqrt(n)` (one-sided 90% CI under the Bartlett asymptotic
> normal approximation, matching the statsforecast/M4 Naive2 canonical form)"

The value 1.645 is the 95th percentile of the standard normal, not the 90th (which is
1.282). The code uses `acf.abs() > critical`, making this a two-sided test at 5%
significance (or equivalently, a 95% confidence interval for each tail).

The **value 1.645 is correct** and matches the statsforecast/M4 Naive2 implementation.
The label "90%-confidence / one-sided 90% CI" is incorrect and will mislead maintainers
auditing the statistical assumptions or trying to replicate the threshold.

**Fix:** Update the docstring to use the accurate label:

```rust
//! The ACF threshold is `1.645 / sqrt(n)` (Bartlett 95% confidence band,
//! |ACF| > threshold ↔ significant at the 5% level two-sided — matching
//! the statsforecast/M4 Naive2 canonical form — see Assumption A1 in 02-RESEARCH.md).
```

And at line 55:

```rust
// Bartlett 95% confidence band (A1/D-08): |ACF| > 1.645/sqrt(n) is significant
// at the 5% two-sided level (statsforecast Naive2 canonical threshold).
let critical = 1.645 / (n as f64).sqrt();
```

### WR-04: D-03 comment in `calculate_mase` says "constant training window" but `actual` is the test slice

**File:** `src/utils/metrics.rs:156-159`

**Issue:** The D-04 guard comment reads:
> "Denominator-collapse guard (D-04): when seasonal naive MAE is zero
> (constant training window at the seasonal lag), substitute a period-1
> naive denominator rather than dropping the series."

`calculate_mase` receives `actual` (the test/holdout slice) and `predicted`. The naive
denominator at line 148-154 is computed from `actual` — the test slice — not from any
training data. The phrase "constant training window" is factually wrong in this context.

The fix is semantically consistent internally (the denominator and numerator both use the
test slice throughout), but the comment embeds a false mental model. A maintainer reading
this comment while investigating a MASE regression may be led to inspect training data
that has no bearing on this code path.

Note: the harness correctly bypasses `calculate_mase` entirely for competition MASE,
using `mase_scale(train, period)` instead. The issue is confined to the internal
`calculate_mase` / `ForecastMetrics::compute` path and its comment.

**Fix:** Correct the comment to describe what the code actually does:

```rust
// Denominator-collapse guard (D-03): when the seasonal naive MAE is zero
// (i.e. the actual test values repeat exactly at the seasonal lag — a
// degenerate test window), substitute a period-1 first-difference MAE
// computed from the same `actual` slice.  Returns None only if that too
// is zero (truly constant test window with no variation at any scale).
```

## Info

### IN-01: Python series skip threshold differs from Rust harness (no impact on M3, risk on other datasets)

**File:** `validation/run_statsforecast.py:376-387` vs `crates/anofox-bench-harness/tests/accuracy.rs:159`

**Issue:** The Python reference generator skips series where `n <= horizon` (line 376),
then additionally where `len(train) < 2` (line 386), giving an effective skip threshold
of `n <= horizon + 1`. The Rust harness skips where `n <= horizon + period + 2` (line
159). For M3 monthly (horizon=18, period=12), the Rust threshold is `n <= 32`, Python's
is `n <= 19`. Series with `n` in `[20..32]` would be included in the Python fixture
but excluded from the Rust harness.

No M3 monthly series falls in this range (M3 monthly minimum length is ~68), so the
computed MASE values are identical on the actual data. The misalignment is a latent
correctness risk if the harness is extended to other datasets (e.g., NN5 daily with
short series).

**Fix:** Align the skip threshold in the Python fixture generator:

```python
# Match Rust: n <= horizon + period + 2
if n <= horizon + period + 2:
    n_series_skipped += 1
    continue
```

(The `len(train) < 2` check below can then be removed as redundant.)

### IN-02: Test `acf_at_lag_noise_below_threshold` name does not match behavior — tests guard, not noise

**File:** `crates/anofox-bench-harness/src/naive2.rs:227-239`

**Issue:** The test is named `acf_at_lag_noise_below_threshold` and its comment describes
testing "A series with no seasonal structure." However, the test body uses a 4-element
series (`n=4`) with `lag=4`, which triggers the `n <= lag` guard returning 0.0 — a
hardcoded guard, not a computed ACF below threshold. A short linear ramp with `n=12` and
`lag=4` would have a positive (not below-threshold) ACF, as the comment itself
acknowledges.

There is no test that genuinely verifies a non-seasonal series produces `|ACF| <
1.645/sqrt(n)` from the actual formula. The guard test is useful but should not be
labelled as a "noise below threshold" behavioral test.

**Fix (preferred):** Add a genuine sub-threshold test and rename the guard test:

```rust
#[test]
fn acf_at_lag_n_le_lag_returns_zero() {
    // Guard: n == lag → 0.0 by definition (no data beyond lag).
    let series = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(acf_at_lag(&series, 4), 0.0);
}

#[test]
fn acf_at_lag_white_noise_below_threshold() {
    // A series that is orthogonal at lag=4: alternating [+1, -1, +1, -1, ...]
    // has ACF(4) = +1, which is above threshold -- so use a deliberately
    // constructed zero-ACF vector, or verify the guard is not the only coverage.
    // (Document that this test case is hard to construct without a known-zero ACF
    // series; the guard path above provides coverage of the zero-return branch.)
}
```

### IN-03: Temp file in loader tests not cleaned on test panic

**File:** `crates/anofox-bench-harness/src/loader.rs:171-173`

**Issue:** `write_temp_tsf` creates a file in `std::env::temp_dir()` with a
`subsec_nanos`-based suffix. The test that uses it (`parse_tsf_extracts_metadata_and_drops_nan`)
calls `std::fs::remove_file(&path).ok()` at line 193, but this line is only reached if no
earlier assertion panics. If the test fails mid-assertion, the temp file persists until OS
cleanup.

This is a minor hygiene issue (not a data leak risk since the file contains only inline
fixture content), but can accumulate stale files on a developer machine that runs failing
tests repeatedly.

**Fix:** Use a test-scoped RAII guard, or wrap the test body in a closure that cleans up
on drop. The simplest change is to move cleanup before assertions:

```rust
let result = parse_tsf_with_meta(&path);
std::fs::remove_file(&path).ok(); // clean up before any assertion can fire
let (freq, horizon, series) = result.expect("parse should succeed");
// ... rest of assertions
```

### IN-04: `accuracy.yml` sets `ANOFOX_DATASET_DIR` to a non-existent path, triggering a `load_m3` error instead of the clean ACCUR-01 env-unset path

**File:** `.github/workflows/accuracy.yml:62`

**Issue:** The workflow sets:
```yaml
ANOFOX_DATASET_DIR: ${{ github.workspace }}/validation/data
```
Since the M3 corpus is not committed to the repository, this directory does not exist in
CI. The harness then calls `dataset_dir_from_env()` which returns `Some(path)` (the env
var IS set), then `load_m3(&dir)` which calls `std::fs::canonicalize(dir)` which returns
`Err` for a non-existent directory. The error path in `run_accuracy_harness` emits a
`WARNING: load_m3 failed: ... — returning empty harness` to stderr before returning.

Tests still pass (exit 0) because the empty-map early return works, but:
1. The stderr WARNING may appear as a noise signal in CI logs for contributors who have
   not set up the corpus.
2. The clean ACCUR-01 skip message ("ANOFOX_DATASET_DIR not set — skipping ...") is
   never shown; instead a different warning appears.

**Fix:** Either leave `ANOFOX_DATASET_DIR` unset in the CI workflow (so the clean env-gate
path fires), or add a `[ -d "$ANOFOX_DATASET_DIR" ] || unset ANOFOX_DATASET_DIR` check
in the run step:

```yaml
- name: Run accuracy tests
  run: |
    # If dataset corpus is absent, clear the var so tests skip cleanly (ACCUR-01).
    [ -d "$ANOFOX_DATASET_DIR" ] || unset ANOFOX_DATASET_DIR
    if [ "${{ inputs.verbose }}" = "true" ]; then
      cargo test -p anofox-bench-harness --test accuracy -- --nocapture
    else
      cargo test -p anofox-bench-harness --test accuracy
    fi
  env:
    ANOFOX_DATASET_DIR: ${{ github.workspace }}/validation/data
```

---

_Reviewed: 2026-08-11T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
