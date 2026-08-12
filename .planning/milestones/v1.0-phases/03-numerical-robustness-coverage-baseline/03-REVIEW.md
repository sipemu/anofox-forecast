---
phase: 03-numerical-robustness-coverage-baseline
reviewed: 2026-08-11T19:30:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - src/models/laplace/gpd_tails.rs
  - src/models/laplace/multiscale.rs
  - src/models/smart.rs
  - tests/edge_case_robustness.rs
  - tests/property_robustness.rs
  - scripts/update_coverage.sh
  - .github/workflows/ci.yml
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: fixes_applied
fixed_at: 2026-08-11T20:00:00Z
fixes_report: 03-REVIEW-FIX.md
---

# Phase 03: Code Review Report

**Reviewed:** 2026-08-11T19:30:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 3 added a 61-test edge-case suite, a 6-property proptest suite, `validate_series_complete`
guards in two Laplace models, a `.clamp()` fix in `smart.rs`, a local coverage capture script,
and a CI ratchet gate. The structural additions are sound. Three warnings are actionable before
this phase is relied upon in production CI; two info items note test quality gaps that should
be addressed in Phase 4.

## Warnings

### WR-01: `assert_predict_finite` silently panics if `predict()` returns `Err` after a successful `fit()`

**File:** `tests/edge_case_robustness.rs:43`

**Issue:** The helper calls `pred_result.expect(desc)`, which panics on `Err`. Every
"no-panic" test that calls this helper (approx. 20 tests) contains a hidden panic path:
if any model's `predict()` can return `Err` after a successful `fit()` on extreme inputs
(e.g., extreme-large series that overflow a variance computation at predict time), the test
panics instead of failing cleanly. The test suite's headline claim is "no test may trigger a
panic" but this path is unconditionally live whenever `predict()` fails.

The GARCH test (line 656) also directly uses `.expect()` on `pred_result` with the same
structural risk.

**Fix:** Replace `.expect(desc)` with a `match`/`assert!(... .is_ok())` pattern, or use
`prop_assert!`-style conditional assertion:
```rust
fn assert_predict_finite(model: &dyn Forecaster, desc: &str) {
    let pred_result = model.predict(1);
    assert!(
        pred_result.is_ok(),
        "predict() failed after successful fit for {}: {:?}",
        desc,
        pred_result
    );
    if let Ok(forecast) = pred_result {
        assert!(
            forecast.primary().iter().all(|v| v.is_finite()),
            "non-finite forecast for: {}",
            desc
        );
    }
}
```

---

### WR-02: Coverage gate scope mismatch — `Generate coverage` uses all-crates scope, `Enforce coverage floor` and the committed baseline use `--package anofox-forecast`

**File:** `.github/workflows/ci.yml:158,163`

**Issue:** The "Generate coverage" step (line 158) runs:
```
cargo llvm-cov --all-features --lcov --output-path lcov.info
```
This instruments the entire workspace (including `anofox-bench-harness` and
`anofox-forecast-js`). The `lcov.info` sent to Codecov therefore reflects a different
scope than the committed baseline (`coverage.json`, captured with `--package
anofox-forecast`). The "Enforce coverage floor" step is correctly scoped, but Codecov
will display a percentage computed over the wider workspace, making it impossible to
directly compare the Codecov badge to the committed floor. If the bench-harness crate has
lower coverage, the Codecov badge will read lower than 90.3% even when the gate passes,
creating confusion for contributors.

**Fix:** Add `--package anofox-forecast` to the "Generate coverage" step so both
`lcov.info` and the floor check use the same scope:
```yaml
- name: Generate coverage
  run: cargo llvm-cov --package anofox-forecast --all-features --lcov --output-path lcov.info
```

---

### WR-03: Coverage floor gate is not wired to block releases — `coverage` job absent from publish `needs:`

**File:** `.github/workflows/ci.yml:174`

**Issue:** The `publish` job's `needs:` list is:
```yaml
needs: [test, clippy, fmt, docs, audit, deny, wasm, wasm-test, js-test]
```
The `coverage` job is absent. A release can be published to crates.io even if the
coverage ratchet gate fails (i.e., coverage has regressed below the committed floor). The
CI gate enforces coverage on PRs only if branch protection rules require all checks to
pass, but that is not enforced at the workflow level.

**Fix:** Add `coverage` to the publish `needs:` list:
```yaml
needs: [test, clippy, fmt, docs, audit, deny, wasm, wasm-test, js-test, coverage]
```

---

## Info

### IN-01: `var_n2_no_panic` test is vacuously true (tautological assertion)

**File:** `tests/edge_case_robustness.rs:722`

**Issue:** The assertion `assert!(result.is_err() || result.is_ok(), ...)` is always
true for any `Result`; it cannot fail. The test exercises the code path (which does
provide panic detection via the test harness's unwind boundary) but the assertion itself
contributes zero signal. If the intent is "must not panic," the comment is the only
documentation of that intent.

**Fix:** If the outcome is genuinely non-deterministic, document that explicitly and
remove the dead assertion, or handle both branches with meaningful assertions:
```rust
match VARForecaster::new(1).fit(&ts) {
    Ok(mut model) => {
        // If it fitted, predict must also not panic.
        let _ = model.predict(1);
    }
    Err(_) => {
        // Error is acceptable for n=2 VAR(1)
    }
}
```

---

### IN-02: Shell heredoc in `update_coverage.sh` embeds provenance strings directly into Python source without escaping — potential JSON corruption on unusual CPU model names

**File:** `scripts/update_coverage.sh:94-117`

**Issue:** The heredoc delimiter is unquoted (`<<EOF`), so shell expands `$CPU`,
`$RUSTC`, `$COV_VER`, and `$OS` directly into the Python script body. These values are
embedded inside Python double-quoted string literals. If any variable contains a
double-quote or backslash (e.g., a CPU model string like `AMD EPYC "Rome"` or a rustc
version from a forked toolchain), the generated Python script would have a syntax error
and `coverage.json` would not be written. On most Linux hosts with standard CPU model
names this is safe; the risk is real but low.

**Fix:** Pipe the values as environment variables and read them in Python, or use
`json.dumps()` to escape:
```bash
GIT_SHA="$GIT_SHA" TIMESTAMP="$TIMESTAMP" RUSTC="$RUSTC" \
COV_VER="$COV_VER" OS="$OS" CPU="$CPU" \
LINES_TOTAL="$LINES_TOTAL" LINES_COVERED="$LINES_COVERED" \
LINES_PCT="$LINES_PCT" FLOOR="$FLOOR" \
python3 - <<'EOF'
import json, os
data = {
  "provenance": {
    "git_sha": os.environ["GIT_SHA"],
    ...
  },
  ...
}
print(json.dumps(data, indent=2))
EOF
```
Note the quoted `'EOF'` delimiter to prevent shell expansion, with values supplied via
environment instead.

---

## Findings Not Raised (scope discipline)

- **Double-validation in `GpdTailsForecaster::fit`**: `validate_series_complete(series)?`
  is called at line 453, then `self.inner.fit(series)?` immediately follows at line 454,
  and `LaplaceForecaster::fit` itself calls `validate_series_complete` at line 2847.
  This means for the NaN/Inf path the series is validated twice. This is harmless
  (idempotent), does not change observable behavior, and is consistent with the project's
  defensive validation pattern. Not flagged.

- **`clamp(0.0, 1.0)` semantic equivalence in `smart.rs:199`**: Confirmed. Both
  `.max(0.0).min(1.0)` and `.clamp(0.0, 1.0)` propagate NaN unchanged. `clamp` with
  literal `0.0 < 1.0` bounds never panics. The existing `ss_tot < 1e-9` guard (line 180)
  prevents the only NaN-producing division. Semantically identical. Clean fix.

- **`validate_series_complete` placement in `multiscale.rs` (line 189)**: Confirmed first
  statement in `fit()`, before `series.primary_values().len()` at line 190. Correct.

- **`validate_series_complete` placement in `gpd_tails.rs` (line 453)**: Confirmed first
  statement in `fit()`, before delegation to `self.inner.fit()` at line 454. Correct.

- **Proptest value range for MSTL**: `-1000.0f64..1000.0` generates only finite,
  non-NaN values. The no-NaN output assertion is therefore testing MSTL's internal
  numerical behavior, not input propagation. Meaningful. Not flagged.

- **Pre-existing clippy debt (~21 warnings) and `skaters_m5_full_auto.rs`**: Explicitly
  out of scope per review brief.

---

_Reviewed: 2026-08-11T19:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
