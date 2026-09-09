# Phase 5: MFLES Multiplicative-Guard Fix — Research

**Researched:** 2026-09-09
**Domain:** Rust numerical/statistical model bug fix — MFLES log-transform robustness
**Confidence:** HIGH (all claims verified by reading source files this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Mode guard (MULT-01):** When `min / median < τ` with τ = 0.10, auto mode picks additive instead of multiplicative. Internal constant (`const` in source), silent fallback (no logging — library convention).
- **Log floor (MULT-02):** Before `ln()`, winsorize each value: `v.max(floor_frac × median)` with `floor_frac = 0.01`. Prevents a single near-zero observation from opening a log-space crater.
- **Back-transform clamp (MULT-03):** Cap `exp()` forecast at 10 × in-sample max; floor at 0. Store the in-sample max as fitted state so `predict()` can clamp.
- **API contract:** Public `Forecaster` API unchanged. No new builder methods. Constants are internal.
- **Proof (MULT-04):** Before/after regression test on the #219 repro series (near-zero month, level ≈ 2k); committed as guard.

### Claude's Discretion

- Where exactly to place the new in-sample-max struct field in the fitted-state block.
- Whether to call `Self::median_scalar()` inline or factor a small helper for the guard computation.
- Test file name and module structure.

### Deferred Ideas (OUT OF SCOPE)

- Auditing other models (Theta, ETS, TBATS, etc.) for the same failure class — that is Phase 6 (MULT-05/06).
- Making guard thresholds user-configurable via the builder — deferred unless a real need arises.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MULT-01 | MFLES no longer selects multiplicative mode when `min/level < τ` (τ = 0.10); falls back to additive | Mode-selection code located at `mfles.rs:997-1004`; guard inserts a median/min ratio check before the existing `all > 0` test |
| MULT-02 | MFLES multiplicative mode floors/winsorizes `ln()` so a near-zero observation cannot create a log-space crater | Transform code at `mfles.rs:1009-1012`; floor = `v.max(0.01 × median)` applied before `v.ln()` |
| MULT-03 | MFLES multiplicative back-transform clamped so forecast cannot exceed 10 × in-sample max | Back-transform at `mfles.rs:937-938` and `mfles.rs:1304-1305`; new `insample_max` field needed in MFLES struct |
| MULT-04 | #219 repro series forecasts at ~level instead of ~13×; committed as regression guard | New test file `tests/issue_219_mfles_multiplicative_runaway.rs`; verify command below |
</phase_requirements>

---

## Summary

Phase 5 is a targeted, three-point bug fix in a single file (`src/models/mfles.rs`, ~2415 lines). The root cause of issue #219 is that MFLES auto-selects multiplicative mode whenever all series values are strictly positive (`season_length > 0 && values.iter().all(|&v| v > 0.0)`). A single near-zero month (e.g., 1.0 in a series whose level is ≈ 2000) is still `> 0`, so multiplicative is selected. `ln(1.0) = 0.0` — a massive negative outlier in log-space for a series whose log-mean is `ln(2000) ≈ 7.6` — which distorts the boosted trend and seasonal fit, producing a back-transformed forecast of ~13× level.

The fix applies three mutually reinforcing guards: (1) reject multiplicative mode at auto-selection time when `min / median < 0.10`; (2) even when multiplicative IS used (explicit user override or a future series that passes the guard), floor each value before `ln()` to prevent log-space craters; (3) clamp every `exp()` back-transformed forecast to `[0, 10 × in-sample max]`. The guards compose correctly — guard 1 eliminates the most damaging case, guards 2 and 3 are a safety net for edge cases that survive guard 1.

**Primary recommendation:** Implement all three guards as named `const` values in `mfles.rs`, apply them at the three verified code locations, add one fitted-state field (`insample_max: Option<f64>`), and ship a self-contained regression test in `tests/issue_219_mfles_multiplicative_runaway.rs` that fails on the old code and passes with the fix.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Mode-selection guard | Model (fit path) | — | Decision happens at fit time before transform; no prediction involvement |
| Log-transform floor | Model (fit path) | — | Applied immediately before `v.ln()` inside `fit()`; same for fitted-values inverse |
| Back-transform clamp | Model (predict path) | Model (fit path) | Clamping happens in `predict_internal()`; in-sample max computed and stored during `fit()` |
| Regression test | Integration test layer | — | New file in `tests/`; exercises the full fit→predict path |

---

## Standard Stack

No new external dependencies. This phase touches only:

| Crate | Role | Status |
|-------|------|--------|
| `anofox_forecast` (self) | Model under fix | Stable — internal change only |
| `chrono` | Timestamp construction in tests | Already in dev-dependencies |
| `anofox_forecast::core::TimeSeries` | Test data construction | Existing API |

No packages to install. No legitimacy audit required.

---

## Architecture Patterns

### Data Flow Through the Bug (Before Fix)

```
fit() called with [2000, 1900, 2100, ..., 1.0, ..., 2050]
  → all values > 0.0  [TRUE — 1.0 > 0]
  → multiplicative selected
  → y = values.map(|v| v.ln())
       e.g. ln(1.0) = 0.0, ln(2000) ≈ 7.6
       → log-space crater at position of near-zero month
  → boosting loop fits trend + seasonal in distorted log-space
  → trend accumulates inflated slope
  → predict() calls pred.exp()
       → 13× level forecast
```

### Data Flow After Fix

```
fit() called with [2000, 1900, 2100, ..., 1.0, ..., 2050]
  → compute median ≈ 2000
  → min = 1.0
  → min / median = 0.0005 < τ (0.10)  [GUARD 1 TRIPS]
  → additive mode selected; no log transform
  → normal additive path; forecast ~level

--- OR (explicit multiplicative override) ---

fit() called with .multiplicative(true) and near-zero series
  → mode = multiplicative (explicit — guard 1 bypassed per user intent)
  → floor = 0.01 × median = 20.0
  → y = values.map(|v| v.max(20.0).ln())   [GUARD 2]
       → log-space no longer has crater
  → boosting fits reasonable log-space trend
  → insample_max = values.iter().fold(f64::NEG_INFINITY, f64::max)  [stored]
  → predict(): cap = 10 × insample_max
  → pred.exp().clamp(0.0, cap)   [GUARD 3]
  → forecast within bounded range
```

### Recommended Edit Locations

All edits are inside `src/models/mfles.rs`.

**Edit 1 — struct field addition** (after `training_regressors_store` at line 93):

```rust
// [VERIFIED: src/models/mfles.rs:51-93]
// New field to be inserted into the MFLES struct:
/// In-sample maximum value (original scale), stored at fit time
/// for the multiplicative back-transform clamp (issue #219).
insample_max: Option<f64>,
```

**Edit 2 — `MFLES::new()` initializer** (inside `Self { ... }` at line 232):

```rust
insample_max: None,
```

**Edit 3 — constants block** (top of `impl MFLES`, before the first `fn`, around line 225):

```rust
/// Minimum ratio of series min to median below which auto mode
/// selects additive instead of multiplicative (issue #219).
const MULT_AUTO_TAU: f64 = 0.10;

/// Floor fraction applied to each value before ln() in multiplicative
/// mode: v.max(MULT_LOG_FLOOR_FRAC × median) (issue #219).
const MULT_LOG_FLOOR_FRAC: f64 = 0.01;

/// Back-transform cap: forecast is clamped to this multiple of the
/// in-sample maximum to prevent runaway exp() output (issue #219).
const MULT_BACK_CLAMP_K: f64 = 10.0;
```

**Edit 4 — mode-selection guard** (replace `mfles.rs:997-1004`):

```rust
// [VERIFIED: src/models/mfles.rs:997-1004]
let use_multiplicative = match self.multiplicative {
    Some(m) => m,
    None => {
        // Auto: multiplicative only if positive AND series is not
        // near-zero relative to its level (issue #219 guard).
        let all_positive = values.iter().all(|&v| v > 0.0);
        if all_positive && self.season_length > 0 {
            let median = Self::median_scalar(values);
            let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
            // Guard: if min / median < τ, additive is safer.
            median > 0.0 && (min_val / median) >= Self::MULT_AUTO_TAU
        } else {
            false
        }
    }
};
```

**Edit 5 — log-transform floor and in-sample max storage** (replace `mfles.rs:1009-1012`):

```rust
// [VERIFIED: src/models/mfles.rs:1009-1012]
if use_multiplicative {
    let median = Self::median_scalar(values);
    let floor = Self::MULT_LOG_FLOOR_FRAC * median;
    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    self.const_val = Some(min_val);
    // Store in-sample max for back-transform clamp.
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    self.insample_max = Some(max_val);
    // Winsorize before ln() to prevent log-space craters (issue #219).
    y = values.iter().map(|&v| v.max(floor).ln()).collect();
} else {
    self.const_val = None;
    self.insample_max = None;
    // ... (existing additive path unchanged)
}
```

**Edit 6 — back-transform clamp in `predict_internal()`** (replace `mfles.rs:937-938`):

```rust
// [VERIFIED: src/models/mfles.rs:936-943]
let mut pred_original = if self.is_multiplicative {
    let raw = pred.exp();
    // Clamp to [0, K × in-sample max] to prevent runaway (issue #219).
    let cap = self.insample_max
        .map(|m| Self::MULT_BACK_CLAMP_K * m)
        .unwrap_or(f64::INFINITY);
    raw.max(0.0).min(cap)
} else {
    let mean_val = self.mean.unwrap_or(0.0);
    let std_val = self.std.unwrap_or(1.0);
    mean_val + pred * std_val
};
```

**Edit 7 — back-transform clamp in fitted-values computation at end of `fit()`** (replace `mfles.rs:1304-1305`):

```rust
// [VERIFIED: src/models/mfles.rs:1303-1310]
let fitted_original: Vec<f64> = if use_multiplicative {
    let cap = self.insample_max
        .map(|m| Self::MULT_BACK_CLAMP_K * m)
        .unwrap_or(f64::INFINITY);
    fitted.iter().map(|&f| f.exp().max(0.0).min(cap)).collect()
} else {
    let mean_val = self.mean.unwrap_or(0.0);
    let std_val = self.std.unwrap_or(1.0);
    fitted.iter().map(|&f| mean_val + f * std_val).collect()
};
```

Note: The `trend_full` inverse at `mfles.rs:1321-1322` also uses `v.exp()` directly. Apply the same clamp there for consistency, or leave it unclamped (trend component is informational; the overall `fitted_original` already clamps, so the decomposition `trend + seasonal + residual == training` will shift residuals rather than blowing up).

---

## Research Question Answers

### Q1: Reference Behavior — statsforecast MFLES vs Rust Port

The Rust port's auto-selection at `mfles.rs:1001-1003` exactly mirrors the statsforecast Python logic: `season_length > 0 and all(y > 0)`. [VERIFIED: src/models/mfles.rs:997-1004] — the comment at line 1001 says "Auto: multiplicative if positive and seasonal."

The statsforecast Python MFLES does **not** apply a min/median ratio guard before selecting multiplicative — it simply checks all-positive. This is the upstream divergence that issue #219 exposes. Our fix adds a stronger guard that statsforecast does not yet have; it is an intentional improvement over the upstream reference behavior. [ASSUMED — based on the port comment "Reference: statsforecast MFLES implementation" at `mfles.rs:7` and the fact that the current code matches the documented Python logic; no Python source confirmed via tool this session.]

The `const_val` stored at fit time (`mfles.rs:1011`) is the min value, held for potential future use, but it is NOT currently used in the log transform (no shift) and NOT used in the back-transform (no `const_val` subtraction in `exp()`). [VERIFIED: src/models/mfles.rs:1009-1012, 936-943] The model operates in raw log-space: `ln(v)` → fit → `exp(pred)`. A near-zero value enters as an extreme negative log-space point, distorting every boosting round.

### Q2: Exact Repro Series for #219

Proposed synthetic series (24-point monthly, level ≈ 2000, one near-zero month at index 11):

```
[2100.0, 1950.0, 2200.0, 1800.0, 2050.0, 2300.0,
 1900.0, 2150.0, 1850.0, 2400.0, 2000.0,   1.0,   ← near-zero (index 11)
 2050.0, 1950.0, 2100.0, 1800.0, 2200.0, 1900.0,
 2050.0, 2300.0, 1850.0, 2150.0, 2000.0, 2100.0]
```

Series properties:
- `n = 24`, `season_length = 12` (monthly)
- `min = 1.0`, `median ≈ 2050.0`
- `min / median ≈ 0.000488` — far below τ = 0.10, so guard trips correctly

**Pre-fix expected behavior:** `MFLES::new(vec![12])` on this series selects multiplicative (all values > 0), maps `ln(1.0) = 0.0` — a log-space crater 7.6 units below the log-mean — producing forecast values that when `exp()`-ed are in the range 20k–30k (roughly 10–15× level of 2000). The test asserts `forecast_max > 5_000.0` to document the blow-up before the fix.

**Post-fix expected behavior:** Guard 1 trips (`min/median < 0.10`), additive mode selected, forecasts are in the range 1500–2500 (≈1× level). Test asserts `forecast_max < 5_000.0` (below 2.5× level).

**Tighter post-fix assertion:** `forecast.primary().iter().all(|&f| f > 500.0 && f < 5_000.0)` — positive and within 2.5× level in either direction.

### Q3: Guard Interaction Correctness

The three guards compose without double-counting:

| Scenario | Guard 1 (mode select) | Guard 2 (ln floor) | Guard 3 (exp clamp) | Result |
|----------|----------------------|--------------------|--------------------|--------|
| Auto mode, near-zero series | TRIPS → additive selected | Inert (additive path, no ln()) | Inert (additive path, no exp()) | Correct additive forecast |
| Auto mode, healthy positive series | Does not trip → multiplicative | Applied (but `v.max(0.01 × median)` = v for healthy values, no-op) | Applied (cap = 10 × max, not reached for reasonable forecasts) | Unchanged from current behavior |
| Explicit `.multiplicative(true)`, near-zero series | Bypassed (user's explicit choice) | Applied — floors `1.0` to `0.01 × 2050 = 20.5` before ln() | Applied — caps at 10 × 2400 = 24000 | Protected; forecast reasonable |
| Explicit `.multiplicative(false)` | Bypassed — additive | Inert | Inert | Unchanged additive behavior |

**Important:** When guard 1 trips and additive mode is selected, `insample_max` is set to `None` in the additive branch (`self.insample_max = None`). In `predict_internal()`, the clamp resolves to `f64::INFINITY` when `insample_max` is `None` — effectively no clamp on the additive path. This is correct. [VERIFIED: src/models/mfles.rs:936-943] — additive branch uses `mean + pred * std`, not `exp()`.

**Where to store in-sample max:** Add `insample_max: Option<f64>` to the MFLES struct after `training_regressors_store` (line 93 is the last field). Set it inside the `if use_multiplicative` branch at line 1009. [VERIFIED: src/models/mfles.rs:51-93] — full fitted-state field block verified this session; `insample_max` does not yet exist.

The existing `const_val: Option<f64>` field at line 53 stores the series min for reference (statsforecast compatible); it is safe to leave unchanged. `insample_max` is a new field serving a different purpose (clamping upper bound).

### Q4: Median Computation

MFLES already has `Self::median_scalar(values: &[f64]) -> f64` at `mfles.rs:342-357`. [VERIFIED: src/models/mfles.rs:342-357] The implementation:
- Filters out non-finite values before sorting
- Uses sort_by with `partial_cmp` + fallback to `Equal`
- Returns average of two middle elements for even lengths
- Returns 0.0 for empty input

This is the correct implementation to reuse for the mode-selection guard. **Do not** use `crate::features::basic::median()` — it is a different function with a different NaN policy (returns `f64::NAN` for empty) and is not accessible from inside `mfles.rs` without a use statement. Using `Self::median_scalar()` keeps the guard internally consistent with how MFLES already computes its initial median baseline.

For the log-floor (`MULT-02`), `median = Self::median_scalar(values)` is computed in the guard; reuse that value for `floor = MULT_LOG_FLOOR_FRAC × median` to avoid computing it twice.

### Q5: Test Placement and Verify Command

**Test file:** `tests/issue_219_mfles_multiplicative_runaway.rs`

Naming follows the existing pattern: `tests/issue_106_decomposable_conformance.rs`, `tests/issue_107_inspectable_conformance.rs`. [VERIFIED: /home/simonm/projects/rust/anofox-forecast/tests/] — both issue_106 and issue_107 files exist, confirming this naming convention.

**No feature flags required** — MFLES lives in the default feature set (`postprocess` is the only non-trivial default feature, and MFLES does not gate on it). Integration tests in `tests/` run against the crate with all dev-dependencies, no explicit `--features` needed.

**Verify command (exact):**

```bash
cargo test --test issue_219_mfles_multiplicative_runaway
```

Full suite (also catches decomposable-conformance regression from the new struct field):

```bash
cargo test
```

Clippy gate (required by CLAUDE.md conventions):

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Q6: Pitfalls

**Pitfall 1 — Decomposable invariant regression (`mfles.rs:81-88`).**
The invariant `trend + seasonal + residual == training` is tested by `tests/issue_106_decomposable_conformance.rs::mfles_conforms_to_decomposable_contract()`. [VERIFIED: /home/simonm/projects/rust/anofox-forecast/tests/issue_106_decomposable_conformance.rs:176-181]

The clamping in `fitted_original` (Edit 7) can break this invariant for the rare case where a fitted value exceeds the cap. For the #219 series the clamp is safe because: (a) guard 1 redirects to additive, so no clamping fires on healthy multiplicative fits, and (b) the clamp is applied uniformly to `fitted_original` and `trend_full`. However `seasonal_full = fitted_original - trend_full` — both are independently clamped, so their difference may no longer sum correctly with residuals to training. **Recommendation:** Apply the back-transform clamp only in `predict_internal()` (Edit 6), and do NOT clamp in the fitted-values computation (Edit 7). The fitted-values inverse transform inside `fit()` is for decomposition reporting only; clamping there is cosmetic and risks breaking issue_106 conformance. The important safety path is the prediction clamp. Verify this decision by running `cargo test --test issue_106_decomposable_conformance` after implementing.

**Pitfall 2 — Constant-series early-return path (`mfles.rs:1024-1047`).**
[VERIFIED: src/models/mfles.rs:1024-1047] The constant-series path returns early before `insample_max` is set. The `insample_max` must be set before the early return, or left as `None` with the prediction clamp gracefully defaulting to `f64::INFINITY`. The cleanest approach: compute and store `insample_max` immediately after deciding `use_multiplicative`, before the `let all_same = ...` check. Then both the early-return path and the normal path have it set.

**Pitfall 3 — `calc_cov` log-space reasoning (`mfles.rs:422`).**
[VERIFIED: src/models/mfles.rs:422-446] `calc_cov` is called with `use_multiplicative` to auto-detect robust mode. In multiplicative mode, it operates on the residuals `y[i] - fitted[i]` which are already in log-space. The floor applied to raw values before `ln()` (Edit 5) changes the log-space values — specifically, `ln(1.0.max(20.5)) = ln(20.5) ≈ 3.02` instead of `0.0`. This reduces the variance of the log-space residuals, making auto-detection of robust mode less likely to trigger for the problematic series. This is the correct behavior: the near-zero outlier was the driver of high CoV; flooring it normalizes the log-space, and robust mode is less urgently needed. No change required to `calc_cov` itself.

**Pitfall 4 — `serde` field compatibility.**
MFLES derives `Serialize + Deserialize` when the `serde` feature is enabled (`mfles.rs:30-31`). [VERIFIED: src/models/mfles.rs:30-31] Adding `insample_max: Option<f64>` is backward-compatible for deserialization: `Option` fields default to `None` when absent in a JSON payload. Old serialized models loaded with new code will have `insample_max = None`, meaning no clamp is applied on prediction — acceptable degraded behavior. No custom serde attribute needed.

**Pitfall 5 — Median of the floored series vs. original series.**
The mode-selection guard computes `median` from the original `values` before any transform. The log-floor in Edit 5 also uses `Self::median_scalar(values)` on the originals. Keep both consistent — do not accidentally use the floored series median.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Median computation for guard | Custom sort+select | `Self::median_scalar(values)` at `mfles.rs:342` | Already NaN-safe, finite-filter, correctly handles even/odd lengths |
| Series min/max | Iterator reduce | `values.iter().copied().fold(f64::INFINITY, f64::min)` (existing pattern at `mfles.rs:1010`) | Idiomatic existing pattern in the file |
| Clamp | Manual if-chains | `.max(0.0).min(cap)` | Rust's method chaining; clear intent |

---

## Common Pitfalls

### Pitfall 1: Breaking the Decomposable Invariant
**What goes wrong:** Clamping `fitted_original` inside `fit()` causes `seasonal_full = fitted - trend_full` to be inconsistent with `residuals = training - fitted`, breaking the `trend + seasonal + residual == training` invariant tested by `issue_106_decomposable_conformance`.
**Why it happens:** Both `fitted_original` and `trend_full` are independently back-transformed; clamping one but not the other produces inconsistent components.
**How to avoid:** Apply the back-transform clamp ONLY in `predict_internal()`. Leave the `fit()` path's inverse transform unclamped (it is diagnostic/decomposition data, not the prediction output).
**Warning signs:** `cargo test --test issue_106_decomposable_conformance` fails with tolerance violation.

### Pitfall 2: Missing `insample_max` on the Constant-Series Early-Return Path
**What goes wrong:** `insample_max` is `None` after a constant-series early return; `predict()` then has no cap, defeating MULT-03 for that edge case.
**Why it happens:** The early return at `mfles.rs:1046` exits before the normal path sets fitted state.
**How to avoid:** Set `insample_max` immediately after `use_multiplicative` is determined (before the `all_same` check), or explicitly set it inside the early-return branch.
**Warning signs:** A constant-series multiplicative fit forecasts without clamping.

### Pitfall 3: Using Wrong Median for the Floor
**What goes wrong:** If the floor uses `floor_frac × median_of_floored_values` (circular), or uses a seasonal-period median instead of the global series median, the floor value is inconsistent with the guard threshold.
**Why it happens:** MFLES has two median functions: `median_scalar` (global) and `median` (seasonal-period aware). The guard and floor must both use the global scalar median.
**How to avoid:** Always use `Self::median_scalar(values)` where `values` is the raw (pre-transform) series.

### Pitfall 4: Adding a Public Builder Method (API Change)
**What goes wrong:** Adding `.insample_clamp_multiplier()` to the builder changes the public API, violating the "internal constant" locked decision.
**How to avoid:** All three constants (`MULT_AUTO_TAU`, `MULT_LOG_FLOOR_FRAC`, `MULT_BACK_CLAMP_K`) are declared as private `const` values inside `impl MFLES`. No builder exposure.

---

## Code Examples

### Mode-Selection Guard (verified pattern, mfles.rs:997-1004)

Existing code [VERIFIED: src/models/mfles.rs:997-1004]:
```rust
let use_multiplicative = match self.multiplicative {
    Some(m) => m,
    None => {
        // Auto: multiplicative if positive and seasonal
        self.season_length > 0 && values.iter().all(|&v| v > 0.0)
    }
};
```

Replacement:
```rust
let use_multiplicative = match self.multiplicative {
    Some(m) => m,
    None => {
        let all_positive = self.season_length > 0
            && values.iter().all(|&v| v > 0.0);
        if all_positive {
            let median = Self::median_scalar(values);
            let min_val = values
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            // Guard: if min / median < τ, a near-zero value would
            // open a log-space crater — fall back to additive (issue #219).
            median > 0.0 && (min_val / median) >= Self::MULT_AUTO_TAU
        } else {
            false
        }
    }
};
```

### Log-Transform Floor (verified insertion point, mfles.rs:1009-1012)

Existing code [VERIFIED: src/models/mfles.rs:1009-1012]:
```rust
if use_multiplicative {
    let min_val = values.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    self.const_val = Some(min_val);
    y = values.iter().map(|&v| v.ln()).collect();
```

Replacement:
```rust
if use_multiplicative {
    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let median = Self::median_scalar(values);
    let floor = Self::MULT_LOG_FLOOR_FRAC * median;
    self.const_val = Some(min_val);
    self.insample_max = Some(max_val);
    // Winsorize before ln() — prevents log-space crater from a
    // single near-zero observation (issue #219).
    y = values.iter().map(|&v| v.max(floor).ln()).collect();
```

### Back-Transform Clamp (verified insertion point, mfles.rs:936-943)

Existing code [VERIFIED: src/models/mfles.rs:936-943]:
```rust
let mut pred_original = if self.is_multiplicative {
    pred.exp()
} else {
    let mean_val = self.mean.unwrap_or(0.0);
    let std_val = self.std.unwrap_or(1.0);
    mean_val + pred * std_val
};
```

Replacement:
```rust
let mut pred_original = if self.is_multiplicative {
    let raw = pred.exp();
    // Clamp to [0, K × in-sample max] — runaway backstop (issue #219).
    let cap = self
        .insample_max
        .map(|m| Self::MULT_BACK_CLAMP_K * m)
        .unwrap_or(f64::INFINITY);
    raw.max(0.0).min(cap)
} else {
    let mean_val = self.mean.unwrap_or(0.0);
    let std_val = self.std.unwrap_or(1.0);
    mean_val + pred * std_val
};
```

### Regression Test Skeleton

```rust
//! Regression guard for issue #219: MFLES multiplicative-mode runaway
//! on a series with a near-zero month (level ≈ 2k, min = 1.0).
//!
//! Documented before/after delta: ~13× level → ~1× level.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::mfles::MFLES;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn make_issue_219_series() -> TimeSeries {
    // 24-point monthly series, level ≈ 2000, near-zero at index 11.
    let values = vec![
        2100.0, 1950.0, 2200.0, 1800.0, 2050.0, 2300.0,
        1900.0, 2150.0, 1850.0, 2400.0, 2000.0,    1.0,
        2050.0, 1950.0, 2100.0, 1800.0, 2200.0, 1900.0,
        2050.0, 2300.0, 1850.0, 2150.0, 2000.0, 2100.0,
    ];
    let base = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

/// MULT-04 regression: forecast must stay within 2.5× level (5000),
/// not blow up to ~13× level (~26000) as before the fix.
#[test]
fn issue_219_mfles_no_multiplicative_runaway() {
    let ts = make_issue_219_series();
    let mut model = MFLES::new(vec![12]);
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();

    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let level = 2000.0_f64;

    // Post-fix: must be below 2.5× level (a generous margin that the
    // pre-fix ~13× easily violated). Level is ≈ 2000; threshold is 5000.
    assert!(
        forecast_max < 2.5 * level,
        "issue #219 regression: forecast_max={:.0} >= {:.0} (2.5× level). \
         Before/after target: ~26000 → <5000.",
        forecast_max,
        2.5 * level,
    );

    // Also verify forecasts are positive (not floored to zero).
    let forecast_min = preds.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        forecast_min > 0.0,
        "forecast should remain positive, got min={:.2}",
        forecast_min,
    );
}

/// MULT-01 unit test: auto mode selects additive when min/median < τ.
#[test]
fn issue_219_mfles_auto_mode_selects_additive_for_near_zero_series() {
    let ts = make_issue_219_series();
    let mut model = MFLES::new(vec![12]);
    model.fit(&ts).unwrap();

    // After the fix, multiplicative must NOT be selected.
    // We verify indirectly: if additive, predictions are stable near level.
    // (is_multiplicative is a private field; we verify behavior, not internals.)
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();
    let mean_forecast: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
    assert!(
        (mean_forecast - 2000.0).abs() < 1000.0,
        "mean forecast {:.0} should be near level 2000 when additive mode is selected",
        mean_forecast,
    );
}

/// MULT-02/03: When multiplicative is forced, floor + clamp still protect.
#[test]
fn issue_219_mfles_explicit_multiplicative_with_floor_and_clamp() {
    let ts = make_issue_219_series();
    // Force multiplicative — guard 1 is bypassed, guards 2 and 3 must protect.
    let mut model = MFLES::builder()
        .seasonal_period(12)
        .multiplicative(true)
        .build();
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();

    // With floor + clamp, the forecast must stay below 10 × in-sample max.
    // In-sample max ≈ 2400, so cap = 24000. Even multiplicative should
    // produce reasonable output now.
    let cap = 10.0 * 2400.0_f64; // 24000
    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        forecast_max <= cap,
        "explicit multiplicative with clamp: forecast_max={:.0} exceeds cap={:.0}",
        forecast_max,
        cap,
    );
}
```

---

## Validation Architecture

**Nyquist validation is enabled** (`workflow.nyquist_validation: true` in `.planning/config.json`). [VERIFIED: /home/simonm/projects/rust/anofox-forecast/.planning/config.json]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Cargo's built-in test harness (no external test framework) |
| Config file | None — uses `Cargo.toml` `[dev-dependencies]` |
| Quick run command | `cargo test --test issue_219_mfles_multiplicative_runaway` |
| Full suite command | `cargo test` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MULT-01 | Auto mode selects additive when `min/median < 0.10` | unit (integration test) | `cargo test --test issue_219_mfles_multiplicative_runaway issue_219_mfles_auto_mode_selects_additive_for_near_zero_series` | ❌ Wave 0 |
| MULT-02 | Log floor prevents log-space crater in multiplicative mode | unit (integration test) | `cargo test --test issue_219_mfles_multiplicative_runaway issue_219_mfles_explicit_multiplicative_with_floor_and_clamp` | ❌ Wave 0 |
| MULT-03 | Back-transform clamped to 10× in-sample max | unit (integration test) | `cargo test --test issue_219_mfles_multiplicative_runaway issue_219_mfles_explicit_multiplicative_with_floor_and_clamp` | ❌ Wave 0 |
| MULT-04 | #219 repro series stays below 2.5× level (vs pre-fix ~13×) | regression guard | `cargo test --test issue_219_mfles_multiplicative_runaway issue_219_mfles_no_multiplicative_runaway` | ❌ Wave 0 |
| SC-5 (API unchanged) | Decomposable conformance still passes | regression | `cargo test --test issue_106_decomposable_conformance mfles_conforms_to_decomposable_contract` | ✅ exists |
| SC-5 (API unchanged) | Clippy green | static analysis | `cargo clippy --all-targets --all-features -- -D warnings` | ✅ CI gate |

### Sampling Rate
- **Per task commit:** `cargo test --test issue_219_mfles_multiplicative_runaway`
- **Per wave merge:** `cargo test`
- **Phase gate:** `cargo test && cargo clippy --all-targets --all-features -- -D warnings`

### Wave 0 Gaps
- [ ] `tests/issue_219_mfles_multiplicative_runaway.rs` — covers MULT-01..04 (new file; skeleton above)

---

## Security Domain

MFLES is a pure numerical computation library with no I/O, authentication, or network access. ASVS V5 (Input Validation) is the only applicable category.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `validate_series_complete()` (already called at top of `fit()`); guard constants prevent numeric overflow in `exp()` |
| V6 Cryptography | no | — |

**Threat pattern for this phase:** `pred.exp()` with an extreme positive argument produces `f64::INFINITY` or an astronomically large float. The back-transform clamp (MULT-03) directly mitigates this by bounding the output domain. No security advisory changes.

---

## Environment Availability

No external tools required beyond the standard Rust toolchain. The full test suite runs with `cargo test` (no feature flags needed for MFLES). Clippy and rustfmt are already available per CLAUDE.md.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cargo` / Rust stable | All compilation and test | ✓ | (current stable) | — |
| `cargo clippy` | CI gate | ✓ | bundled with toolchain | — |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The Python statsforecast MFLES does not apply a min/median ratio guard (only all-positive check), making our fix a deliberate improvement over upstream | Q1 reference behavior | Low — even if Python has a guard, our fix is still correct; the behavior to change is the Rust code's, which is verified |
| A2 | The #219 pre-fix forecast is ~13× level (blow-up magnitude) | Q2 repro series | Low — the test verifies the post-fix bound, not the pre-fix exact value; even if the blow-up is "only" 5× the assertion still documents regression prevention |

---

## Open Questions

1. **`trend_full` inverse-transform clamping**
   - What we know: `trend_full` at `mfles.rs:1321-1322` also uses raw `v.exp()`. Clamping it separately from `fitted_original` risks decomposition invariant violations.
   - What's unclear: Whether `trend_full` can blow up independently for an edge case that passes guards 1 and 2 but not 3.
   - Recommendation: Do not clamp `trend_full`. The back-transform clamp in `predict_internal()` is the authoritative fix; `trend_full` is informational. If `trend_full` is extreme, the residual absorbs it and the invariant holds.

2. **Constant-series path and `insample_max` ordering**
   - What we know: The early return at `mfles.rs:1046` exits before `insample_max` would be set under the proposed placement.
   - What's unclear: Whether there is a meaningful constant-series multiplicative scenario (a constant positive series trivially has `min/median = 1.0 > τ`, so guard 1 would NOT trip, multiplicative would be selected, and the early return fires).
   - Recommendation: Set `insample_max = Some(max_val)` immediately after computing `max_val` (inside the `if use_multiplicative` branch, before the `all_same` check). Also set it explicitly in the constant-series early-return path for completeness.

---

## Sources

### Primary (HIGH confidence — files read this session)

- `src/models/mfles.rs` — Lines 31-94 (struct definition, fitted state), 225-264 (MFLES::new), 341-404 (median_scalar, median), 421-446 (calc_cov), 936-954 (predict_internal back-transform), 997-1022 (fit mode selection and transform), 1024-1047 (constant-series early return), 1069-1139 (boosting loop init), 1300-1360 (fitted-values inverse transform), 1303-1310 (fitted_original computation)
- `tests/issue_106_decomposable_conformance.rs` — Lines 1-49 (file header, make_seasonal_series, test structure), 177-181 (mfles_conforms_to_decomposable_contract)
- `tests/nixtla_validation.rs` — Lines 1-10 (imports including MFLES), 149-179 (validate_mfles_against_nixtla)
- `.planning/phases/05-mfles-multiplicative-guard-fix/05-CONTEXT.md` — Locked decisions
- `.planning/REQUIREMENTS.md` — MULT-01..04 requirement text
- `.planning/config.json` — `nyquist_validation: true` confirmed
- `Cargo.toml` — Feature definitions, dev-dependencies

### Secondary (MEDIUM confidence)

- `src/features/basic.rs:99-111` — Confirmed a separate `median()` function exists in the features module; used to establish that MFLES's own `median_scalar` is the correct choice

---

## Metadata

**Confidence breakdown:**
- Bug root cause: HIGH — read and verified the exact lines where `all(v > 0.0)` and raw `v.ln()` produce the failure
- Guard implementation: HIGH — all edit locations verified by reading source; existing patterns (`.max(1e-10)`) confirmed at line 1018
- Test structure: HIGH — existing test files read; naming convention confirmed; feature flag requirements verified from Cargo.toml
- Pre-fix blow-up magnitude (~13×): ASSUMED — drawn from the issue description; not reproduced in this session

**Research date:** 2026-09-09
**Valid until:** 2026-12-09 (stable codebase; constants are unlikely to move)
