# Phase 4: Prioritized Improvement Backlog & Top-Value Fixes — Research

**Researched:** 2026-08-11
**Domain:** Rust library hardening — NaN/Inf guards, coverage improvement, accuracy gap analysis, backlog construction
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **P1 raw-vec NaN/Inf guards (V-01/V-02/V-03):** add per-series, per-element NaN/Inf scans at the top of `GlobalETS::fit`, `GlobalCroston::fit`, and `GlobalTheta::fit` (raw `&[Vec<f64>]` APIs) returning `ForecastError::InvalidParameter` before the estimation loop.
- **G-01 — GlobalTheta 0% coverage:** add at least a smoke test (constructor + basic fit/predict on valid data + the new NaN-guard path).
- **V-04 (VAR variant divergence): DOCUMENT ONLY** — do not refactor error types this phase.
- **Proof:** each landed fix is proven with a before/after `coverage.json` delta (re-run `scripts/update_coverage.sh`). Capture the delta explicitly.
- **Accuracy MASE gap — time-boxed best-effort:** bounded investigation into the AutoETS M3-monthly MASE gap (~1.045 vs 0.8633 reference). Land a change only if it measurably improves MASE. If it does not converge within the time-box, STOP — keep `accuracy.json` deferred/UNLOCKED, leave the gap as the #1 ranked backlog item. Never force a lock.
- **Backlog:** rank findings from all 8 measurement dimensions using REAL numbers; flag `iai.json`/`criterion.json` as manual-capture-pending placeholders. Do NOT fabricate perf deltas.
- **Baseline updates committed as separate deliberate changes; coverage floor ratcheted up in CI.**

### Claude's Discretion

- Exact backlog document location/format (recommend `.planning/baselines/BACKLOG.md` or a phase-dir markdown)
- The precise NaN-guard error variant per model (match each model's existing convention)
- The accuracy investigation's specific hypotheses
- The time-box size

### Deferred Ideas (OUT OF SCOPE)

- Full closure of the AutoETS MASE gap (if the time-boxed attempt does not converge) — ranked #1 backlog item for a future dedicated accuracy effort.
- `iai.json` / `criterion.json` real-number capture — requires specific hardware; documented as manual-capture-pending, not attempted here.
- V-04 VAR error-variant unification — documented as a known inconsistency, not refactored this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IMPR-01 | Consolidated improvement backlog ranks findings across all 8 dimensions by value vs. effort using real baseline numbers | Backlog Construction section: 8 dimensions identified, real vs placeholder baselines mapped, ranking rubric defined |
| IMPR-02 | Highest-value improvements landed, each with a documented before/after delta in the relevant baseline file and a regression guard | NaN-Guard Implementation section: exact insertion points, error variant, code pattern. Accuracy Investigation section: lever shortlist, measurement method, STOP criterion |
| IMPR-03 | Baseline updates after an improvement are committed deliberately (separate change), never auto-written by CI | Before/After Delta & Ratchet Mechanics section: commit workflow, CI gate mechanics |
</phase_requirements>

---

## Summary

Phase 4 closes the Performance & Validation Hardening milestone. The guaranteed deliverable is landing the three P1 raw-vec NaN/Inf guards (V-01 through V-03) plus a GlobalTheta smoke test (G-01), proven by a coverage.json before/after delta, then ratcheting the CI floor upward. The speculative deliverable is a time-boxed accuracy investigation that may or may not close the M3-monthly MASE gap.

The codebase investigation confirms all three guard insertion points, the correct `ForecastError::InvalidParameter` variant (consistent with `VAR::fit`), and that `GlobalTheta` is genuinely at 0% — its entire public API is untested. The accuracy gap (+21% vs reference) is large; analysis of the AutoETS code surface suggests the gap is more likely an initialization or seasonal-detection issue than an optimizer-bounds problem, but no low-risk lever is certain to close it. The honest recommendation is to treat the accuracy lock as a fallback-to-deferred outcome.

**Primary recommendation:** Land the guards and smoke test first (safe, measurable); time-box the accuracy investigation to ≤2 hours of exploration; commit baseline updates as separate deliberate commits after each fix; ratchet the coverage floor in a dedicated commit.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| NaN/Inf guards at fit() entry | Library `src/` — model fit paths | Test suite (`tests/`) for verification | The guard lives in the model, not in test infrastructure; tests only exercise the new error path |
| GlobalTheta smoke test | Test suite (`tests/` integration tests) | — | Coverage lift is entirely in the test layer; no src/ changes for the test itself |
| Coverage measurement & ratchet | Scripts + CI (`scripts/update_coverage.sh`, `.github/workflows/ci.yml`) | `.planning/baselines/coverage.json` | The floor is read from the JSON by CI; the script writes it; neither is in src/ |
| Accuracy measurement (ACCUR-08 anchor) | Harness crate (`crates/anofox-bench-harness/tests/accuracy.rs`) | `src/models/exponential/auto_ets.rs` (if a lever is found) | The harness drives the measurement; any src/ change must be validated against the harness |
| Backlog document | `.planning/baselines/BACKLOG.md` | — | Planning artifact, not src/; sits alongside other baseline files |

---

## Project Constraints (from CLAUDE.md)

- Rust 2021 edition, stable toolchain; CI also tests beta/nightly.
- Clippy `-D warnings` must stay green on any `src/` change.
- Public `Forecaster` API stays backward-compatible.
- All measurement code stays in `benches/`, `tests/`, `scripts/`, harness crate — nothing new enters `src/` except the three guard insertions (which fix correctness, not measurement).
- `accuracy.yml` must remain `workflow_dispatch`-only; never gates PR merges.
- No customer/client names in code, comments, or test names. Use structural descriptions ("panel of N series").
- No automatic seasonal-period-detection integration.
- `git commit --no-verify` required for planning-doc commits (cargo-fmt pre-commit hook blocks them).

---

## NaN-Guard Implementation

### V-01: GlobalETS::fit

**File:** `src/models/exponential/global_ets.rs`
**Insertion point:** After the `is_empty()` check (line 82–88) and the length check (lines 95–101), before the `self.states = ...` initialization at line 104. [VERIFIED: src/models/exponential/global_ets.rs:81-103]

Existing guards (verbatim from file):
```rust
// Line 82-88:
if all_series.is_empty() {
    return Err(ForecastError::InsufficientData {
        needed: 1,
        got: 0,
        hint: Some("GlobalETS requires at least one series".into()),
    });
}
// Lines 95-101:
if n <= start_idx + 2 {
    return Err(ForecastError::InsufficientData { ... });
}
```

There is NO per-element NaN/Inf check. NaN values pass directly into `Self::initialize_state()` and the Nelder-Mead objective.

**Guard to insert (after line 101, before line 104):**
```rust
// V-01: per-series NaN/Inf guard — raw &[Vec<f64>] API bypasses validate_series_complete()
for (i, s) in all_series.iter().enumerate() {
    if s.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(ForecastError::InvalidParameter(format!(
            "series {} contains NaN or Inf values",
            i
        )));
    }
}
```

**Error variant:** `ForecastError::InvalidParameter(String)` — matches `VAR::fit` lines 119–124 exactly. [VERIFIED: src/models/var.rs:119-124]

**VAR reference pattern (verbatim):**
```rust
// var.rs lines 119-124:
if series.iter().any(|v| v.is_nan() || v.is_infinite()) {
    return Err(ForecastError::InvalidParameter(format!(
        "Variable {} contains NaN or Inf values",
        i
    )));
}
```

**Will any existing test break?** No. The existing tests on `GlobalETS` all use finite values. The guard only fires on NaN/Inf input, which no existing test provides to this function.

---

### V-02: GlobalCroston::fit

**File:** `src/models/intermittent/global_croston.rs`
**Insertion point:** After the `is_empty()` check (lines 77–83), before the `extracted: Vec<Option<...>>` block at line 86. [VERIFIED: src/models/intermittent/global_croston.rs:76-103]

Existing guard (verbatim):
```rust
// Lines 77-83:
if all_series.is_empty() {
    return Err(ForecastError::InsufficientData {
        needed: 1,
        got: 0,
        hint: Some("GlobalCroston requires at least one series".into()),
    });
}
```

No per-element check. NaN/Inf elements in a demand series pass into `Self::extract_demands()` and then into `Self::total_mse()`.

**Guard to insert (after line 83, before line 86):**
```rust
// V-02: per-series NaN/Inf guard — raw &[Vec<f64>] API bypasses validate_series_complete()
for (i, s) in all_series.iter().enumerate() {
    if s.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(ForecastError::InvalidParameter(format!(
            "series {} contains NaN or Inf values",
            i
        )));
    }
}
```

**Error variant:** `ForecastError::InvalidParameter(String)` — consistent with V-01/VAR pattern.

**Will any existing test break?** No. GlobalCroston's tests (doc-example only) use sparse zero/non-zero finite values.

---

### V-03: GlobalTheta::fit

**File:** `src/models/theta/global_theta.rs`
**Insertion point:** After the `is_empty()` check (lines 67–72), before the `nelder_mead(...)` call at line 76. [VERIFIED: src/models/theta/global_theta.rs:66-109]

Existing guard (verbatim):
```rust
// Lines 67-72:
if all_series.is_empty() {
    return Err(ForecastError::InsufficientData {
        needed: 1,
        got: 0,
        hint: Some("GlobalTheta requires at least one series".into()),
    });
}
```

No per-element check. `total_sse()` at line 132 reads raw values directly — NaN propagates into `error * error` and accumulates silently into `f64::NAN` SSE, then into `alpha`.

**Additional risk in GlobalTheta::fit:** After the optimizer runs (line 92), the `states` loop at lines 94–107 does:
```rust
let mut level = values[0];
for &y in values.iter().skip(1) {
    level = self.alpha * y + (1.0 - self.alpha) * level;
}
```
NaN in any `values[i]` propagates `level` to NaN and then to `state.b` via `ols_slope()`.

**Guard to insert (after line 72, before line 75/76):**
```rust
// V-03: per-series NaN/Inf guard — raw &[Vec<f64>] API bypasses validate_series_complete()
for (i, s) in all_series.iter().enumerate() {
    if s.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(ForecastError::InvalidParameter(format!(
            "series {} contains NaN or Inf values",
            i
        )));
    }
}
```

**Error variant:** `ForecastError::InvalidParameter(String)`.

**Will any existing test break?** There are NO existing tests for GlobalTheta — the file is at 0% coverage. The guard cannot break what does not exist.

---

### V-04: VAR error-variant divergence — DOCUMENT ONLY

`VAR::fit` (lines 119–124) returns `InvalidParameter("Variable N contains NaN or Inf values")` on the raw-vec API. `VARForecaster::fit` routes through `validate_series_complete()` which returns `MissingValues`. The guard is present but the two public entry points return divergent variants. Document in BACKLOG.md as a known inconsistency. No code change this phase. [VERIFIED: src/models/var.rs:119-124]

---

## GlobalTheta Smoke Test (G-01)

### Why GlobalTheta is at 0%

[VERIFIED: .planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md:20-20]
The gap inventory records: "Entire file 0% covered — no test exercises GlobalTheta model at all". The doc-example in the module comment is not `#[doc(test)]`-tested or otherwise executed. Lines 41–183 of `src/models/theta/global_theta.rs` are fully uncovered.

### Constructor API (verified)

From reading the file [VERIFIED: src/models/theta/global_theta.rs:41-183]:

```rust
pub fn new() -> Self           // theta=2.0, alpha=0.5
pub fn with_theta(theta: f64) -> Self
pub fn alpha(&self) -> f64
pub fn fit(&mut self, all_series: &[Vec<f64>]) -> Result<()>
pub fn predict(&self, horizon: usize) -> Vec<Vec<f64>>
```

`predict()` returns `Vec<Vec<f64>>` (not `Result`) — it returns an empty vec if not fitted. The doc example in the module header matches the real API.

### Minimal smoke test structure

A minimal test covering G-01 needs three paths:
1. Constructor + valid fit + predict (happy path — exercises new, with_theta, alpha, fit, predict, total_sse, ols_slope, ThetaState)
2. NaN-guard path (exercises the new V-03 guard)
3. Empty-input path (exercises the existing is_empty() guard — optionally: this is already the only existing guard)

**Recommended test file:** `tests/global_theta_smoke.rs` (integration test, matches project naming pattern; avoids cluttering `src/`).

**Minimal test skeleton:**
```rust
use anofox_forecast::models::theta::GlobalTheta;
use anofox_forecast::error::ForecastError;

#[test]
fn global_theta_basic_fit_predict() {
    // Two monotone increasing series of length 50
    let series: Vec<Vec<f64>> = vec![
        (0..50).map(|i| 10.0 + 0.3 * i as f64).collect(),
        (0..50).map(|i| 20.0 - 0.1 * i as f64).collect(),
    ];
    let mut model = GlobalTheta::new();
    model.fit(&series).expect("fit must succeed on valid data");

    let alpha = model.alpha();
    assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0,1), got {}", alpha);

    let forecasts = model.predict(10);
    assert_eq!(forecasts.len(), 2, "must produce one forecast per series");
    for fc in &forecasts {
        assert_eq!(fc.len(), 10, "each forecast must have horizon=10 steps");
        for &v in fc {
            assert!(v.is_finite(), "all forecast values must be finite");
        }
    }
}

#[test]
fn global_theta_nan_guard() {
    let series = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.0, f64::NAN, 3.0],  // NaN in second series
    ];
    let mut model = GlobalTheta::new();
    let err = model.fit(&series).expect_err("fit must fail on NaN input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}", err
    );
}

#[test]
fn global_theta_inf_guard() {
    let series = vec![vec![1.0, f64::INFINITY, 3.0]];
    let mut model = GlobalTheta::new();
    let err = model.fit(&series).expect_err("fit must fail on Inf input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}", err
    );
}

#[test]
fn global_theta_empty_input_guard() {
    let mut model = GlobalTheta::new();
    let err = model.fit(&[]).expect_err("fit must fail on empty slice");
    assert!(
        matches!(err, ForecastError::InsufficientData { .. }),
        "expected InsufficientData, got {:?}", err
    );
}

#[test]
fn global_theta_with_theta_constructor() {
    let series: Vec<Vec<f64>> = vec![
        (0..30).map(|i| i as f64).collect(),
    ];
    let mut model = GlobalTheta::with_theta(1.5);
    model.fit(&series).expect("fit must succeed");
    let forecasts = model.predict(5);
    assert_eq!(forecasts.len(), 1);
    assert_eq!(forecasts[0].len(), 5);
}
```

### Expected coverage lift

Before (current): `global_theta.rs` — 0/183 lines covered.
After: Exercising `new()`, `with_theta()`, `alpha()`, `fit()` (happy path + guard path + empty path), `predict()`, `total_sse()`, `ols_slope()`, `Default::default()`.

The file has 183 lines. Functions covered by the smoke test map to approximately 140–155 lines (excluding the `Default` impl which is a one-liner, and error paths already partially exercised). Estimated lift: +140 to +155 lines covered in `global_theta.rs`.

Whole-library current: 70,982 / 77,743 lines = 91.30%.
After adding ~150 lines: 71,132 / 77,743 ≈ 91.49% (rough estimate — actual measurement via `scripts/update_coverage.sh` is authoritative).

The new ratchet floor would be approximately 91.49% − 1.0% = 90.49%, up from the current 90.3%.

---

## Time-Boxed Accuracy Investigation

### Current gap

From `STATE.md` and ACCUR-08 test [VERIFIED: .planning/STATE.md:103]:
- anofox AutoETS M3-monthly MASE: **1.0452**
- statsforecast 2.0.3 reference: **0.8633**
- Gap: +0.1819 (+21%)
- Tolerance: ±0.02 → target band [0.8433, 0.8833]

The gap is large: to pass the anchor the library needs to reduce its MASE by ~0.16 to ~0.20.

### Honest likelihood assessment

**This gap is almost certainly NOT closable by low-risk parameter tweaks.** Here is why:

1. The optimizer in `ETS::optimize_params()` already uses multiple starting points (5 starts for trend models, 6 for seasonal), tolerance `1e-10`, and `max_iter` 2000–3300 per candidate. It is not obviously under-powered. [VERIFIED: src/models/exponential/ets.rs:1015-1019, 1043, 1139]

2. The seasonal F-ratio gate in `AutoETS::fit()` [VERIFIED: src/models/exponential/auto_ets.rs:430-471] suppresses seasonal candidates when F ≤ 1.0. This is a custom heuristic not present in statsforecast. For some M3-monthly series with weak seasonality this gate may suppress the correct seasonal model, forcing a worse non-seasonal fit.

3. The minimum-series-length threshold for seasonal models is `3 * seasonal_period` (= 36 observations for monthly/period=12) [VERIFIED: src/models/exponential/auto_ets.rs:427]. statsforecast uses `2 * m` (= 24). This means short M3-monthly series (~24–35 observations) are forced non-seasonal by anofox but seasonal by statsforecast.

4. The AICc criterion is default in both implementations. Model-pool differences are unlikely to account for 21%.

5. The initialization strategy [VERIFIED: src/models/exponential/ets.rs:622-704] uses regression-based level/trend for non-seasonal models (matching statsforecast) and classical decomposition for seasonal models. This is broadly correct.

**Identified LOW-RISK levers (prioritized by likelihood of impact):**

| # | Lever | Location | Expected Impact | Risk | Time Estimate |
|---|-------|----------|----------------|------|---------------|
| L1 | Lower seasonal minimum-series threshold from `3 * period` to `2 * period` | `auto_ets.rs:427` | HIGH — brings 24–35 obs series back into the seasonal candidate pool, matching statsforecast behavior | LOW — reduces a single comparison threshold; no algorithmic change | 30 min |
| L2 | Remove or relax the F-ratio seasonal gate (try F > 0.5 or disable entirely) | `auto_ets.rs:430-471` | MEDIUM — some series suppressed by this heuristic will get the right seasonal model; others may pick a worse one | LOW-MEDIUM — removing the gate entirely risks picking seasonal on trend-only series, but AICc should self-correct | 45 min |
| L3 | Ensure `AutoETS::new()` with no period config treats monthly data as seasonal (period=12) | `accuracy.rs` harness call, `auto_ets.rs:426` | MEDIUM — the harness calls `AutoETS::new()` without `.with_period(12)`, so `seasonal_period = self.config.seasonal_period.unwrap_or(1)` gives period=1 → NO SEASONAL CANDIDATES AT ALL | HIGH — if confirmed, this is the root cause of the gap | 15 min investigation |
| L4 | Add `period=12` to the harness AutoETS call | `accuracy.rs:229` | HIGH if L3 confirmed | LOW in the harness (test-only change) | 15 min |

**Critical finding — L3/L4:** Reading the harness [VERIFIED: crates/anofox-bench-harness/tests/accuracy.rs:229]:
```rust
let mut autoets = AutoETS::new();
```
And reading `AutoETS::fit()` [VERIFIED: src/models/exponential/auto_ets.rs:426]:
```rust
let seasonal_period = self.config.seasonal_period.unwrap_or(1);
let has_seasonal = seasonal_period > 1 && values.len() >= 3 * seasonal_period;
```
When `seasonal_period = 1`, `has_seasonal = false` and ALL seasonal candidates are excluded. The harness creates `AutoETS::new()` without specifying a period, so for M3-monthly series (period=12) the model runs a non-seasonal search only. This is almost certainly the dominant cause of the MASE gap. The statsforecast Python call uses `AutoETS(season_length=12)`.

**Recommended investigation order:**
1. First (15 min): Confirm by adding `AutoETS::with_period(12)` in the harness for the monthly frequency (a harness-only change), re-run the anchor test, and observe the MASE delta.
2. If MASE drops dramatically (expected): this is the root cause. The harness change is the fix. No `src/` changes needed.
3. If MASE still fails the anchor after L4: apply L1 (lower threshold to 2*period), re-run.
4. If still failing: try L2 (relax F-ratio gate), re-run.
5. Time-box: spend no more than 2 hours total. If the anchor still fails after L4 + L1 + L2, **STOP**. Do not force a lock. Document gap as #1 backlog item with root-cause analysis.

### Measurement method

Run the ACCUR-08 anchor test with the M3 dataset available:
```bash
ANOFOX_DATASET_DIR=/path/to/validation/data \
  cargo test --package anofox-bench-harness \
  --test accuracy accur08_anchor_m3_monthly_autoets \
  -- --nocapture 2>&1 | grep -E "ACCUR-08|MASE|PASSED|FAILED"
```

The test reports `autoets_mase` in the eprintln. Compare before/after.

### STOP criterion (non-negotiable)

Stop the investigation if ANY of:
1. 2 hours elapsed without a passing ACCUR-08 anchor.
2. Any proposed change touches the core ETS likelihood or smoothing equations (algorithmic rewrite, out of scope).
3. Any proposed change would break an existing passing test.

If stopped: keep `accuracy.json` absent (deferred). Record the investigation's findings and root-cause hypothesis in BACKLOG.md as #1 item.

### If investigation succeeds

If the anchor passes (MASE within 0.8433–0.8833):
1. Commit the harness/src change.
2. Re-run the full harness with `ANOFOX_WRITE_ACCURACY_BASELINE=1` to emit `accuracy.json`:
   ```bash
   ANOFOX_DATASET_DIR=/path/to/data ANOFOX_WRITE_ACCURACY_BASELINE=1 \
     cargo test --package anofox-bench-harness --test accuracy -- --nocapture
   ```
3. Commit `accuracy.json` as a separate deliberate change (IMPR-03).

---

## Backlog Construction

### The 8 Measurement Dimensions and Baseline Status

| # | Dimension | Baseline File | Status | Real Numbers Available? |
|---|-----------|---------------|--------|------------------------|
| 1 | Code Coverage | `coverage.json` | Committed | YES — 91.30% line coverage (70,982/77,743 lines), floor 90.3% [VERIFIED: .planning/baselines/coverage.json:12-18] |
| 2 | WASM Binary Size | `wasm_size.json` | Committed | YES — 2,838,958 bytes [VERIFIED: .planning/baselines/wasm_size.json:11] |
| 3 | Peak Memory (dhat) | `dhat.json` | Committed | YES — real peak_bytes per model family [VERIFIED: .planning/baselines/dhat.json:11-35] |
| 4 | Reference Accuracy | `statsforecast_reference.json` | Committed | YES — M3 monthly MASE=0.8633, quarterly=1.1436, yearly=2.6954 [VERIFIED: .planning/baselines/statsforecast_reference.json] |
| 5 | Accuracy (anofox) | `accuracy.json` | ABSENT (deferred, anchor failed) | NO — anofox MASE=1.0452 known from STATE.md but not locked in a file |
| 6 | Instruction Count (iai) | `iai.json` | Committed PLACEHOLDER | NO — all values are 0; requires valgrind ≥3.20 machine [VERIFIED: .planning/baselines/iai.json] |
| 7 | Wall-clock (criterion) | `criterion.json` | Committed PLACEHOLDER | NO — all median_ns=0.0; requires quiet local machine [VERIFIED: .planning/baselines/criterion.json] |
| 8 | Numerical Robustness | `03-GAP-INVENTORY.md` | Committed (20 rows) | YES — 5 P1 / 9 P2 / 6 P3 rows from Phase 3 |

### Ranking Rubric

For each backlog item, score on three axes (1–5 each):

| Axis | 5 | 1 |
|------|---|---|
| **Value** | Correctness risk or large accuracy/perf delta with real evidence | No real evidence; cosmetic |
| **Effort** | One function change, verified in hours | Multi-week algorithmic work |
| **Risk** | Guard path only; can't break existing passing tests | Changes existing passing test behavior; algorithmic |

**Priority score = Value × (Effort × Risk) / 5** (higher = do first). Items with real baseline evidence rank ahead of placeholder items.

### Recommended BACKLOG.md Location and Format

**Location:** `.planning/baselines/BACKLOG.md` (alongside other baseline files, consumable by future planning sessions).

**Format:**
```markdown
# Improvement Backlog

**Last updated:** YYYY-MM-DD
**Phase 4 inputs:** coverage.json (real), wasm_size.json (real), dhat.json (real),
statsforecast_reference.json (real), accuracy.json (absent/deferred), iai.json (placeholder),
criterion.json (placeholder), 03-GAP-INVENTORY.md (20 rows).

## Ranked Items

| Rank | ID | Title | Dimension | Value | Effort | Risk | Evidence | Status |
|------|----|-------|-----------|-------|--------|------|----------|--------|
| 1 | ACC-01 | Close AutoETS M3-monthly MASE gap ... | Accuracy | 5 | 3 | 2 | MASE=1.0452 vs ref=0.8633 | Deferred (if P4 time-box fails) OR Landed |
...
```

---

## Before/After Delta & Ratchet Mechanics

### Coverage Delta Capture

**Step 1 — Capture BEFORE number (already in `coverage.json`):**
The current baseline is [VERIFIED: .planning/baselines/coverage.json:12-18]:
- `lines_total`: 77743
- `lines_covered`: 70982
- `lines_percent`: 91.30339709041328
- `ratchet_floor_percent`: 90.3

**Step 2 — Make the code change** (add guards + smoke test).

**Step 3 — Capture AFTER number:**
```bash
bash scripts/update_coverage.sh
```
The script re-runs `cargo llvm-cov --package anofox-forecast --all-features --json --summary-only`, computes `ratchet_floor_percent = lines_percent - 1.0`, and overwrites `coverage.json`. [VERIFIED: scripts/update_coverage.sh:52-83]

**Step 4 — Document the delta explicitly:**
In the plan's task output, record: "Before: 91.30%, After: XX.XX%, delta: +Y.YY pp, new floor: ZZ.ZZ%".

**Step 5 — Commit the baseline update as a separate deliberate commit:**
```bash
git add .planning/baselines/coverage.json
git commit --no-verify -m "chore(baselines): ratchet coverage floor to XX.X% after V-01/V-02/V-03 + G-01"
```
Never bundle the baseline update into the same commit as the code change.

### Ratchet Mechanics

The CI gate reads the floor from the committed file [VERIFIED: .github/workflows/ci.yml:159-163]:
```yaml
- name: Enforce coverage floor
  run: |
    FLOOR=$(jq '.coverage.ratchet_floor_percent' .planning/baselines/coverage.json)
    echo "Coverage ratchet floor: ${FLOOR}%"
    cargo llvm-cov --package anofox-forecast --all-features --summary-only --fail-under-lines "$FLOOR"
```

Updating the floor: re-run `scripts/update_coverage.sh`, commit the new `coverage.json`. CI will automatically enforce the higher floor on subsequent PRs. The ratchet is up-only by convention: `ratchet_floor_percent = lines_percent - 1.0`.

### Accuracy Baseline Commit (if investigation succeeds)

The harness emits `accuracy.json` ONLY when both conditions are met:
1. `ANOFOX_WRITE_ACCURACY_BASELINE=1` env var is set
2. The ACCUR-08 anchor assertion has passed (tested internally before writing)

[VERIFIED: crates/anofox-bench-harness/tests/accuracy.rs:31-34]

After successful write, commit separately:
```bash
git add .planning/baselines/accuracy.json
git commit --no-verify -m "chore(baselines): lock accuracy.json — AutoETS M3-monthly MASE=X.XXXX (anchor passed)"
```

---

## Common Pitfalls

### Pitfall 1: Forcing an accuracy.json lock when the ACCUR-08 anchor fails

**What goes wrong:** Committing `accuracy.json` with a known-failing anchor makes the committed number untrustworthy — future comparisons cannot distinguish "baseline improved" from "we relaxed the gate".

**Why it happens:** Pressure to complete the phase; the emit function has a dual-key guard but it can be bypassed by direct file construction.

**How to avoid:** The `emit_accuracy_baseline_if_write_flag_set` function contains an internal assert that panics if the anchor check fails — rely on this. Never write `accuracy.json` manually. If the function panics, the investigation hasn't converged.

**Warning signs:** The ACCUR-08 anchor test fails; you're considering adjusting the tolerance or hardcoding a passing value.

---

### Pitfall 2: Fabricating deltas for placeholder baselines

**What goes wrong:** Writing backlog items like "criterion shows AutoETS fit takes Xms — improvement opportunity" when `criterion.json` has all zeros. The reader cannot verify the claim.

**How to avoid:** All `iai.json` and `criterion.json` backlog items MUST be marked `manual-capture-pending` with the capture command. Never infer a numeric value from placeholder zeros.

**Warning signs:** A backlog item cites a criterion or iai value that is not 0.0 when you know the file has placeholders.

---

### Pitfall 3: A NaN-guard that changes a passing test's behavior

**What goes wrong:** An existing test passes a series with a NaN at position 0 (e.g., testing that the model handles it gracefully with a warning). The guard now returns `Err(InvalidParameter)` instead, breaking the test.

**How to avoid:** Search for existing tests that call `GlobalETS::fit`, `GlobalCroston::fit`, or `GlobalTheta::fit` with potentially non-finite values before adding the guard. The scan in Phase 3 found no existing tests on GlobalTheta, and GlobalETS/GlobalCroston tests use finite demo values (confirmed by reading doc examples).

**Warning signs:** `cargo test` fails after adding the guard; the failing test was previously passing.

---

### Pitfall 4: Bundling baseline updates with code changes in a single commit

**What goes wrong:** If a bisect identifies the code change as introducing a regression, the baseline delta is also in the same commit, making it impossible to revert just the code without the baseline reverting too.

**How to avoid:** Always commit in this order: (1) code change, (2) tests, (3) baseline update as a separate deliberate commit. IMPR-03 requires this.

---

### Pitfall 5: The `AutoETS::new()` vs `AutoETS::with_period(12)` root cause

**What goes wrong:** The accuracy gap investigation spends time on optimizer tuning (L3/L4 in the lever table) when the root cause is that the harness creates `AutoETS::new()` without specifying period=12, causing period to default to 1 and ALL seasonal candidates to be excluded.

**How to avoid:** Verify L3 first (15 minutes) before touching any optimizer parameters. This is the highest-probability root cause given the observed gap magnitude.

---

### Pitfall 6: Coverage measurement scope mismatch

**What goes wrong:** Running coverage without `--package anofox-forecast` or without `--all-features` produces different numbers from the committed baseline, making the before/after delta meaningless.

**How to avoid:** Always use `bash scripts/update_coverage.sh` which enforces the correct scope. Never run `cargo llvm-cov` manually with different flags and then compare. [VERIFIED: scripts/update_coverage.sh:52-58]

---

## Standard Stack

No new dependencies are introduced in this phase. All tools are already present.

| Tool | Version | Purpose | Already Available |
|------|---------|---------|------------------|
| `cargo-llvm-cov` | 0.8.4 | Coverage measurement + ratchet | Yes [VERIFIED: .planning/baselines/coverage.json:7] |
| `jq` | system | JSON extraction in coverage script | Used in update_coverage.sh |
| `ForecastError::InvalidParameter` | — | Error variant for NaN guards | Defined in src/error.rs [VERIFIED: src/error.rs:24-25] |

## Package Legitimacy Audit

No new packages are introduced in this phase. This section is not applicable.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NaN/Inf element detection in a `Vec<f64>` | Custom NaN propagation tracker | `iter().any(\|v\| v.is_nan() \|\| v.is_infinite())` | Standard library method; same pattern as VAR::fit lines 119-121 |
| Coverage measurement | Custom line counter | `cargo llvm-cov` with `--json --summary-only` | Already integrated in scripts/update_coverage.sh |
| Accuracy baseline writing | Manual JSON construction | `emit_accuracy_baseline_if_write_flag_set()` in harness | Has dual-key guard that enforces anchor-passing precondition |

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Raw-vec fit paths rely on optimizer to handle NaN (silent propagation) | Explicit per-element scan before estimation loop | Fails fast with actionable error; matches VAR::fit convention |
| GlobalTheta entirely uncovered | Smoke test covers constructor, fit, predict, guard | Coverage lift ~150 lines; correctness signal for an otherwise invisible model |
| Accuracy baseline deferred (MASE gap +21%) | Time-boxed investigation; root cause likely the missing `period=12` in harness | If L4 is the fix: accuracy.json lockable in one harness change; gap closes to near reference |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `AutoETS::new()` without `.with_period(12)` is the dominant cause of the MASE gap | Time-Boxed Accuracy Investigation | If wrong, the gap requires deeper algorithmic investigation beyond the time-box; fallback is to defer |
| A2 | Adding ~150 lines of coverage in global_theta.rs will push the whole-library % above 91.49% | GlobalTheta Smoke Test | If wrong (e.g., line count is lower), the ratchet still moves upward, just less than estimated |
| A3 | The doc-example tests in global_theta.rs are not currently executed by `cargo test` | GlobalTheta Smoke Test | If they are executed (e.g., via cargo test --doc), the baseline 0% figure is wrong — but the GAP-INVENTORY explicitly records 0% after a full llvm-cov run |

---

## Open Questions

1. **Is the `AutoETS::new()` + no period the root cause of the MASE gap?**
   - What we know: The harness uses `AutoETS::new()` without period; `seasonal_period.unwrap_or(1)` returns 1; `has_seasonal = (1 > 1)` = false → no seasonal candidates. statsforecast uses `season_length=12`.
   - What's unclear: Whether the M3-monthly series in the corpus are long enough (≥3×12=36 observations) for the seasonal gate to matter even if period were passed.
   - Recommendation: The 15-minute L3 check settles this definitively. Prioritize it as investigation step 1.

2. **Will the three NaN-guard commits trigger clippy warnings?**
   - What we know: The pattern is identical to VAR::fit lines 119–124 which already passes clippy.
   - What's unclear: Whether clippy has a lint for `iter().any(|v| v.is_nan() || v.is_infinite())` vs `iter().any(|v| !v.is_finite())`.
   - Recommendation: Use the `!v.is_finite()` form — shorter and clippy-preferred in Rust.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cargo-llvm-cov` | Coverage delta capture | Yes | 0.8.4 | None — required for coverage |
| M3 dataset (`ANOFOX_DATASET_DIR`) | Accuracy investigation (ACCUR-08) | Conditional | — | Skip accuracy investigation (fallback: keep deferred) |
| `jq` | `update_coverage.sh` extraction | Assumed available | — | Script has python3 fallback [VERIFIED: scripts/update_coverage.sh:67-76] |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `cargo test` + `cargo-llvm-cov` for measurement |
| Config file | None (standard Rust test harness) |
| Quick run command | `cargo test --package anofox-forecast -- --test global_theta_smoke` |
| Full suite command | `cargo test --package anofox-forecast --all-features` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| IMPR-02 (V-01/V-02/V-03) | NaN/Inf guard returns `InvalidParameter` | unit/integration | `cargo test global_theta_nan_guard global_ets_nan_guard global_croston_nan_guard` | No — Wave 0 gap |
| IMPR-02 (G-01) | GlobalTheta smoke test | integration | `cargo test --package anofox-forecast -p anofox-forecast --test global_theta_smoke` | No — Wave 0 gap |
| IMPR-02 (accuracy) | ACCUR-08 anchor passes | integration (requires env) | `ANOFOX_DATASET_DIR=... cargo test accur08_anchor_m3_monthly_autoets` | Yes (accuracy.rs) |
| IMPR-03 | coverage.json ratchet floor rises | manual verification | `bash scripts/update_coverage.sh; jq .coverage coverage.json` | Yes (script exists) |

### Wave 0 Gaps

- [ ] `tests/global_theta_smoke.rs` — covers G-01 + V-03 NaN guard
- [ ] NaN-guard tests for V-01 (GlobalETS) and V-02 (GlobalCroston) — can live in existing integration test files or in new `tests/global_model_nan_guards.rs`

---

## Security Domain

No security-relevant changes in this phase. The NaN guards add correctness but not authentication, authorization, or cryptographic changes. ASVS V5 (Input Validation) is nominally applicable since the guards validate inputs, but the risk is misuse-of-library (incorrect numerical outputs), not a security vulnerability.

---

## Sources

### Primary (HIGH confidence — verified by reading files this session)

- `src/models/exponential/global_ets.rs` — GlobalETS::fit insertion point confirmed (lines 81–103)
- `src/models/intermittent/global_croston.rs` — GlobalCroston::fit insertion point confirmed (lines 76–103)
- `src/models/theta/global_theta.rs` — GlobalTheta API confirmed, 0% coverage consistent with GAP-INVENTORY
- `src/models/var.rs` — VAR::fit NaN-guard pattern confirmed (lines 119–124)
- `src/error.rs` — ForecastError enum variants confirmed (lines 10–68)
- `src/models/exponential/auto_ets.rs` — AutoETS seasonal period logic confirmed (lines 426–427, 432–477)
- `crates/anofox-bench-harness/tests/accuracy.rs` — AutoETS::new() without period confirmed (line 229); ACCUR-08 anchor mechanics confirmed (lines 677–718)
- `.planning/baselines/coverage.json` — real numbers confirmed
- `.planning/baselines/wasm_size.json` — real numbers confirmed
- `.planning/baselines/dhat.json` — real numbers confirmed
- `.planning/baselines/statsforecast_reference.json` — real numbers confirmed
- `.planning/baselines/iai.json` — placeholder zeros confirmed
- `.planning/baselines/criterion.json` — placeholder zeros confirmed
- `scripts/update_coverage.sh` — ratchet mechanics confirmed
- `.github/workflows/ci.yml` — CI coverage gate confirmed (lines 159–163)
- `.planning/STATE.md` — MASE gap value (1.0452 vs 0.8633) confirmed (line 103)
- `.planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md` — all 20 rows confirmed

### Tertiary (LOW confidence)

- Estimated coverage lift ~150 lines for GlobalTheta [ASSUMED] — exact number requires running `scripts/update_coverage.sh` after adding tests.

---

## Metadata

**Confidence breakdown:**
- NaN-guard insertion points: HIGH — read the source files, confirmed exact line ranges and guard patterns
- GlobalTheta smoke test API: HIGH — read the full 183-line file, all public methods verified
- Accuracy investigation root cause hypothesis: MEDIUM — the `AutoETS::new()` without period finding is verified from code, but whether it explains the full 21% gap is an empirical question to be answered by running the harness
- Backlog structure: HIGH — based on real baseline numbers from all 6 committed files
- Coverage lift estimate: LOW — [ASSUMED] rough calculation before running the measurement

**Research date:** 2026-08-11
**Valid until:** End of Phase 4 (all findings reference committed files and stable source paths)
