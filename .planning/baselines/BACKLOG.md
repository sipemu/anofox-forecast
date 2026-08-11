# Improvement Backlog

**Last updated:** 2026-08-11
**Milestone:** Performance & Validation Hardening (v1.0, Phase 1–4)
**Phase 4 inputs:**
- `coverage.json` — REAL (post-04-01 ratchet: 91.44%, floor 90.4%)
- `wasm_size.json` — REAL (2,838,958 bytes)
- `dhat.json` — REAL (peak_bytes per model family)
- `statsforecast_reference.json` — REAL (M3 monthly MASE=0.8633, quarterly=1.1436, yearly=2.6954)
- `accuracy.json` — ABSENT/DEFERRED (04-02 anchor FAILED: anofox MASE=0.8923 vs upper tolerance=0.8833)
- `iai.json` — PLACEHOLDER (all instruction_count=0 — requires Valgrind ≥3.20 machine)
- `criterion.json` — PLACEHOLDER (all median_ns=0.0 — requires quiet local machine)
- `03-GAP-INVENTORY.md` — REAL (20 rows: V-01..V-04, G-01..G-10, A-01..A-05)

---

## Ranking Rubric

Score each open item on three axes (1–5 each), then compute:

**Priority = Value × (Effort × Risk) / 5**

(Higher score = do first. Items with real baseline evidence rank ahead of placeholder items.)

| Axis | 5 | 3 | 1 |
|------|---|---|---|
| **Value** | Correctness risk, large measured accuracy/perf delta with real evidence | Moderate correctness signal, some real evidence | No real evidence; cosmetic / print-only |
| **Effort** | One-function change, verified in < 1 hour | A few files, 2–4 hours | Multi-week algorithmic work |
| **Risk** | Guard path only; cannot break existing passing tests | Touches existing behavior; tests required | Changes core math or algorithmic equations |

Real-evidence rows rank ahead of placeholder rows (an unscored "pending" item has no priority until a real number exists).

---

## Ranked Open Items

Items ordered by priority score (descending). Only OPEN/DEFERRED items appear here.

| Rank | ID | Title | Dimension | Value | Effort | Risk | Priority | Evidence | Status |
|------|----|-------|-----------|-------|--------|------|----------|----------|--------|
| 1 | ACC-01 | Close residual AutoETS M3-monthly MASE gap | Accuracy | 5 | 3 | 3 | 9.0 | anofox MASE=0.8923 vs ref=0.8633 (gap=+0.0290; +0.0090 outside tolerance ±0.02); statsforecast_reference.json + 04-02 anchor run | Deferred |
| 2 | G-07 | SmartForecaster dispatch branch coverage | Numerical Robustness / Coverage | 4 | 3 | 2 | 4.8 | 34.4% file coverage; AID classification thresholds and retail/continuous routing branches untested; 03-GAP-INVENTORY.md G-07 | Open |
| 3 | G-09 | Batch partial-failure error aggregation | Numerical Robustness / Coverage | 4 | 2 | 2 | 3.2 | 42.2% file coverage; `Vec<Result<_>>` Err arms untested; 03-GAP-INVENTORY.md G-09 | Open |
| 4 | G-10 | CQR calibration edge paths | Numerical Robustness / Coverage | 4 | 2 | 2 | 3.2 | 43.0% file coverage; small-calibration-set and boundary-alpha behavior uncovered; 03-GAP-INVENTORY.md G-10 | Open |
| 5 | G-06 | Quantile anomaly scoring edge cases | Numerical Robustness / Coverage | 3 | 2 | 2 | 2.4 | 34.2% file coverage; output range [0,1] and constant-series edge case uncovered; 03-GAP-INVENTORY.md G-06 | Open |
| 6 | G-08 | GPD tail shape-parameter edge paths | Numerical Robustness / Coverage | 3 | 2 | 2 | 2.4 | 42.3% file coverage; xi < 0 / xi > 1 regimes not tested; 03-GAP-INVENTORY.md G-08 | Open |
| 7 | G-02 | Changepoint-integrated season detection | Numerical Robustness / Coverage | 3 | 3 | 2 | 3.6 | `resolve_with_data()` 21% covered; `detect_changepoints()` 0% covered; 03-GAP-INVENTORY.md G-02 | Open |
| 8 | G-04 | Inspectable model explanation coverage | Numerical Robustness / Coverage | 3 | 3 | 2 | 3.6 | 15.4% file coverage; `impl Inspectable` methods untested for most models; 03-GAP-INVENTORY.md G-04 | Open |
| 9 | G-05 | ChangeDetector alternative dispatch methods | Numerical Robustness / Coverage | 3 | 2 | 2 | 2.4 | `predict_n_bkps()`, `predict_eps()`, `fit_predict_n_bkps()` never called; `Err(Unsupported)` path uncovered; 03-GAP-INVENTORY.md G-05 | Open |
| 10 | V-04 | VAR error-variant divergence (known inconsistency) | Numerical Robustness | 2 | 2 | 2 | 1.6 | `VAR::fit()` raw-vec API returns `InvalidParameter("Variable N contains NaN")`; `VARForecaster::fit()` via `validate_series_complete()` returns `MissingValues` — divergent variants from the same logical condition; 03-GAP-INVENTORY.md V-04 | Open (known issue — NOT refactored this phase) |
| 11 | A-04 | `benchmark_auto_ets_batch_vs_individual` assertion gap | Test Quality | 2 | 1 | 4 | 1.6 | Prints timing speedup but no `assert!(speedup > 1.0)`; passes without asserting correctness; 03-GAP-INVENTORY.md A-04 | Open |
| 12 | A-05 | `benchmark_global_ets_ann_vs_individual` assertion gap | Test Quality | 2 | 1 | 4 | 1.6 | Same pattern: timing + print, no output correctness bound; 03-GAP-INVENTORY.md A-05 | Open |
| 13 | A-01 | `order_selection_comparison` assertion gap | Test Quality | 2 | 1 | 4 | 1.6 | Prints ARIMA order comparison table; no check that Rust/Python orders agree or forecast is finite; 03-GAP-INVENTORY.md A-01 | Open |
| 14 | A-02 | `arima_210_parameter_comparison` assertion gap | Test Quality | 2 | 1 | 4 | 1.6 | Prints AR coefficients and intercept but no tolerance check vs Python statsmodels reference; 03-GAP-INVENTORY.md A-02 | Open |
| 15 | A-03 | `investigate_outliers` assertion gap | Test Quality | 2 | 1 | 4 | 1.6 | Diagnostic/investigation test; no assertion on outlier detection accuracy or output shape; 03-GAP-INVENTORY.md A-03 | Open |
| 16 | WSZ-01 | WASM binary size reduction | WASM Size | 3 | 3 | 3 | 5.4 | 2,838,958 bytes (wasm_size.json); no measured reduction target yet; Effort 3 because dead-code/feature audit needed | Open |
| 17 | MEM-01 | AutoETS peak memory reduction | Peak Memory | 3 | 4 | 3 | 7.2 | dhat.json: auto_ets=290,440 bytes (highest of all families); auto_arima=191,268, auto_ensemble=199,976; auto_ets is outlier | Open |

### Ranking Rationale

**ACC-01 (#1):** The highest-evidence, highest-consequence open item. The real MASE numbers (0.8923 vs 0.8633) come from a committed anchor run; the gap (+0.0290, +0.0090 above tolerance) is precisely quantified. Two low-risk source-level levers remain untried (L1: lower seasonal minimum 3×→2×period at `auto_ets.rs:427`; L2: relax F-ratio gate at `auto_ets.rs:430-471`). Value=5 because accuracy is a primary product-quality metric; Effort=3 because each lever is a 1–2 line change + one anchor run; Risk=3 because L2 touches existing seasonal-gate logic (test impact to be measured). Priority formula: 5×(3×3)/5 = 9.0.

**G-07, G-09, G-10 (#2–4):** Real coverage % from 03-GAP-INVENTORY. These three are rated Value=4 because they cover safety-relevant dispatch logic (SmartForecaster routing), error aggregation correctness (batch Err arms), and prediction-interval validity (CQR boundary behavior). Effort=2–3 (targeted test additions). Risk=2 (tests only, no src changes). G-07 slightly lower priority than G-09/G-10 because effort is slightly higher (more routing branches to cover).

**G-02 and G-04 (#7–8):** These are higher-effort P2 gaps. G-02 requires the `seasonal-detection` feature and changepoint dataset; G-04 requires exercising `impl Inspectable` across most model families. Value=3 (coverage confidence, not correctness risk). Effort=3.

**V-04 (#10):** The guard IS present in `VAR::fit()`; the inconsistency is the error variant returned (InvalidParameter vs MissingValues depending on the call path). Correctness risk is low (the error IS returned; only the type differs). Effort=2 (one-line change to align variants or a doc note). Risk=2 (changes an error variant; any code matching on the variant must be updated). Not refactored this phase per user constraint.

**A-01–A-05 (#11–15):** P3 items. Tests exist and pass but assert nothing. Value=2 (weak correctness signal); Effort=1 (adding an assert is trivial); Risk=4 because a newly added assertion may expose a pre-existing flaw and "break" a previously green test — the risk here is discovering that the library behavior does not match the expectation, which is good, but sets expectations appropriately. Priority = 2×(1×4)/5 = 1.6 for all five.

**WSZ-01 (#16) and MEM-01 (#17):** Real numbers from wasm_size.json and dhat.json respectively. WSZ-01: 2.84 MB WASM with no reduction target measured yet; a dead-code/feature audit could yield 5–20% reduction. MEM-01: auto_ets at 290 KB is the highest-peak-memory model family (auto_ensemble=200 KB, auto_arima=191 KB, auto_theta=133 KB); understanding and reducing this gap has real production value. Priority adjusted: MEM-01 scores 7.2 (Value=3, Effort=4 because root-cause needed, Risk=3) but sits at the end of the open list because the effort to establish a root cause is higher and the correctness risk is zero.

---

## Manual-Capture-Pending Items

These items have no real numeric baseline yet. They are excluded from the ranked table — no priority score can be assigned without a measured number. Each row carries the exact capture command to unblock measurement.

| ID | Title | Dimension | Placeholder Baseline | Capture Command | Notes |
|----|-------|-----------|---------------------|-----------------|-------|
| PERF-IAI-01 | bench_auto_ets_fit instruction count | Instruction Count (iai) | instruction_count=0 (placeholder in iai.json) | Requires Valgrind ≥3.20 machine. Run: `cargo bench --package anofox-bench-harness --bench iai_suite` on a machine with `valgrind --tool=callgrind` available. Capture stdout JSON output and commit to `iai.json`. | iai.json was committed as a structural placeholder — all values are 0. Do NOT infer any numeric delta from these zeros. |
| PERF-IAI-02 | bench_auto_arima_fit instruction count | Instruction Count (iai) | instruction_count=0 (placeholder in iai.json) | Same as PERF-IAI-01. | Same machine requirement. |
| PERF-IAI-03 | bench_batch_100 instruction count | Instruction Count (iai) | instruction_count=0 (placeholder in iai.json) | Same as PERF-IAI-01. | Same machine requirement. |
| PERF-CRT-01 | auto_arima_fit_predict_n200 wall-clock | Wall-clock (criterion) | median_ns=0.0 (placeholder in criterion.json) | Requires a quiet local machine (no background processes, CPU governor set to performance). Run: `cargo bench --package anofox-bench-harness --bench criterion_suite -- --output-format json 2>/dev/null \| jq '.benchmarks'`. Commit results to `criterion.json`. | criterion.json was committed as a structural placeholder — all values are 0.0. |
| PERF-CRT-02 | auto_ets_fit_predict_n200_p12 wall-clock | Wall-clock (criterion) | median_ns=0.0 (placeholder in criterion.json) | Same as PERF-CRT-01. | Same machine requirement. |
| PERF-CRT-03 | auto_theta, naive, croston, ensemble, laplace wall-clock (8 benchmarks) | Wall-clock (criterion) | median_ns=0.0 (placeholder in criterion.json) | Same as PERF-CRT-01. | Covers 8 remaining criterion benchmark entries (parallel + no_parallel profiles). |

**Important:** Until these captures are run on appropriate hardware and the real numbers committed, no wall-clock or instruction-count improvement can be claimed or ranked. Any optimization undertaken in the meantime should document an expected improvement direction (e.g., "expected to reduce instruction count in ETS optimize_params by removing redundant alloc") but must be re-measured post-commit to verify the direction.

---

## Landed This Phase

Items completed and committed in Phase 4 (04-01 and 04-02). NOT backlog candidates — they are done.

| ID | Title | Dimension | Before | After | Delta | Evidence | Phase | Commit |
|----|-------|-----------|--------|-------|-------|----------|-------|--------|
| V-01 | GlobalETS NaN/Inf guard | Numerical Robustness | No per-element guard | `ForecastError::InvalidParameter("series N contains NaN or Inf values")` returned on fit | Guard closes silent-propagation risk | tests/global_model_nan_guards.rs: 3 tests (nan_guard, inf_guard, valid_panel_ok) | 04-01 | 3fc82aa |
| V-02 | GlobalCroston NaN/Inf guard | Numerical Robustness | No per-element guard | Same guard pattern as V-01 | Guard closes silent-propagation risk | tests/global_model_nan_guards.rs: 3 tests | 04-01 | 3fc82aa |
| V-03 | GlobalTheta NaN/Inf guard | Numerical Robustness | No per-element guard | Same guard pattern as V-01 | Guard closes silent-propagation risk | tests/global_theta_smoke.rs: nan_guard + inf_guard tests | 04-01 | e2d2e73 |
| G-01 | GlobalTheta smoke test (0% → covered) | Coverage | 0% line coverage for entire global_theta.rs | Constructor, fit, predict, with_theta(), alpha(), and guard paths all exercised | Closes the single largest 0%-coverage model void | tests/global_theta_smoke.rs: 5 integration tests | 04-01 | e2d2e73 |
| COV-01 | Coverage ratchet: 91.30% → 91.44% (floor 90.3% → 90.4%) | Coverage | 91.30% / floor 90.3% | 91.44% / floor 90.4% | +0.14 pp coverage, +0.1 pp floor | coverage.json — real numbers; separate deliberate commit | 04-01 | 84f9863 |
| ACC-HARNESS | AutoETS M3-monthly harness period=12 fix (partial improvement; anchor NOT locked) | Accuracy | MASE=1.0452 (period=1 default, no seasonal candidates) | MASE=0.8923 (period=12 passed, seasonal candidates enabled) | −0.1529 (−14.6%); closes 84% of gap; anchor FAILED (0.8923 > 0.8833 upper tolerance) | 04-02 anchor run; crates/anofox-bench-harness/tests/accuracy.rs | 04-02 | b6a75cd |

---

## Known Inconsistency: V-04 VAR Error-Variant Divergence

**Status:** Documented, not refactored.

The `VAR` model family has two public fit paths that return different error variants for the same logical condition (NaN or Inf input values):

- `VAR::fit(&mut self, data: &[Vec<f64>])` at `var.rs:119-124`: returns `ForecastError::InvalidParameter("Variable N contains NaN or Inf values")` via an explicit per-variable scan.
- `VARForecaster::fit(...)` at `var_forecaster.rs`: routes through `validate_series_complete()` on the `TimeSeries` wrapper, which returns `ForecastError::MissingValues` — a semantically different variant.

The guard IS present (the input is rejected); only the error type diverges. A caller matching on the specific variant receives different results depending on which entry point they use. This is a known inconsistency, not a silent-failure risk.

**Why not refactored this phase:** Per user constraint in 04-RESEARCH.md, V-04 is document-only for this milestone. The fix (aligning the two paths to return the same variant) is low-effort (Value=2, Effort=2, Risk=2, Priority=1.6) and sits at Rank 10 in the open table.

**Recommended next action:** In a follow-up robustness cycle, either (a) have `VARForecaster::fit()` call `VAR::fit()` internally (so the raw-vec guard is the canonical path), or (b) add an explicit `InvalidParameter` guard at the `VARForecaster` entry that fires before `validate_series_complete()`. Option (a) is lower risk.

---

## Accuracy Gap Detail: ACC-01 Handoff

**Root cause (confirmed by 04-02 investigation):** After the L4 harness fix (period=12), the remaining MASE gap of +0.0290 (0.8923 vs reference 0.8633) reflects a genuine small algorithmic difference in seasonal model selection between anofox-forecast and statsforecast. Two untried low-risk source-level levers remain:

| # | Lever | Location | Expected Gap Closure | Risk |
|---|-------|----------|---------------------|------|
| L1 | Lower seasonal minimum from 3×period to 2×period | `src/models/exponential/auto_ets.rs:427` | MEDIUM — brings 24–35 obs monthly series into seasonal pool (matches statsforecast behavior) | Low (affects only short-series threshold; no algorithmic change) |
| L2 | Relax or remove F-ratio seasonal gate | `src/models/exponential/auto_ets.rs:430-471` | MEDIUM — some series suppressed by the gate may fit better seasonally | Medium (gate touches existing seasonal-selection logic; test impact unknown) |

**Recommended approach:** Try L1 first (lower risk, single threshold change). Measure MASE delta via ACCUR-08 anchor. If MASE ≤ 0.8833, emit `accuracy.json`:
```bash
ANOFOX_DATASET_DIR=/path/to/m3/data ANOFOX_WRITE_ACCURACY_BASELINE=1 \
  cargo test --package anofox-bench-harness --test accuracy -- --nocapture
```
If L1 does not close the gap, apply L2 and re-measure. Do NOT force-write `accuracy.json` — the emit function's internal guard panics if the anchor fails.

**Entry condition for locking accuracy.json:** Re-run the anchor:
```bash
ANOFOX_DATASET_DIR=/path/to/m3/data \
  cargo test --package anofox-bench-harness \
  --test accuracy accur08_anchor_m3_monthly_autoets \
  -- --nocapture 2>&1 | grep -E "ACCUR-08|MASE|PASSED|FAILED"
```
MASE must be ≤ 0.8833 (reference 0.8633 + tolerance 0.02). Commit `accuracy.json` as a separate deliberate change (IMPR-03 discipline).

---

## Dimension Coverage Summary

| # | Dimension | Baseline File | Status | Open Items | Notes |
|---|-----------|---------------|--------|------------|-------|
| 1 | Code Coverage | `coverage.json` | Real — 91.44%, floor 90.4% | G-02, G-04..G-10 (9 rows) | Post-04-01 ratchet; V-01/V-02/V-03/G-01 landed |
| 2 | WASM Binary Size | `wasm_size.json` | Real — 2,838,958 bytes | WSZ-01 | No reduction measured yet; audit needed |
| 3 | Peak Memory (dhat) | `dhat.json` | Real — auto_ets=290,440 bytes peak | MEM-01 | auto_ets is outlier vs other families |
| 4 | Reference Accuracy | `statsforecast_reference.json` | Real — M3 monthly=0.8633, quarterly=1.1436, yearly=2.6954 | ACC-01 (the gap vs reference) | Reference locked; anofox side still deferred |
| 5 | Accuracy (anofox) | `accuracy.json` | ABSENT/DEFERRED — anchor failed | ACC-01 | After L1/L2 levers, lock when anchor passes |
| 6 | Instruction Count (iai) | `iai.json` | Placeholder — all zeros | PERF-IAI-01..03 | Requires Valgrind ≥3.20 machine |
| 7 | Wall-clock (criterion) | `criterion.json` | Placeholder — all 0.0 ns | PERF-CRT-01..03 | Requires quiet machine; 14 benchmark entries total |
| 8 | Numerical Robustness | `03-GAP-INVENTORY.md` | Real — 20 rows inventoried | V-04, G-02, G-04..G-10, A-01..A-05 | V-01/V-02/V-03/G-01 closed this phase |
