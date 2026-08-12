# Phase 3: Coverage Gap Inventory

**Generated:** 2026-08-11
**Coverage at time of inventory:** 91.30% line coverage (70,982 / 77,743 lines)
**Tool:** cargo-llvm-cov 0.8.4 — `cargo llvm-cov --package anofox-forecast --all-features --text --output-path /tmp/coverage_report.txt`
**Baseline ref:** `.planning/baselines/coverage.json` — ratchet floor 90.3%

> Structured for Phase 4 backlog. Each row = one actionable improvement item.
> **Priority rubric:**
> - **P1** = correctness risk: missing input guard, silent NaN propagation, panic-able path, or uncovered error path
> - **P2** = coverage gap: function/branch hit rate below ~50% with no structural reason; improvement adds confidence
> - **P3** = assertion density: test exists and passes but asserts nothing meaningful (no-panic only); weak signal

---

## Uncovered Paths

| # | File | Function / Line range | Missing invariant | Priority | Notes |
|---|------|-----------------------|-------------------|----------|-------|
| G-01 | `src/models/theta/global_theta.rs` | `fn new()` (line 43), `fn with_theta()` (line 53), `fn alpha()` (line 61), `fn fit()` (line 66), `fn predict()` (line 113) | Entire file 0% covered — no test exercises GlobalTheta model at all | P1 | 0% total coverage; `fit()` has length guard but no NaN guard on raw `Vec<f64>` elements; see V-03 in Missing-Validation section |
| G-02 | `src/seasonality/traits.rs` | `fn detect_changepoints()` (line 212), `fn default()` (lines 32, 113), `fn auto()` (line 122), `fn resolve_with_data()` (line 182) | Changepoint-integrated season detection entirely uncovered; `resolve_with_data()` 21% covered | P2 | `detect_changepoints` uses `fdars-core` under `seasonal-detection` feature; no test calls the auto-resolution path |
| G-03 | `src/monitor/sequential_table.rs` | `fn regenerate_crit_val_table()` (line 403) | Intentionally `#[ignore]` test — not a real gap | P3 | Marked `#[ignore = "regeneration utility"]`; excluded from P1/P2; noted here for completeness |
| G-04 | `src/models/inspect.rs` | All public `Explanation*` / `ModelInspector` methods (file 15.4% overall) | Explanation/inspector coverage based on file header mismatch — actual covered functions are test helpers; production impl methods uncovered | P2 | The covered lines are mostly test functions at the bottom; the `impl Inspectable` methods are not tested for most models |
| G-05 | `src/changepoint/detector.rs` | `fn predict_n_bkps()` (line 100), `fn predict_eps()` (line 110), `fn fit_predict_n_bkps()` (line 124), `fn min_size()` (line 26) | Alternative `ChangeDetector` dispatch methods never called from test suite | P2 | Default `predict_n_bkps` and `predict_eps` return `Err(Unsupported)` — important error path not exercised |
| G-06 | `src/anomaly/quantile.rs` | Quantile anomaly scoring functions (file 34.2%) | Missing invariant: no test covers anomaly score output range `[0, 1]` or edge case on constant series | P2 | `anomaly` feature is gated `cfg(feature = "anomaly")` but `--all-features` enables it; low coverage indicates the anomaly scoring subsystem is under-tested |
| G-07 | `src/models/smart.rs` | `SmartForecaster` dispatch logic (file 34.4%) | Many routing branches (retail vs continuous, AID class boundaries) not covered — only happy-path exercised | P2 | Smart forecaster's family-classification and fallback logic represents product-level risk if misrouted; edge-case coverage needed for the AID classification thresholds |
| G-08 | `src/models/laplace/gpd_tails.rs` | GPD tail distribution edge paths (file 42.3%) | Missing: corner cases in tail shape parameter `xi` estimation (xi < 0, xi > 1 regimes) not tested | P2 | Laplace leaf model; GPD with xi ≈ 0 should degenerate to exponential; no test checks this boundary |
| G-09 | `src/batch.rs` | `batch::*` parallelisation paths (file 42.2%) | Missing: error aggregation paths when some series fail inside batch run — no test verifies partial-failure handling returns correct error vector length | P2 | Batch functions return `Vec<Result<_>>`; uncovered paths suggest the Err arm is untested |
| G-10 | `src/postprocess/cqr.rs` | CQR calibration edge paths (file 43.0%) | Missing: calibration on very small calibration sets (n < q) and boundary alpha values | P2 | Conformalized quantile regression; boundary behaviour on short calibration sets is safety-critical for prediction interval validity |

---

## Assertion-Free Tests

> Tests that exist and pass but contain no `assert!`, `prop_assert!`, `panic!`, or calls to helpers that assert — they prove no-panic but assert no invariant.
> **Note:** Tests using `assert_*` helpers or `panic!()` as their assertion mechanism are NOT listed here (they do assert; the scanning heuristic was too broad).

| # | File | Test name | Missing assertion | Priority |
|---|------|-----------|-------------------|----------|
| A-01 | `tests/auto_arima_order_selection.rs` | `order_selection_comparison` | Prints order comparison table but asserts nothing; no check that Rust and Python orders agree or that forecast is finite | P3 |
| A-02 | `tests/auto_arima_validation.rs` | `arima_210_parameter_comparison` | Prints AR coefficients and intercept but asserts nothing about them (no tolerance check vs Python statsmodels reference) | P3 |
| A-03 | `tests/autoarima_outlier_investigation.rs` | `investigate_outliers` | Investigation / diagnostic test — no assertion on outlier detection accuracy or output shape | P3 |
| A-04 | `tests/batch_validation.rs` | `benchmark_auto_ets_batch_vs_individual` | Prints timing speedup but asserts nothing (no `assert!(speedup > 1.0)`, no success-rate check, no forecast-value check) | P3 |
| A-05 | `tests/batch_validation.rs` | `benchmark_global_ets_ann_vs_individual` | Same pattern — timing + print, no assertion on output correctness or any speedup bound | P3 |

**Note on `issue_106_decomposable_conformance.rs`:** Tests in that file call `check_conformance()` which contains `assert_eq!` and `assert!` — they ARE asserting. Similarly, `m4_daily_accuracy_regression.rs` tests use `panic!()` as their assertion, and `statsforecast_comparison.rs` tests call `assert_forecasts_match()`. These were initial false positives from a broad scan and are excluded from the P3 list above.

---

## Missing-Validation P1 Items

> `fit()` paths missing NaN/Inf guards on raw `Vec<f64>` input — filed as P1 for Phase 4.
> These models use a `&[Vec<f64>]` (panel/matrix) API rather than `&TimeSeries`, so
> `validate_series_complete()` does not apply. They require a per-element NaN/Inf scan.

| # | File | Function | Risk | Recommended fix |
|---|------|----------|------|-----------------|
| V-01 | `src/models/exponential/global_ets.rs` | `GlobalETS::fit(&mut self, all_series: &[Vec<f64>])` (line 81) | NaN/Inf propagates silently from raw `Vec<f64>` elements into GlobalETS fitted parameters (`alpha`, `beta`, seasonal state) without an explicit per-element guard; the only guard is `all_series.is_empty()` (line 82) and a length check. NaN in any series element is passed directly to the optimiser. | Add a per-series, per-element scan before the estimation loop: `for (i, s) in all_series.iter().enumerate() { if s.iter().any(\|v\| v.is_nan() \|\| v.is_infinite()) { return Err(ForecastError::InvalidParameter(format!("series {} contains NaN/Inf", i))); } }` |
| V-02 | `src/models/intermittent/global_croston.rs` | `GlobalCroston::fit(&mut self, all_series: &[Vec<f64>])` (line 76) | Same risk: NaN/Inf from any panel series propagates into GlobalCroston's shared alpha/beta parameters. Guard on line 77 checks only `is_empty()`, not element values. | Same per-series NaN/Inf scan pattern as V-01 |
| V-03 | `src/models/theta/global_theta.rs` | `GlobalTheta::fit(&mut self, all_series: &[Vec<f64>])` (line 66) | Same risk as V-01/V-02 PLUS the entire `GlobalTheta` model is at **0% test coverage** — the NaN path has never been exercised. Even the `is_empty()` guard on line 67 has never been hit. | Per-series NaN/Inf scan + add at least a smoke test that the constructor and a basic fit/predict cycle work on valid data |
| V-04 | `src/models/var.rs` | `VAR::fit(&mut self, data: &[Vec<f64>])` (line 96) | `VAR::fit()` does check per-variable NaN/Inf at lines 119-124 (`series.iter().any(\|v\| v.is_nan() \|\| v.is_infinite())` → `InvalidParameter`). **However, `VARForecaster::fit()` in `var_forecaster.rs` routes through `validate_series_complete()` (on the `TimeSeries` wrapper) but does NOT call `VAR::fit()`'s NaN guard separately** — the `TimeSeries` validator returns `MissingValues` variant rather than the `VAR`-specific `InvalidParameter("Variable N contains NaN")`. The guard is present but the two error variants diverge in type; the `VAR::fit()` raw-vec NaN path is semantically different from the `VARForecaster` path. | Lower priority than V-01–V-03 (guard IS present); consider documenting the divergent error variant as a known inconsistency in Phase 4 |

---

## Additional P1 Rows: Panicking Models and Proptest Counterexamples

### Panicking Models (from 03-01 SUMMARY)

**None.** Plan 03-01 found **zero panicking models**. Every representative model family (`Naive`, `AutoETS`, `ARIMA(1,0,1)`, `Theta`, `TBATS`, `Croston`, `MSTLForecaster`, `GARCH`, `VARForecaster`, `LaplaceForecaster`) returned `Err(_)` on all malformed inputs (NaN, Inf, empty, n=2, extreme-scale). No `// GAP:` annotations were added to `tests/edge_case_robustness.rs`.

### Proptest Counterexamples (from 03-02 SUMMARY)

**None.** Plan 03-02 found **zero proptest counterexamples**. All six property tests across changepoint metrics, MSTL decomposition, and CV fold generator passed across all randomly generated inputs. No shrunk failing cases were produced. `.proptest-regressions/` contains only the `.gitkeep` placeholder.

---

## Summary Row Counts for Phase 4 Backlog

| Priority | Count | Source |
|----------|-------|--------|
| P1 — Missing validation (raw-vec) | 4 rows (V-01 to V-04) | 03-01 deferred handoff |
| P1 — Panicking models | 0 | 03-01 found none |
| P1 — Proptest counterexamples | 0 | 03-02 found none |
| P1 — 0%-coverage model | 1 (G-01: GlobalTheta) | llvm-cov text report |
| P2 — Coverage gaps | 9 rows (G-02, G-04 to G-10) | llvm-cov text report |
| P3 — Assertion-free tests | 5 rows (A-01 to A-05) | grep/python scan |
| P3 — Ignored test utility | 1 (G-03) | llvm-cov text report |
| **Total** | **20 rows** | — |

**Phase 4 recommended starting point:** V-01 through V-03 (raw-vec NaN guards) and G-01 (GlobalTheta 0% coverage) are the highest-value targets — they address correctness risk and the largest coverage void simultaneously.
