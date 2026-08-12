# Requirements: anofox-forecast — Performance & Validation Hardening

**Defined:** 2026-08-09
**Core Value:** Every claimed capability is measured, and every improvement is proven with a before/after number.

## v1 Requirements

Requirements for this hardening milestone. Each maps to a roadmap phase. All measurement
requirements follow the pattern: build harness → capture committed baseline → guard against
regression.

### Measurement Infrastructure (MEAS)

- [x] **MEAS-01**: A committed baseline store exists at `.planning/baselines/` holding JSON baselines per dimension; CI reads baselines but never writes them
- [x] **MEAS-02**: A maintainer can capture/refresh baselines via documented `scripts/update_*.sh` on a quiet local machine
- [x] **MEAS-03**: CI workflows exist for benchmarking (`bench.yml`) and WASM size (`wasm-size.yml`); the accuracy workflow (`accuracy.yml`) is `workflow_dispatch`-only and never gates PR merges
- [x] **MEAS-04**: All measurement code lives in `benches/`, `tests/`, `scripts/`, and a harness crate — nothing new is added to library `src/`

### Compute Performance (PERF)

- [x] **PERF-01**: A criterion benchmark suite covers fit + predict across model families (ARIMA, ETS, Theta, baseline, intermittent, ensemble, Laplace), single-series and batch, with committed local baselines
- [x] **PERF-02**: iai-callgrind instruction-count gates run in CI on 3–5 critical hot paths (e.g., AutoETS fit, ARIMA fit, batch-100)
- [x] **PERF-03**: Native-parallel and WASM/single-thread (no-Rayon) profiles are measured and reported separately
- [x] **PERF-04**: A native peak-memory measurement (dhat) asserts bounds for major model families
- [x] **PERF-05**: The compiled release WASM binary size is tracked against a committed baseline with a delta threshold in CI
- [x] **PERF-06**: Known WASM dead code (unused `inner()` methods, unused imports in `crates/anofox-forecast-js/`) is removed before the WASM size baseline is locked

### Accuracy & Statistical Methodology (ACCUR)

- [x] **ACCUR-01**: A dataset loader reads the local `validation/data/` corpus (Monash `.tsf` + CSV) for M3, M4 sample, Tourism, NN5, gated on the `ANOFOX_DATASET_DIR` env var
- [x] **ACCUR-02**: The accuracy harness uses expanding-window (rolling-origin) CV with per-fold temporal-integrity assertions (`train_end < test_start`); no in-sample/`fitted_values()` metrics ever appear in accuracy tables
- [x] **ACCUR-03**: MASE is computed with the correct seasonal denominator and a guard against denominator collapse on intermittent/constant series (no silent NaN/Inf in aggregates)
- [x] **ACCUR-04**: sMAPE, RMSE, and MAE are implemented/verified with correct denominators
- [x] **ACCUR-05**: MSIS and an empirical interval-coverage check are implemented for prediction intervals
- [x] **ACCUR-06**: A Naive2 baseline (autocorrelation-gated seasonal/non-seasonal) is available as the accuracy reference
- [x] **ACCUR-07**: Accuracy is reported with per-frequency stratification (never a single cross-frequency aggregate)
- [x] **ACCUR-08**: The committed accuracy baseline is validated against a published reference (statsforecast AutoETS M3 monthly, MASE ≈ 0.93) before being locked

### Numerical Robustness & Input Handling (ROBUST)

- [x] **ROBUST-01**: An edge-case input suite covers constant series, n=2, all-zeros/intermittent, NaN/Inf-containing, zero-length, and extreme-scale inputs — each asserting the correct `ForecastError` variant (or valid output), never a panic
- [x] **ROBUST-02**: `fit()` paths are audited so that `validate_series_complete()` (or equivalent) runs before parameter estimation across model families
- [x] **ROBUST-03**: Property-based tests (proptest) cover known-fragile areas (e.g., changepoint metrics, MSTL, CV boundary conditions) asserting no-panic / no-NaN invariants

### Code Correctness & Coverage (COVER)

- [x] **COVER-01**: A code-coverage baseline is captured via cargo-llvm-cov and committed; a coverage floor is enforced in CI
- [x] **COVER-02**: A gap inventory identifies uncovered paths and assertion-free tests, filed as improvement-backlog candidates (assertion density, not just line %)

### Reference-Library Benchmarking (BENCH)

- [x] **BENCH-01**: A documented cross-library comparison runs anofox-forecast against a reference implementation (via the existing `run_statsforecast.py`) on shared datasets, horizons, and preprocessing
- [x] **BENCH-02**: Any "model A beats model B" / "beats reference" claim is gated by a Diebold–Mariano significance test when the accuracy gap is < 5%

### Improvement Delivery (IMPR)

- [x] **IMPR-01**: A consolidated improvement backlog ranks findings across all 8 dimensions by value vs. effort using real baseline numbers
- [x] **IMPR-02**: The highest-value improvements are landed, each with a documented before/after delta in the relevant baseline file and a regression guard
- [x] **IMPR-03**: Baseline updates after an improvement are committed deliberately (separate change), never auto-written by CI

## v2 Requirements

Deferred to a future cycle. Tracked but not in this roadmap.

### Probabilistic Depth (PROB)

- **PROB-01**: CRPS for distributional outputs (LaplaceForecaster, BootstrapPredictor)
- **PROB-02**: Pinball loss at target quantiles for QRA/conformal outputs
- **PROB-03**: PIT histogram / KS calibration test for Gaussian outputs
- **PROB-04**: Conditional coverage stratified by series characteristics

### Extended Accuracy (XACC)

- **XACC-01**: OWA (M4 primary ranking metric) once Naive2 is validated
- **XACC-02**: RMSSE for intermittent/retail (Croston/IMAPA/TSB/ADIDA) families
- **XACC-03**: Per-horizon accuracy decomposition (h=1, h=1–3, h=4–6, full)
- **XACC-04**: Full M4 corpus run (100K series) as an opt-in CI job

### Extended Robustness (XROB)

- **XROB-01**: Mutation-score tracking alongside line coverage
- **XROB-02**: WASM runtime memory profiling (no in-process equivalent of dhat today)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New forecasting models / families | This is a hardening cycle, not feature expansion |
| Public `Forecaster` API redesign / breaking changes | Must stay backward-compatible unless a fix demands it (log as Key Decision) |
| New automatic seasonal-period-detection integration into models | Explicitly out of scope; also non-reproducible inside benchmark evaluation |
| New Python bindings | Out of scope this cycle |
| Playground/UI feature work | Only WASM-size/perf measurement touches this area |
| MAPE on zero-containing series, R², single hold-out accuracy | Anti-metrics — misleading for intermittent-demand use cases; never add or claim |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MEAS-01 | Phase 1 | Complete |
| MEAS-02 | Phase 1 | Complete |
| MEAS-03 | Phase 1 | Complete |
| MEAS-04 | Phase 1 | Complete |
| PERF-01 | Phase 1 | Complete |
| PERF-02 | Phase 1 | Complete |
| PERF-03 | Phase 1 | Complete |
| PERF-04 | Phase 1 | Complete |
| PERF-05 | Phase 1 | Complete |
| PERF-06 | Phase 1 | Complete |
| ACCUR-01 | Phase 2 | Complete |
| ACCUR-02 | Phase 2 | Complete |
| ACCUR-03 | Phase 2 | Complete |
| ACCUR-04 | Phase 2 | Complete |
| ACCUR-05 | Phase 2 | Complete |
| ACCUR-06 | Phase 2 | Complete |
| ACCUR-07 | Phase 2 | Complete |
| ACCUR-08 | Phase 2 | Complete |
| BENCH-01 | Phase 2 | Complete |
| BENCH-02 | Phase 2 | Complete |
| ROBUST-01 | Phase 3 | Complete |
| ROBUST-02 | Phase 3 | Complete |
| ROBUST-03 | Phase 3 | Complete |
| COVER-01 | Phase 3 | Complete |
| COVER-02 | Phase 3 | Complete |
| IMPR-01 | Phase 4 | Complete |
| IMPR-02 | Phase 4 | Complete |
| IMPR-03 | Phase 4 | Complete |

**Coverage:**

- v1 requirements: 28 total (note: original count of 27 was a typo; actual count is 28 across MEAS×4 + PERF×6 + ACCUR×8 + ROBUST×3 + COVER×2 + BENCH×2 + IMPR×3)
- Mapped to phases: 28/28
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-09*
*Last updated: 2026-08-09 after roadmap creation — traceability populated*
