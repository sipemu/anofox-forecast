# Roadmap: anofox-forecast — Performance & Validation Hardening

**Created:** 2026-08-09
**Granularity:** Standard
**Coverage:** 28/28 v1 requirements mapped

## Core Value

Every claimed capability is measured, and every improvement is proven with a before/after number.

## Phases

- [ ] **Phase 1: Measurement Infrastructure & Compute Baselines** - Stand up CI workflows, committed baseline store, and capture speed/memory/WASM-size baselines (after dead-code cleanup)
- [ ] **Phase 2: Accuracy Harness & Statistical Methodology** - Build dataset loader, implement correct metrics, validate against reference, commit accuracy and cross-library baselines
- [ ] **Phase 3: Numerical Robustness & Coverage Baseline** - Add edge-case and property-based tests, capture coverage baseline with CI floor, document gap inventory
- [ ] **Phase 4: Prioritized Improvement Backlog & Top-Value Fixes** - Rank all findings by value/effort using real baseline numbers, land highest-value improvements with before/after deltas

## Phase Details

### Phase 1: Measurement Infrastructure & Compute Baselines

**Goal**: The measurement backbone exists — CI workflows run, the baseline store is initialized, and every compute/memory/WASM-size number is committed and trustworthy (dead code removed before the size baseline is locked)

**Depends on**: Nothing (first phase)

**Requirements**: MEAS-01, MEAS-02, MEAS-03, MEAS-04, PERF-01, PERF-02, PERF-03, PERF-04, PERF-05, PERF-06

**Success Criteria** (what must be TRUE):

  1. `.planning/baselines/` contains committed JSON baseline files for criterion, iai-callgrind, dhat peak-memory, and WASM binary size — and CI reads them but never overwrites them
  2. `bench.yml` and `wasm-size.yml` GitHub Actions workflows are green; iai-callgrind instruction-count gates block regressions on ubuntu CI
  3. A maintainer can reproduce any baseline by running the documented `scripts/update_*.sh` on a local machine — no special access or secret env vars required
  4. Known WASM dead code (unused `inner()` methods, unused imports in `crates/anofox-forecast-js/`) is removed and confirmed absent before the WASM size baseline is committed
  5. Native-parallel and WASM/single-thread benchmark profiles are reported in separate sections; criterion baselines are captured locally (not on GitHub Actions) to avoid wall-clock noise

**Plans**: 3 plans
**Wave 1**

- [ ] 01-01-PLAN.md — Tracer: harness crate + D-02 schema, PERF-06 dead-code cleanup, WASM-size baseline + wasm-size.yml gate (MEAS-01..04, PERF-05, PERF-06)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 01-02-PLAN.md — iai-callgrind instruction gate: iai_suite bench, update_iai.sh, bench.yml (PERF-02)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 01-03-PLAN.md — criterion + dhat dimensions: baseline_suite, dhat_peak tests, update_criterion.sh/update_dhat.sh (PERF-01, PERF-03, PERF-04)

---

### Phase 2: Accuracy Harness & Statistical Methodology

**Goal**: A statistically correct accuracy harness produces trustworthy per-frequency numbers — validated against a published reference result before any baseline is locked — and a documented cross-library comparison is committed

**Depends on**: Phase 1

**Requirements**: ACCUR-01, ACCUR-02, ACCUR-03, ACCUR-04, ACCUR-05, ACCUR-06, ACCUR-07, ACCUR-08, BENCH-01, BENCH-02

**Success Criteria** (what must be TRUE):

  1. The dataset loader reads `validation/data/` (M3, M4 sample, Tourism, NN5) when `ANOFOX_DATASET_DIR` is set; the `accuracy.yml` workflow is `workflow_dispatch`-only and never blocks a PR merge
  2. MASE, sMAPE, RMSE, MAE, MSIS, and empirical interval coverage are implemented with correct denominators and a MASE denominator-collapse guard; no NaN or Inf appears silently in any aggregate metric table
  3. The accuracy harness uses expanding-window CV with temporal-integrity assertions (`train_end < test_start`) in every fold; `fitted_values()` is never used as a proxy for out-of-sample accuracy
  4. Accuracy is reported with per-frequency stratification (Yearly/Quarterly/Monthly/Weekly/Daily/Hourly); the committed `baselines/accuracy.json` matches published statsforecast AutoETS M3 monthly MASE ≈ 0.93 before being locked
  5. The cross-library comparison against the reference implementation runs on shared datasets/horizons/preprocessing; any accuracy-gap claim under 5% is gated by a Diebold-Mariano significance test

**Plans**: TBD

---

### Phase 3: Numerical Robustness & Coverage Baseline

**Goal**: Every model family handles malformed and edge-case inputs with a correct `ForecastError` (never a panic), a coverage baseline is committed with a CI floor enforced, and a gap inventory identifies the highest-priority uncovered paths

**Depends on**: Phase 2

**Requirements**: ROBUST-01, ROBUST-02, ROBUST-03, COVER-01, COVER-02

**Success Criteria** (what must be TRUE):

  1. An edge-case input suite covers constant series, n=2, all-zeros/intermittent, NaN/Inf-containing, zero-length, and extreme-scale inputs — each test asserts the correct `ForecastError` variant and no test triggers a panic
  2. All `fit()` paths across model families are confirmed to call `validate_series_complete()` (or equivalent) before parameter estimation; any missing call is either fixed or filed as a P1 improvement-backlog item
  3. Proptest property-based tests cover known-fragile areas (changepoint metrics, MSTL decomposition, CV boundary conditions) asserting no-panic and no-NaN invariants across random inputs
  4. A code-coverage baseline is committed to `baselines/coverage.json` via cargo-llvm-cov and a coverage floor is enforced in CI — CI fails if coverage drops below the floor
  5. A gap inventory lists uncovered paths and assertion-free tests with enough detail to serve as improvement-backlog input (file + function + missing invariant)

**Plans**: TBD

---

### Phase 4: Prioritized Improvement Backlog & Top-Value Fixes

**Goal**: Real baseline numbers from all three preceding phases drive a ranked backlog; the highest-value improvements are landed, each proven by a before/after delta in the relevant baseline file, with tightened regression guards

**Depends on**: Phase 3

**Requirements**: IMPR-01, IMPR-02, IMPR-03

**Success Criteria** (what must be TRUE):

  1. A consolidated improvement backlog ranks findings from all 8 measurement dimensions by value/effort using the actual baseline numbers captured in Phases 1–3; the ranking rationale is documented
  2. Each landed improvement has a documented before/after delta in the relevant baseline file (e.g., MASE delta in `accuracy.json`, instruction-count delta in the iai baseline, size delta in `wasm_size.json`)
  3. Baseline updates after an improvement are committed in a separate, deliberate change — never auto-written by CI; updated regression guards in CI reflect the new tighter thresholds

**Plans**: TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Measurement Infrastructure & Compute Baselines | 0/? | Not started | - |
| 2. Accuracy Harness & Statistical Methodology | 0/? | Not started | - |
| 3. Numerical Robustness & Coverage Baseline | 0/? | Not started | - |
| 4. Prioritized Improvement Backlog & Top-Value Fixes | 0/? | Not started | - |

---
*Roadmap created: 2026-08-09*
*Last updated: 2026-08-09 after initial creation*
