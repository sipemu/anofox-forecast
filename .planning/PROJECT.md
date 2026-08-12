# anofox-forecast — Performance & Validation Hardening

## What This Is

`anofox-forecast` is a mature Rust time-series forecasting library (v0.15.8) with 30+ models
behind a common `Forecaster` trait, seasonal decomposition, feature extraction, uncertainty
quantification, and WebAssembly/JS bindings published as `@sipemu/anofox-forecast`.

This project is a **proactive, whole-library performance and validation hardening cycle**: for
each quality dimension, stand up a repeatable measurement harness, capture baseline numbers,
produce a prioritized improvement backlog, then land the highest-value fixes — each proven with a
before/after delta. It is aimed at the maintainers of the library, not a new end-user feature.

## Core Value

**Every claimed capability is measured, and every improvement is proven with a before/after
number.** No unquantified "it feels faster" or "it looks more accurate" — measurement is the
backbone that makes the improvements trustworthy and non-regressing.

## Current State

**Shipped: v1.0 — Performance & Validation Hardening (2026-08-12).** Phases 1–4 complete, 28/28
requirements satisfied, milestone audit INTEGRATED. Delivered: committed measurement baselines per
dimension (speed/memory/WASM-size/coverage) with CI guards; a statistically correct accuracy harness
(competition MASE, Naive2, Diebold-Mariano, pinned statsforecast cross-library reference);
numerical-robustness edge-case + property suites with per-family NaN/Inf guards; a CI-enforced
coverage floor (90.4%); and a ranked improvement backlog with top-value fixes landed (each proven by a
before/after delta).

**Open (carried to next milestone, see `baselines/BACKLOG.md`):** #1 AutoETS M3-monthly accuracy gap
(period=12 fix cut MASE 1.0452→0.8923; still +0.0290 above the 0.8633 reference anchor, so
`accuracy.json` remains deferred/unlocked); iai/criterion baselines await manual hardware capture;
optional Nyquist validation for Phases 3–4.

## Next Milestone Goals

To be defined via `/gsd-new-milestone`. The v1.0 backlog is the natural seed — the AutoETS accuracy
gap is the highest-value candidate.

## Requirements

### Validated

<!-- Inferred from the existing codebase (see .planning/codebase/). These already ship and are relied upon. -->

- ✓ 30+ forecasting models under a unified `Forecaster` trait (ARIMA, ETS, Theta, TBATS, GARCH, baseline, intermittent, VAR/Kalman/hierarchical, ensemble, Laplace) — existing
- ✓ Seasonal decomposition & trend extraction (STL, MSTL, Fourier, HP, Hamilton, polynomial) — existing
- ✓ Feature extraction, forecastability metrics, outlier/changepoint/anomaly detection — existing
- ✓ Validation & postprocessing: cross-validation, bootstrap, conformal prediction, calibration, quantile methods — existing
- ✓ Batch multi-series forecasting with Rayon parallelism (`parallel` feature) — existing
- ✓ WebAssembly/JS bindings + published npm package + browser playground on GitHub Pages — existing
- ✓ Existing quality gates: criterion benchmarks, proptest, clippy `-D warnings`, cargo-audit/deny — existing
- ✓ Compute-speed benchmark harness with captured baselines across model families (criterion wall-clock + iai-callgrind instruction gates; native `parallel` and single-thread profiles) — Phase 1
- ✓ Memory & WASM-size measurement (dhat peak-allocation tests + committed `wasm_size.json`; PERF-06 dead-code cleanup before size baseline) — Phase 1

### Active

<!-- The hardening goals for this cycle. Hypotheses until shipped and validated. -->

**Measurement backbone (per dimension: harness → baseline → tracked over time)**
- [ ] Forecast-accuracy harness over vendored standard competition datasets (M-competitions, Tourism, etc.) with standard metrics (MASE, RMSE, sMAPE)
- [ ] Numerical-robustness test suite for edge cases (near-singular matrices, convergence limits, NaN/Inf, extreme scales)
- [ ] Statistical-methodology validation: correctness of CV splits, interval/conformal coverage checks
- [ ] Code-correctness & coverage baseline: test-coverage measurement + gap identification across model families
- [ ] Input-robustness suite: missing values, too-short series, wrong/irregular frequency, empty/constant input
- [ ] Cross-reference benchmark comparing accuracy/behavior against a reference implementation on shared datasets

**Prioritization & improvement**
- [ ] Consolidated, prioritized improvement backlog ranking findings across all 8 dimensions by value/effort
- [ ] Land the highest-value improvements, each with a documented before/after delta and a guard against regression

### Out of Scope

- New forecasting models or model families — this is a hardening cycle, not feature expansion
- New automatic seasonal-period-detection integration into models — deliberately excluded
- API/breaking redesigns of the public `Forecaster` trait — improvements must stay backward-compatible unless a fix demands otherwise (logged as a Key Decision if so)
- New Python bindings — out of scope for this cycle
- Playground/UI feature work beyond what's needed to measure WASM size/perf

## Context

- **Codebase map is current** — see `.planning/codebase/` (ARCHITECTURE, STACK, STRUCTURE, CONVENTIONS, TESTING, INTEGRATIONS, CONCERNS), refreshed 2026-08-09.
- **Existing measurement surface to build on:** criterion (benchmarking), proptest (property tests), wasm-bindgen-test, cargo-audit/deny. Baselines should extend these rather than replace them.
- **Known architectural anti-patterns already documented** (ARCHITECTURE.md): occasional `.unwrap()`/`.expect()` in internal helpers, a few `fit()` paths skipping `validate_series_complete()`, some mutable-state leakage, duplicate seasonality detection between AutoETS/AutoARIMA. These are prime robustness/correctness backlog candidates.
- **WASM constraint:** the `wasm32-unknown-unknown` target forbids the `parallel` (Rayon) feature — compute-speed and memory work must account for both native (parallel) and WASM (single-threaded) profiles.
- **Reference data:** standard public competition datasets to be vendored/fetched as part of the accuracy harness.

## Constraints

- **Tech stack**: Rust 2021, stable toolchain (CI also tests beta/nightly); WASM via wasm-pack — measurement tooling must fit this toolchain.
- **Compatibility**: Public `Forecaster` API stays backward-compatible; the published npm package `@sipemu/anofox-forecast` must keep building.
- **Performance target philosophy**: improvements are only "done" when backed by a reproducible before/after measurement; no unmeasured optimizations.
- **Feature gates**: work must respect existing feature flags (`distributional`, `postprocess`, `anomaly`, `forecastability`, `seasonal-detection`, `parallel`, `serde`, `js`) and their WASM restrictions.
- **CI hygiene**: clippy `-D warnings` and cargo-audit/deny gates must stay green.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Measurement-first: build harnesses & baselines before improving | Can't prove improvement (or catch regressions) without a baseline; whole-library scope demands it | ✓ Validated — Phase 1 stood up criterion/iai/dhat/wasm-size harnesses with committed baselines and CI read-only gates |
| CI reads baselines but never overwrites them (MEAS-01); criterion/iai captured locally, not on Actions | Wall-clock noise on shared CI runners makes committed baselines untrustworthy; separation keeps numbers reproducible | ✓ Applied — Phase 1 (`wasm-size.yml`/`bench.yml` gate-only; `scripts/update_*.sh` for local capture) |
| iai.json/criterion.json committed as structural placeholders pending maintainer hardware capture | Dev machine lacks valgrind ≥ 3.20; criterion needs a quiet machine — schema/gates land first, numbers populated later | ✓ Accepted — Phase 1 UAT confirmed harness complete; numeric population is a documented maintainer step |
| Whole-library scope across all 8 dimensions | User wants systematic, proactive confidence — not a point fix | — Pending |
| Use standard competition datasets as the accuracy corpus | Industry-recognized, comparable to reference libraries, avoids synthetic-only bias | — Pending |
| Exclude new models / API redesigns / auto period-detection integration | Keeps the cycle a hardening pass, not feature creep | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-10 after Phase 1*
