# Project Research Summary

**Project:** anofox-forecast — Performance & Validation Hardening
**Domain:** Rust time-series forecasting library — measurement harness, baseline capture, and targeted improvement
**Researched:** 2026-08-09
**Confidence:** MEDIUM

## Executive Summary

`anofox-forecast` is a mature (v0.15.8) Rust forecasting library with 30+ models, existing criterion benchmarks, and a published WASM/npm package. This hardening cycle is not a feature expansion — it is a systematic effort to make every capability claim measurable and every improvement provable. The research confirms a clear build order: measurement infrastructure must come before any optimization claim, because a baseline that was built on statistical or tooling mistakes corrupts everything downstream. The highest-risk work in this entire cycle is getting the statistical methodology right in the accuracy harness; a harness that uses in-sample metrics, shuffled folds, or a zero-denominator MASE will produce plausible-looking but meaningless numbers.

The recommended tooling stack pairs criterion (local, developer-facing, wall-clock) with iai-callgrind (CI, deterministic instruction counts) for speed, dhat for native allocation assertions, cargo-llvm-cov for coverage, and twiggy/wasm-opt for WASM size. Every measurement dimension gets a committed JSON baseline in `.planning/baselines/` — CI reads baselines but never writes them; maintainers update them deliberately after proven improvements. Two important local assets discovered during research reduce uncertainty: a `validation/data/` directory already containing ~1.4 GB of competition datasets (gitignored), and a Python `run_statsforecast.py` reference script that will anchor cross-library comparisons. The only high-uncertainty engineering task is the Rust TSF loader needed to read the Monash `.tsf` format.

The dominant risk across all phases is statistical-methodology traps that corrupt baselines before the work even begins. In-sample vs. out-of-sample confusion, MASE denominator collapse on intermittent series, and incorrect CV window schemes are all silent — they produce numbers but the numbers are wrong. The remediation strategy is sequencing: the accuracy harness metric layer must be code-reviewed and validated (against known M3 reference results for AutoETS monthly) before any baseline is committed. A second critical sequencing constraint is WASM-specific: the known dead code in `crates/anofox-forecast-js/` (7 unused `inner()` methods) must be removed before the WASM size baseline is locked in, or the baseline will undercount true optimization potential.

## Key Findings

### Recommended Stack

The tooling decision is dual-layer for speed: criterion for local wall-clock iteration, iai-callgrind for CI regression gates. Criterion wall-clock is too noisy on GitHub Actions runners to gate PRs reliably; iai-callgrind's instruction counts are deterministic. Both tools run on different schedules — criterion on a weekly schedule or PR label, iai-callgrind on every PR where relevant (ubuntu-latest only, Valgrind required). For memory, dhat 0.3.3 is wired behind a `dhat-heap` Cargo feature and used in native integration tests to assert `peak_bytes` bounds. WASM size is tracked deterministically by running `wasm-pack build --release` and committing the byte count in `.planning/baselines/wasm_size.json`; twiggy identifies the largest retained-size contributors when size grows. Coverage uses cargo-llvm-cov 0.8.7, which supersedes tarpaulin (the existing Makefile reference) for accuracy and multi-platform support. wee_alloc must not be used — it was archived August 2025 with known memory leaks.

**Core technologies:**
- **criterion 0.8.2**: wall-clock microbenchmarks — local dev, developer-facing latency iteration; already in repo
- **iai-callgrind 0.16.1**: instruction-count CI regression gate — deterministic on shared runners; ubuntu-only (Valgrind required)
- **critcmp**: CLI diffing of criterion baseline JSON files — enables before/after comparison across git refs
- **dhat 0.3.3**: in-process peak-heap assertions in native integration tests — gated by `dhat-heap` Cargo feature
- **cargo-llvm-cov 0.8.7**: LLVM source-based coverage — HTML/LCOV/JSON output; replaces tarpaulin
- **twiggy 0.8.0 + wasm-opt (via wasm-pack)**: WASM binary size profiling and shrinkage
- **csv 1.3**: zero-dependency CSV reader for M3/M4 fixture loading in `[dev-dependencies]`
- **Committed JSON baselines in `.planning/baselines/`**: single source of truth for all measurement dimensions

### Expected Features (Measurement Harness Capabilities)

**Must have (table stakes — missing these makes published results untrustworthy):**
- MASE with correct seasonal denominator and intermittent-series guard (denominator collapse → NaN guard required)
- sMAPE with correct denominator `(|y| + |ŷ|)/2`, not `|y|` alone
- Empirical interval coverage check (fraction of test points within nominal interval) — prerequisite for MSIS interpretation
- MSIS (Mean Scaled Interval Score) — standard M4 probabilistic track metric; currently missing from `calculate_metrics`
- Naive2 baseline (autocorrelation-gated: SeasonalNaive if seasonal, else Naive) — required for correct MASE denominator on yearly series and for OWA
- Rolling-origin (expanding-window) cross-validation — verify existing `cross_validate` never uses future data
- Per-frequency stratification in benchmark runner (Yearly/Quarterly/Monthly/Weekly/Daily/Hourly)
- Out-of-sample metrics only — in-sample fitted residuals must never appear in accuracy tables

**Should have (differentiators that enable credible external comparisons):**
- OWA (Overall Weighted Average) — M4 primary ranking metric; requires correct Naive2 baseline first
- RMSSE — M5/intermittent demand metric; relevant to Croston/IMAPA/TSB/ADIDA model families
- Pinball loss at target quantiles — evaluates conformal/QRA quantile outputs
- CRPS for distributional outputs (LaplaceForecaster, BootstrapPredictor)
- Per-horizon accuracy decomposition (h=1, h=1-3, h=4-6, full horizon) — detects horizon-specific degradation
- Diebold-Mariano significance test — required before claiming "model A beats model B"
- M3 full corpus run (3003 series) — the single most credible external validation; compare against published statsforecast AutoETS monthly results

**Defer (scope out of this cycle unless gaps emerge):**
- PIT histogram / KS calibration test for Gaussian outputs
- Conditional coverage stratified by series characteristics
- M5 loader (hierarchical retail, complex structure)
- Mutation score tracking alongside line coverage

**Anti-features (never add, never claim):**
- MAPE on zero-containing series — undefined on intermittent demand which is a core use case
- R² on forecast horizon — misleading, not used in any competition
- Single hold-out split accuracy — high variance; not comparable to competition results
- Aggregate accuracy without per-frequency stratification — hides category-level failure
- Automatic seasonal period detection inside benchmark evaluation — non-reproducible

### Architecture Approach

The harness is a consumer of the library, never a modifier of it. All measurement code lives in `benches/`, `tests/`, and `scripts/` — nothing enters `src/`. The baseline artifact store is `.planning/baselines/` (committed JSON), which CI reads but never writes. Three new GitHub Actions workflows are added: `bench.yml` (criterion + iai-callgrind, scheduled weekly + PR label), `wasm-size.yml` (WASM size delta on every PR), and `accuracy.yml` (workflow_dispatch only, requires dataset env var). The existing `ci.yml` is unchanged except for adding `--json --summary-only` to the coverage step to enable baseline comparison.

**Major components:**
1. **Speed benchmark layer** (`benches/`) — extends existing criterion benches with iai-callgrind hot-path gates; covers fit+predict across model families, single vs. batch, native-parallel vs. no-parallel profiles
2. **Memory measurement layer** (`tests/memory_native.rs`) — dhat::Alloc peak-bytes assertions; native-only; fresh process per series to avoid heap-state contamination
3. **Accuracy harness** (`tests/accuracy_harness.rs` or `crates/anofox-forecast-harness/`) — dataset-driven; gated on `ANOFOX_DATASET_DIR` env var; reads from `validation/data/` (1.4 GB, already present); outputs per-model/dataset/horizon metrics compared against `.planning/baselines/accuracy.json`
4. **WASM size tracking** (CI step in `wasm-size.yml`) — `wasm-pack build --release`, `wc -c`, compare to `baselines/wasm_size.json`; twiggy for size attribution
5. **Coverage layer** (extends existing `ci.yml`) — cargo-llvm-cov with JSON summary; compare to `baselines/coverage.json`; fail if below floor
6. **Baseline artifact store** (`.planning/baselines/`) — committed JSON for all five dimensions; updated only by maintainer-run `scripts/update_*.sh`

### Critical Pitfalls

1. **In-sample metrics reported as out-of-sample accuracy (A5)** — Temptation to use `fitted_values()` as a shortcut. Prevention: accuracy harness must be code-reviewed to confirm it uses walk-forward CV exclusively; every metric table must label its evaluation source. Verify by checking whether the harness ever calls `fitted_values()`.

2. **MASE denominator collapse on intermittent/constant series (A3)** — Croston/IMAPA/TSB series with many zeros produce zero or near-zero seasonal naive MAE, making MASE undefined (NaN/Inf). This silently corrupts aggregate metrics. Prevention: guard the denominator with `< 1e-8 * series_mean.abs()` threshold; exclude affected series from MASE aggregation and report them separately. The existing NaN-as-error-signal anti-pattern in `entropy.rs` must not be replicated.

3. **WASM dead code before size baseline is locked (D1 / sequencing)** — The 7 unused `inner()` methods and `RecipeKind` import in `crates/anofox-forecast-js/` are compiled into the WASM binary. If the size baseline is captured before cleanup, the baseline overstates current size and the cleanup looks like an optimization when it is just dead-code removal. Prevention: remove dead code first, then capture the WASM size baseline.

4. **Criterion baselines captured in CI (B2)** — GitHub Actions runners produce noisy wall-clock numbers. CI criterion baselines cannot reliably detect <10% regressions and generate false positives. Prevention: capture criterion baselines on a quiet local machine; commit the JSON to `.planning/baselines/criterion/`; CI uses iai-callgrind instruction counts for hard gates, not criterion timing.

5. **Rolling-origin CV leakage (A1)** — The harness built on top of `cross_validate()` can accidentally introduce future data via full-series preprocessing (scaling, Box-Cox, period detection) fitted before the fold loop. Prevention: all preprocessing must happen inside the fold loop on training data only; assert `train_end < test_start` in every fold.

6. **Debug WASM binary as baseline (D4)** — `wasm-pack build` defaults to `--dev` mode (5-10× larger binary). Prevention: all size baselines captured with `wasm-pack build --release`; CI script asserts the build profile.

7. **Native-parallel vs. WASM-single-thread conflation (D2)** — Rayon silently falls back to sequential execution in WASM. Prevention: maintain two separate benchmark profiles; document explicitly in the npm README.

## Implications for Roadmap

The research is unambiguous about ordering: measurement infrastructure and metric correctness must precede any optimization or comparison claim.

### Phase 1: Measurement Backbone & Baseline Capture

**Rationale:** Nothing downstream is trustworthy without committed baselines. WASM dead-code cleanup is gated here — it must happen before the WASM size baseline is committed.

**Delivers:**
- `.planning/baselines/` with committed JSON for all five dimensions (criterion, iai, accuracy, coverage, wasm_size, memory)
- `bench.yml`, `wasm-size.yml` CI workflows live and running
- iai-callgrind hot-path gates active on ubuntu CI
- WASM dead code removed (`inner()` methods, `RecipeKind` import); size baseline committed
- Criterion baselines captured locally and committed via critcmp export

**Addresses:** Stack decisions (criterion + iai-callgrind + dhat + cargo-llvm-cov + twiggy); Architecture Pattern 1 (committed JSON baselines); Architecture Pattern 4 (native vs WASM split)

**Avoids:** B1 (debug build benchmarks), B2 (CI timing noise), D4 (debug WASM baseline), D1 (monomorphization bloat without a baseline)

**Research flag:** Standard tooling patterns; Rust TSF loader is the one high-uncertainty task — if scoped to Phase 2, consider a Python pre-conversion fallback for Phase 1 to avoid blocking the accuracy baseline.

### Phase 2: Statistical Methodology & Accuracy Harness

**Rationale:** The highest-risk phase. Mistakes here corrupt every accuracy baseline. The existing `validation/data/` corpus and `run_statsforecast.py` reference script reduce data-preparation risk significantly.

**Delivers:**
- TSF/CSV dataset loader reading from `validation/data/` (M3, M4 sample, Tourism, NN5)
- Metric implementation: MASE (with seasonal denominator + intermittent guard), sMAPE, RMSE, MAE, MSIS, empirical coverage, pinball loss; all reviewed against FEATURES.md formula specifications
- Naive2 baseline with autocorrelation gate
- Accuracy harness gated on `ANOFOX_DATASET_DIR` env var; expanding-window CV with temporal-integrity assertions
- Per-frequency stratification; per-horizon accuracy decomposition
- `accuracy.yml` workflow (workflow_dispatch)
- Committed `baselines/accuracy.json` validated against published statsforecast AutoETS M3 monthly reference (MASE ~0.93)

**Addresses:** All P1 features from FEATURES.md; OWA and RMSSE as P2 follow-ons after Naive2 is correct

**Avoids:** A1 (temporal leakage), A2 (wrong CV window), A3 (MASE denominator collapse), A4 (marginal-only coverage), A5 (in-sample metrics), C3 (cherry-picked datasets), C4 (mismatched horizon aggregation)

**Research flag:** Needs careful planning review. TSF loader is highest-uncertainty engineering task. Metric formula correctness must be cross-checked against `run_statsforecast.py` output on at least one known series before any baseline is committed.

### Phase 3: Numerical Robustness & Coverage Baseline

**Rationale:** The codebase has ~4,768 `.unwrap()`/`.expect()` calls and documented anti-patterns. This phase captures the coverage baseline, identifies gaps, and adds robustness tests — creating the foundation for the improvement backlog.

**Delivers:**
- Coverage baseline in `baselines/coverage.json` via cargo-llvm-cov; floor enforced in CI
- `tests/robustness_edge.rs`: constant series, n=2 series, all-zeros (intermittent), NaN/Inf-containing, zero-length, extreme-scale inputs — one test per `ForecastError` variant asserting the variant type, not a panic
- `tests/memory_native.rs`: dhat::Alloc peak-bytes assertions for major model families
- Gap inventory: uncovered paths + assertion-free tests identified; filed as backlog candidates
- Proptest added to known-fragile areas (changepoint metrics, MSTL, CV boundary conditions)

**Avoids:** E1 (coverage % without assertions), E2 (error branches uncovered), B3 (allocator measurement variance)

**Research flag:** Standard patterns (cargo-llvm-cov, dhat, proptest). Assertion-density audit is a human judgment call during planning.

### Phase 4: Prioritized Improvement Backlog & Top-Value Fixes

**Rationale:** With all five dimensions baselined and robustness gaps documented, the backlog can be ranked by value/effort using real before-numbers. Each fix is merged only after the relevant harness confirms the improvement.

**Delivers:**
- Consolidated improvement backlog ranked by value (MASE delta, size reduction, coverage uplift, instruction-count reduction) vs. effort
- Top N improvements landed, each with documented before/after delta in the relevant baseline file
- Updated baselines committed after each improvement (separate PRs, not auto-committed)
- Regression guards in CI updated to new, tighter thresholds

**Avoids:** B5 (parallel vs. single-thread conflation in speed comparisons), C1/C2 (preprocessing and hyperparameter mismatch in cross-library comparisons)

**Research flag:** Specific improvements are unknown until baselines exist. Planning for this phase should begin after Phase 3 completes. Cross-library comparison claims require DM significance test if accuracy difference is < 5%.

### Phase Ordering Rationale

- Phase 1 before everything: CI workflows and committed baselines make all subsequent before/after claims trustworthy. WASM dead-code cleanup belongs here because it affects the size baseline.
- Phase 2 before Phase 4: No accuracy improvement claim is credible before the harness metric correctness is validated against a known reference.
- Phase 3 can begin alongside Phase 2 for coverage/memory infrastructure, but the gap inventory depends on coverage data.
- Phase 4 last: the backlog cannot be prioritized without knowing baseline numbers from all three preceding phases.
- The `accuracy.yml` workflow is workflow_dispatch from the start — it never gates PR merges.

### Research Flags

Phases needing deeper research or careful planning attention:
- **Phase 1:** Rust TSF loader for the Monash `.tsf` format — no mature crate exists; a hand-rolled parser is likely needed. Scope carefully; consider Python pre-conversion as a fallback.
- **Phase 2:** Metric formula correctness validation — plan a specific pre-commit verification step comparing harness output against `run_statsforecast.py` on the same series before any baseline is locked in.
- **Phase 4:** Cross-library comparison methodology — plan the preprocessing and hyperparameter matching checklist (PITFALLS C1, C2) before running any comparison numbers.

Phases with standard, well-documented patterns:
- **Phase 1 (tooling):** criterion, iai-callgrind, cargo-llvm-cov, twiggy, dhat are all well-documented. Installation and wiring are mechanical.
- **Phase 3 (robustness):** proptest, dhat testing mode, and cargo-llvm-cov coverage reporting are standard Rust patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | All tools verified on crates.io July-Aug 2026; wee_alloc archived status is HIGH confidence; TSF loader design is the one gap |
| Features | MEDIUM | Metric formulas grounded in Hyndman & Koehler (2006) and M4 Competitors Guide; OWA/RMSSE formula details are secondary-source |
| Architecture | MEDIUM | Patterns are well-established Rust benchmark practice; WASM-specific measurement limits are confirmed gaps |
| Pitfalls | HIGH | Grounded in documented codebase concerns (CONCERNS.md), established forecasting methodology literature, and Rust/WASM toolchain specifics |

**Overall confidence:** MEDIUM — tooling and methodology are well-understood; the Rust TSF loader and metric validation step against a reference implementation are the two remaining uncertainties.

### Gaps to Address

- **Rust TSF loader:** No authoritative crate exists for the Monash `.tsf` format. The parser design needs to be scoped before Phase 2 planning. Fallback: pre-convert `.tsf` to CSV with a one-time Python script.
- **MSIS for non-Gaussian intervals:** Confirm the library's existing interval output format (point + lower + upper) is accessible from the test harness without internal API changes.
- **getrandom/wasm_js in stochastic WASM paths:** Bootstrap and conformal WASM paths require `Crypto.getRandomValues`. The current WASM build configuration's getrandom feature setup is not confirmed; audit in Phase 1 alongside dead-code cleanup.
- **Cross-library comparison baseline:** `run_statsforecast.py` exists locally but the specific models, horizons, and dataset slices it uses need to be documented and locked before any comparison numbers are committed.

## Sources

### Primary (HIGH confidence)
- crates.io version verification (criterion 0.8.2, iai-callgrind 0.16.1, dhat 0.3.3, cargo-llvm-cov 0.8.7, twiggy 0.8.0) — July-Aug 2026
- wee_alloc GitHub archive notice — Aug 25 2025
- M4 Competitors Guide (UNIC, 2018) — MSIS formula specification
- Hyndman & Koehler (2006) — MASE definition and intermittent demand caveats

### Secondary (MEDIUM confidence)
- iai-callgrind GitHub (clockworklabs fork) — features, Valgrind CI dependency
- cargo-llvm-cov GitHub (taiki-e) — setup, rustc 1.87+ installation requirement
- Rust and WebAssembly Book — WASM code size tooling (twiggy, wasm-opt)
- The Rust Performance Book (nnethercote) — dhat/heaptrack benchmark guidance
- Monash Forecasting Repository (forecastingdata.org) — dataset access, TSF format, CC-BY 4.0 licensing
- M4-methods GitHub (Mcompetitions) — M4 CSV dataset files
- Nixtla statsforecast published M3 AutoETS monthly results — cross-library comparison anchor
- Retzlaff et al. (ICML 2025) — conformal prediction conditional vs. marginal coverage for time series
- getrandom docs and GitHub Issue #267 — wasm_js feature placement guidance

### Tertiary (LOW confidence)
- Diebold-Mariano test asymptotic variance estimation — single overview source; implementation details need validation
- critcmp baseline export workflow — community pattern; not official criterion documentation

---
*Research completed: 2026-08-09*
*Ready for roadmap: yes*
