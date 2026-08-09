# Pitfalls Research

**Domain:** Forecasting-library benchmarking and statistical validation (Rust, WASM)
**Researched:** 2026-08-09
**Confidence:** HIGH — grounded in documented codebase concerns, established forecasting methodology literature, and Rust/WASM toolchain specifics

---

## A. Statistical-Methodology Traps

### Pitfall A1: Standard k-Fold CV Applied to Time Series (Temporal Leakage)

**What goes wrong:**
Shuffled k-fold cross-validation breaks time order so that some training folds contain observations that are chronologically later than the test fold. The model effectively sees the future during training. Accuracy estimates are wildly optimistic, and model selection based on these scores picks the most overfit model rather than the most generalizable one.

**Why it happens:**
Developers copy CV patterns from tabular ML without adapting them to temporal data. The `cross_validate()` utility in `src/utils/cross_validation.rs` already implements walk-forward splits — the risk is in the *evaluation harness* built on top, not necessarily the library itself. Bugs include: computing rolling-window features over the full dataset before the CV loop, normalization fitted on the full series before splitting, and AutoForecast model-selection wrappers that accidentally reuse state across folds.

**How to avoid:**
- The training window must always end strictly before the test window begins — no gap-free overlap.
- All feature engineering, scaling (Box-Cox lambda, StandardScaler), and period detection must be fitted inside the fold loop on training data only, never on the full series.
- For the accuracy harness being built this cycle: use `TimeSeries::slice()` or a verifiable split index, and assert `train_end < test_start` in every fold's construction.
- Validate that `CVConfig` enforces `n >= 2 * min_train_size` rather than silently producing degenerate folds (known fragile area, CONCERNS.md).

**Warning signs:**
- Accuracy metric on CV is substantially better than on a simple held-out final window.
- AutoARIMA/AutoETS selecting a dramatically more complex model than expected (sign of overfitting to leaked future).
- MASE < 0.5 on seasonal series without a compelling structural reason.

**Phase to address:** Accuracy-harness phase (measurement backbone). Every fold construction must be code-reviewed for temporal integrity before baselines are accepted.

---

### Pitfall A2: Wrong CV Window Scheme for the Series Characteristics

**What goes wrong:**
Using a rolling (fixed-width) window when the series benefits from all history, or using an expanding window when the series has structural breaks. Either choice silently biases metric aggregation. Shorter training windows systematically underperform, making a library look worse than it is; longer windows over-represent early-period characteristics on regime-changed series.

**Why it happens:**
The choice between expanding and sliding windows is rarely documented in harness code. Competition datasets (M4, M5, Tourism) mix series lengths and structural properties. A single window scheme applied uniformly distorts per-category aggregates.

**How to avoid:**
- For competition-dataset harnesses: use expanding windows as the default (matches M4/M5 methodology), document the choice, and flag any series where expanding window gives anomalous results.
- Minimum training size must cover at least two full seasonal cycles (e.g., for monthly data with period 12, min `min_train_size = 24`). Enforce this as an assertion in the harness rather than a parameter.
- Seasonality-aware split design: ensure each fold sees at least one complete seasonal cycle in training; never split a fold mid-season.
- For AutoARIMA and AutoETS CV selection within `cv_select.rs`: the window scheme used for model selection must match the window scheme used for final evaluation.

**Warning signs:**
- Different window schemes produce MASE values that differ by > 20% on the same model/dataset combination.
- Short-series subsets show very high variance in metrics across folds.
- The minimum fold count drops below 3 for any series in the harness (statistically meaningless).

**Phase to address:** Accuracy-harness phase. Harness configuration must be documented and locked before baselining.

---

### Pitfall A3: MASE Denominator Collapse on Intermittent / Near-Zero Series

**What goes wrong:**
MASE is scaled by the in-sample seasonal-naive MAE. On intermittent demand series (many zeros, sparse non-zeros), or on very short series, this denominator can be zero or near-zero, producing undefined or astronomically inflated MASE values. One bad series in a batch can dominate aggregate reporting.

**Why it happens:**
The library has Croston, IMAPA, TSB, and ADIDA for intermittent demand. When the accuracy harness runs MASE across all model families on a mixed dataset, intermittent series will silently produce `f64::NAN` or `f64::INFINITY` denominators, which propagate (silently, given the library's known NaN-as-error-signal pattern in `entropy.rs`) into mean aggregates.

**How to avoid:**
- Guard the MASE denominator: if the seasonal-naive MAE is below a threshold (e.g., `< 1e-8 * series_mean.abs()`), exclude the series from MASE aggregation and report it separately as "MASE undefined."
- For intermittent series: prefer CRPS or cumulative-demand MASE (scale by cumulative sum rather than period-by-period naive), or use the Hyndman-Koehler scaled error computed over non-zero periods only.
- sMAPE is also problematic on zero-valued actuals (division by zero or by near-zero mean). The harness must guard this too.
- Carry all metric calculations through `Result<f64>` rather than relying on `f64::NAN` propagation — the existing NaN-return pattern in `src/features/entropy.rs` is an anti-pattern; the harness must not replicate it.

**Warning signs:**
- Any aggregate MASE value > 10 on standard benchmark datasets without a known pathological series.
- Aggregate metric is `NaN` (indicates no guarding in aggregation).
- Discrepancy between median MASE and mean MASE > 5× (outlier contamination).

**Phase to address:** Accuracy-harness phase. The metric computation layer must be reviewed before any baselines are locked in.

---

### Pitfall A4: Interval Coverage Evaluated Only as Marginal (Not Checking Conditional Structure)

**What goes wrong:**
Conformal prediction and bootstrap prediction intervals guarantee marginal coverage: "across all test points, 90% contain the true value." But if coverage is 90% on average yet 100% on easy series and 50% on hard ones (e.g., high-variance or near-changepoint segments), the intervals are systematically miscalibrated exactly where they matter most. Reporting one aggregate coverage number passes the check while the product is broken for its primary use cases.

**Why it happens:**
Coverage is easy to compute as a single scalar. The library has conformal and bootstrap postprocessors (`src/postprocess/`). The validation phase is likely to produce a single coverage table and declare success.

**How to avoid:**
- Report coverage stratified by: (a) series length bucket, (b) series volatility bucket (CV of the series), (c) forecast horizon bucket (h=1, h=1-6, h=7-12 for monthly), (d) changepoint proximity.
- A 90% nominal level should hit 88-92% in every stratum, not just in aggregate.
- Test exchangeability assumption: for conformal methods on time series, exchangeability is violated by temporal correlation. The coverage guarantee is approximate; verify empirically that it degrades gracefully rather than collapsing.
- Add a statistical test (binomial confidence interval on coverage) rather than just a point estimate to detect whether coverage deviation is noise or systematic.

**Warning signs:**
- Coverage within ±0.5% of nominal on aggregate but wide variation across horizons.
- Coverage on volatile series (CV > 1.0) significantly below nominal.
- No conditional stratification reported.

**Phase to address:** Statistical-methodology validation phase (distinct from the accuracy harness).

---

### Pitfall A5: In-Sample Fit Metrics Reported as Out-of-Sample Accuracy

**What goes wrong:**
Fitted values (one-step-ahead in-sample residuals) are used to compute MASE/RMSE and reported as the model's forecast accuracy. These are optimistic by construction — the model was fit to minimize exactly this residual. Models with more parameters always win, making AutoARIMA overfit systematically appear superior.

**Why it happens:**
It is tempting to use `Forecaster::fitted_values()` as a shortcut instead of setting up a full walk-forward harness. The residuals are already computed; a simple metric call produces a number that looks like accuracy.

**How to avoid:**
- The accuracy harness must produce out-of-sample metrics exclusively. No metric aggregated over fitted values should appear in accuracy tables.
- Label every metric in reports explicitly: "in-sample (fitted residuals)" vs. "out-of-sample (walk-forward CV)" — never mix them.
- If in-sample residuals are used for diagnostics (Ljung-Box, ACF of residuals), clearly scope that section as "model diagnostics," not "accuracy."

**Warning signs:**
- Accuracy metrics shown without specifying the evaluation window source.
- Models with many parameters (AutoARIMA with high p,q) rank first in every metric.
- No description of fold scheme in the harness code or documentation.

**Phase to address:** Accuracy-harness phase (harness design must be reviewed before any numbers are locked in).

---

## B. Compute-Benchmarking Traps

### Pitfall B1: Benchmarking Non-Release Builds (Cargo Default Is Debug)

**What goes wrong:**
`cargo bench` compiles benchmarks in release mode by default, but ad-hoc benchmark runners, scripts, or IDE "run" buttons may use `cargo run` or `cargo test` in debug mode. Debug builds have no inlining, no LLVM optimizations, and SIMD intrinsics may not be used. Reported numbers can be 10-100× slower than production. A baseline captured in debug mode will permanently mislead prioritization.

**Why it happens:**
The distinction between `cargo bench` (release) and `cargo test` (debug) is not obvious. Custom harnesses or criterion invocations without `--release` fall into debug silently. The project uses SIMD (`src/simd.rs`) which is a no-op in debug mode.

**How to avoid:**
- All benchmark harnesses must invoke `cargo bench` or `cargo test --release`.
- The CI step that captures baselines must explicitly pass `--release` and assert the build profile in the output.
- Add a compile-time assertion or a `#[cfg(debug_assertions)]` guard in the benchmark main that panics with a clear message if run in debug mode.
- Document in the benchmark README: "Never run with `cargo run`. Always use `cargo bench`."

**Warning signs:**
- Any benchmark showing fit times for ETS/ARIMA > 1 second on a 100-point series (characteristic of debug builds).
- The `criterion` output shows outlier ratios > 20% (may indicate mixed build artifacts in cache).
- SIMD code path is never exercised (check with `#[inline(never)]` on fallback path).

**Phase to address:** Compute-benchmark harness phase (first phase of the hardening cycle).

---

### Pitfall B2: Non-Reproducible Criterion Results from CI Noise

**What goes wrong:**
Criterion runs on shared CI infrastructure (GitHub Actions runners, virtualized environments) produce measurements with high variance due to CPU frequency scaling, noisy neighbors, turbo boost jitter, and cache state. A 5-15% performance change is indistinguishable from environmental noise. The before/after deltas that are the core value proposition of this project become meaningless.

**Why it happens:**
CI is the most convenient place to run benchmarks automatically. But virtualized CI environments do not support turbo boost disabling, CPU isolation, or pinned core affinity.

**How to avoid:**
- Criterion baselines for comparison are captured on a dedicated quiet machine (not CI), stored in `benches/baselines/`, and committed. CI compares against committed baselines with a noise threshold (e.g., ±10% = no regression, > 15% = flag for human review).
- Use `criterion`'s `--save-baseline` and `--load-baseline` flags for structured comparison.
- For CI: use wall-clock ordering tests ("model A must be faster than model B") rather than absolute timing assertions, which are inherently environment-dependent.
- Disable turbo boost (`echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo`) and background services when capturing reference baselines locally.
- Report median throughput, not mean, since criterion's bootstrap median is more robust to outlier spikes.

**Warning signs:**
- Criterion reports "change: +8.4% (p < 0.05)" on a benchmark that was not touched in the diff.
- Confidence intervals wider than 10% of the median on any benchmark.
- Successive CI runs of the same commit report different ranking of two close models.

**Phase to address:** Compute-benchmark harness phase. The baseline-capture protocol must be documented and reviewed before any optimization claims are made.

---

### Pitfall B3: Allocator / Memory-Measurement Variance Masking Real Changes

**What goes wrong:**
Peak allocation measurements using the system allocator are noisy because the allocator reuses freed blocks in non-deterministic order. Repeated measurements of the same allocation path can differ by 5-30% due to fragmentation state. If allocator-level measurements are used as a performance gate, false failures and false passes are common.

**Why it happens:**
The project has no memory profiling baseline yet. When adding one (DHAT, heaptrack, or a custom allocator wrapper), the first implementation often measures "current peak" which depends on the order of prior allocations in the process.

**How to avoid:**
- Use a fresh process per benchmark series (each criterion group spawns clean); avoid measuring allocation in a process that has already warmed up a large heap.
- For WASM-size measurement, measure the compiled `.wasm` binary with `wasm-pack build --release` and `wc -c` — this is fully deterministic and reproducible.
- For native peak allocation, use DHAT or `jemalloc`'s epoch-based stats reset; never use OS-level RSS as a proxy for peak allocation (RSS lags and is affected by kernel page reclaim).
- Track allocation counts (not just bytes) using a counting allocator wrapper to detect accidental O(n²) allocation patterns independently of allocation sizes.

**Warning signs:**
- Memory measurements vary by > 10% across runs on the same input without code changes.
- Peak memory claims are based on `/proc/self/statm` or OS resident set size.
- Memory and compute benchmarks are run in the same process (heap state from memory benchmark contaminates timing benchmark).

**Phase to address:** Memory and WASM-size measurement phase.

---

### Pitfall B4: Benchmark Warm-Up Misconfiguration Hiding Cold-Path Cost

**What goes wrong:**
Criterion's default 3-second warm-up fills CPU caches and branch predictor state. This gives a best-case throughput number for the steady-state hot path. If the real use case is fitting one series per request (cold start), the benchmark overestimates performance for that scenario by up to 5×. Conversely, for the batch `auto_ets()` path, if warm-up is omitted or too short, first-run cold-cache costs are included in the median.

**Why it happens:**
Default warm-up works for microbenchmarks. The fitting path for ARIMA/ETS involves large matrix allocations and lbfgs initialization that have different cold vs. warm behavior. No deliberate decision about cold vs. warm measurement was made in the existing benchmarks.

**How to avoid:**
- Explicitly document whether each benchmark is a "cold-start" or "steady-state" measurement.
- For cold-start benchmarks: use `Criterion::bench_function` with a `setup` closure that re-creates all data structures from scratch on each iteration, and reduce iteration count to a small number.
- For batch / throughput benchmarks: allow default warm-up, measure per-series throughput across a batch of 100+ series.
- For ARIMA convergence benchmarks: add a warm-up-disabled variant to measure worst-case branch-prediction cost.

**Warning signs:**
- Benchmark shows very different timing for `iter_count=1` vs. `iter_count=100` beyond what amortization explains.
- Batch benchmark reports linear speedup up to 8 cores but single-series benchmark shows Rayon thread-startup overhead dominating (sign that batch > single is being compared apples-to-oranges).

**Phase to address:** Compute-benchmark harness phase.

---

### Pitfall B5: Comparing Native-Rayon Performance Against Reference Libraries That Are Also Parallel

**What goes wrong:**
The batch `auto_ets()` path uses Rayon parallelism on native. When benchmarked against a Python reference (statsforecast, statsmodels) that also uses multiprocessing, the comparison conflates parallelism benefits with algorithmic efficiency. The library may claim a 10× speedup that is actually a 2× algorithmic advantage and 5× Rayon vs. single-process Python advantage.

**Why it happens:**
Convenience: run both on the same machine, report wall-clock time, declare victory.

**How to avoid:**
- Cross-reference benchmarks must be documented as: single-threaded-Rust vs. single-threaded-Python (algorithmic fairness), and parallel-Rust vs. parallel-Python (system-level fairness) separately.
- Pin both to a single thread using `RAYON_NUM_THREADS=1` and Python's `n_jobs=1` for the algorithmic comparison.
- Report the benchmark configuration (thread count, dataset size, hardware) in every comparison table.

**Warning signs:**
- Speedup reported without specifying thread count for either side.
- The library shows 50× advantage on batch that collapses to 3× single-threaded (Rayon explains the gap, not the algorithm).

**Phase to address:** Cross-reference benchmark phase.

---

## C. Unfair Reference-Library Comparison Traps

### Pitfall C1: Mismatched Preprocessing Between Library and Reference

**What goes wrong:**
anofox-forecast's `AutoETS` or `AutoARIMA` silently applies Box-Cox transformation, seasonal differencing, or outlier removal before fitting. The reference library (statsforecast, statsmodels) may not apply the same transforms by default. A MASE comparison without matching preprocessing compares two different effective models, not two implementations of the same algorithm.

**Why it happens:**
The library has a rich transformation pipeline (`src/transform/`) that is partially baked into model defaults. Reference library defaults are different. No one checks what transforms are actually applied.

**How to avoid:**
- Before running comparison benchmarks, document exactly which preprocessing is applied by each side: (a) log/Box-Cox transform, (b) detrending/differencing, (c) outlier winsorization, (d) missing-value imputation.
- Configure both sides to use identical preprocessing or explicitly disable it on both sides and compare base algorithms.
- Output the actual fitted parameters from both sides for one sample series and confirm they agree within numerical precision before accepting any comparative accuracy number.

**Warning signs:**
- Large accuracy gap between the library and a reference implementation on a simple, well-understood dataset (e.g., AirPassengers, M4 yearly).
- Reference produces wider prediction intervals than the library for the same nominal level — sign of different variance estimation or transform.

**Phase to address:** Cross-reference benchmark phase.

---

### Pitfall C2: Different Default Hyperparameters Between Library and Reference

**What goes wrong:**
AutoARIMA's parameter search space (max p, max q, information criterion: AIC vs. BIC vs. AICc, stepwise vs. exhaustive) differs between this library and statsforecast or R's `auto.arima`. The "same" algorithm selects different orders, producing legitimately different accuracy. Reporting this as the library being better or worse is misleading.

**Why it happens:**
Reference implementations often default to stepwise=true, AICc, max_p=5. anofox-forecast's defaults may differ. Nobody checks.

**How to avoid:**
- For every model class in the comparison: write out both libraries' effective configuration as a checklist and verify they match on: information criterion, max parameter bounds, seasonal search on/off, stationarity test (KPSS vs. ADF vs. PP), and convergence tolerance.
- Prefer comparing "simple fixed models" (e.g., ARIMA(1,1,1)(1,1,1)[12]) where configuration is fully specified, before comparing "auto" variants.

**Warning signs:**
- AutoARIMA selects different (p,d,q) orders across the two libraries on the same series — expected if search space differs; should be explained, not ignored.
- One library systematically selects higher-order models (sign of different information criterion).

**Phase to address:** Cross-reference benchmark phase.

---

### Pitfall C3: Cherry-Picked Datasets Reporting Biased Results

**What goes wrong:**
A benchmark run on 5 hand-selected series where the library happens to outperform is published as "library beats statsforecast." The selection criteria are not reported. The result is not generalizable.

**Why it happens:**
Easy to run on a few known-good series first, then stop. Confirmation bias accelerates this.

**How to avoid:**
- Use only vendored standard competition datasets as the comparison corpus (M4, M5, Tourism, Monash Archive). These are pre-specified and cannot be cherry-picked post-hoc.
- Report full dataset averages, broken down by series type (yearly/quarterly/monthly/weekly/daily/hourly for M4), not means across a selected subset.
- Pre-register the evaluation protocol (dataset, metric, window scheme, model configuration) before running any numbers.

**Warning signs:**
- Comparison uses fewer than 100 series without a documented reason for the small sample.
- Only one dataset is used for comparison — especially if it was selected after seeing the results.
- Per-category breakdown (yearly vs. monthly) is omitted from reporting.

**Phase to address:** Cross-reference benchmark phase.

---

### Pitfall C4: Mismatched Horizon or Aggregation Across Metrics

**What goes wrong:**
The library is compared at horizon h=1 while the reference is tuned for h=12. Or: sMAPE is averaged across all horizons while MASE is reported only for h=1. The metrics are not comparable across different aggregation schemes.

**Why it happens:**
Metrics are reported as scalars without specifying the horizon or aggregation method. The default `calculate_metrics()` call may average over all horizons, while the reference library paper reports h=1.

**How to avoid:**
- All metric tables must specify: (a) forecast horizon or range, (b) aggregation method (mean, median, weighted), (c) series types included.
- The accuracy harness must output horizon-stratified metrics (h=1, h=1-3, h=4-6, full horizon) as separate columns, never collapse to a single number for comparison purposes.

**Warning signs:**
- A competitor paper reports MASE 0.85 for a dataset but the harness produces MASE 1.1 — difference may be entirely due to horizon aggregation, not accuracy.
- No horizon breakdown in the comparison output.

**Phase to address:** Accuracy-harness phase (metric schema must be defined before any numbers are computed).

---

## D. WASM-Specific Traps

### Pitfall D1: Monomorphization Bloat from Generic Forecaster Wrappers

**What goes wrong:**
The WASM binding layer in `crates/anofox-forecast-js/src/forecaster.rs` (3,316 lines) wraps each model family in its own concrete type. When generic code (e.g., utility functions templated over the `Forecaster` trait) is included in WASM, Rust instantiates a separate copy per concrete type. With 30+ model families, this can add hundreds of kilobytes to the `.wasm` binary without any functional benefit.

**Why it happens:**
Rust monomorphization is silent. There is no warning when a generic function is instantiated 30 times. The WASM binary grows incrementally with each new model wrapper, and nobody notices until the binary is measured.

**How to avoid:**
- Establish a `.wasm` binary size baseline with `wasm-pack build --release` as part of the measurement-backbone phase, and track it in CI (fail if size grows by > 5% without an explicit approval).
- Use `twiggy top` or `wasm-objdump --section-headers` to identify the largest symbols; investigate any function that appears more than 5 times with different suffixes (monomorphization).
- Prefer `dyn Forecaster` (dynamic dispatch) over generics in the WASM binding layer — the WASM target has no Rayon speedup to justify the static dispatch cost.
- The existing dead code (7 unused `inner()` methods, `RecipeKind` import) should be removed before the size baseline is locked in, since dead code that is not `#[cfg(test)]` is still compiled into the binary.

**Warning signs:**
- `wasm-pack build` output shows `.wasm` size > 2 MB before gzip compression.
- `twiggy top` reveals 10+ variants of the same function name with type suffixes.
- Adding a new model wrapper increases binary size by > 20 KB.

**Phase to address:** Memory and WASM-size measurement phase (establish baseline); clean up dead code before baseline capture.

---

### Pitfall D2: No-Rayon Performance Confusion (WASM vs. Native Profiles)

**What goes wrong:**
Batch-mode benchmarks run on native with Rayon enabled show impressive throughput. The WASM build silently falls back to single-threaded sequential execution (Rayon's WASM fallback mode). Documentation, playground demos, and the npm package claim "fast batch processing" but the WASM user sees single-threaded performance. No separate WASM throughput baseline exists.

**Why it happens:**
Rayon's WASM fallback is silent — it does not panic or warn, it just runs sequentially. The `parallel` feature is correctly feature-gated, but nowhere in the WASM benchmarks is the single-threaded cost explicitly measured and documented.

**How to avoid:**
- Maintain two separate benchmark profiles: `native-parallel` (with `parallel` feature) and `wasm-single-thread` (without, targeting WASM). Both must have baselines.
- The WASM playground should display a per-prediction timing in the browser console using `performance.now()` — this makes WASM perf visible to users and prevents silent regression.
- Document explicitly in the npm package README: "batch mode is single-threaded in WASM; for high throughput, use the native Rust library with the `parallel` feature."
- Avoid any benchmark comparison that mixes native-parallel and WASM-single-thread numbers.

**Warning signs:**
- WASM playground shows progressively increasing latency per series for batch operations (sequential Rayon fallback scaling linearly).
- A PR enables the `parallel` feature on a WASM build target — this is a CI break, not a speedup.

**Phase to address:** Compute-benchmark harness phase (both profiles must be established simultaneously).

---

### Pitfall D3: `getrandom` / JS Crypto API Not Activated for WASM Stochastic Models

**What goes wrong:**
Bootstrap forecasting, conformal prediction, and GARCH simulation paths require random number generation. The `getrandom` crate on `wasm32-unknown-unknown` does not automatically use `Crypto.getRandomValues` — it requires the `wasm_js` feature to be explicitly activated. Without it, calls to `rand` panic at runtime in the browser. If the `wasm_js` feature is enabled as a library dependency (not just for tests), it forces `wasm-bindgen` into every downstream consumer, causing binary bloat.

**Why it happens:**
The `getrandom` docs contain a strong recommendation against enabling `wasm_js` in library crates except for test targets. This is easy to miss, and the panic only appears at runtime under the specific execution path that generates random numbers (e.g., when the user calls `bootstrap_forecast()` from JS).

**How to avoid:**
- Enable `getrandom/wasm_js` only in the `[dev-dependencies]` or in a `wasm-bindgen-test` configuration, not in `[dependencies]`.
- For stochastic features that require randomness in WASM, provide a JS-side seed injection API (`set_rng_seed(seed: u64)`) rather than depending on the runtime environment for entropy.
- Add a `wasm-bindgen-test` that explicitly calls each stochastic code path and asserts no panic — this catches getrandom configuration errors before they reach npm users.

**Warning signs:**
- `cargo build --target wasm32-unknown-unknown` succeeds but `wasm-bindgen-test` crashes with "getrandom: unsupported target" or similar runtime error.
- The `Cargo.lock` for the WASM crate includes `wasm-bindgen` as a transitive dependency from `getrandom` even though the consumer did not request it.

**Phase to address:** WASM binding integration test phase (part of the code-correctness and coverage baseline).

---

### Pitfall D4: Debug Information Included in WASM Binary Inflating Size Baseline

**What goes wrong:**
`wasm-pack build` (without `--release`) includes debug information and symbol names in the `.wasm` binary. The resulting size may be 5-10× larger than the release build. If the size baseline is accidentally captured in non-release mode, all subsequent release comparisons will appear to show dramatic size reductions when no optimization was actually done.

**Why it happens:**
`wasm-pack build` defaults to `--dev` (debug) mode. The `--release` flag must be specified explicitly. This is a common mistake when exploring WASM output.

**How to avoid:**
- All size baselines must be captured with `wasm-pack build --release`.
- The CI script that captures the size baseline must print and assert the build profile before reporting the number.
- Use `wasm-opt -Oz` as a post-build step (wasm-pack applies this automatically in release mode) and verify it is applied by checking for the `producers` section in `wasm-objdump`.

**Warning signs:**
- `.wasm` binary > 10 MB (characteristic of an unoptimized debug build of this library size).
- `wasm-objdump` shows a large `name` section (debug symbols not stripped).

**Phase to address:** Memory and WASM-size measurement phase.

---

## E. Coverage-Metric Misuse

### Pitfall E1: Chasing Line Coverage Percentage Without Behavioral Assertions

**What goes wrong:**
A test that calls `model.fit(&series)` and discards the result achieves 100% line coverage for the fit path while asserting nothing. The CI coverage gate turns green. The library ships with a fit path that corrupts state under edge cases because no assertion checked the output. This is especially dangerous for the known fragile paths: `src/changepoint/metrics.rs`, `src/seasonality/mstl.rs`, and the cross-validation boundary conditions documented in CONCERNS.md.

**Why it happens:**
Coverage tools measure execution, not correctness. When coverage targets are set as a hard gate (e.g., "must be > 80%"), developers write tests that execute code paths without asserting expected behavior, because the tool cannot tell the difference.

**How to avoid:**
- Coverage is a discovery tool for uncovered paths, not a quality gate in itself. Set a coverage floor to prevent gross neglect (e.g., 60%), but do not reward increasing coverage without behavioral assertions.
- For every new test added to improve coverage: require at least one of: (a) a numerical assertion on model output (fitted parameters within known bounds), (b) an error-type assertion for invalid inputs, or (c) a round-trip assertion (fit-predict-refit produces consistent state).
- For the fragile areas specifically: add property-based tests (proptest) that fuzz inputs and assert safety invariants (no panic, no NaN in output) — these are richer than line-coverage tests.
- Track mutation score alongside line coverage. If mutation score < 40% while line coverage > 70%, the tests are executing paths without catching regressions.

**Warning signs:**
- Coverage increases by > 5% in a PR that adds tests with no assertions (or only `assert!(result.is_ok())`).
- The fragile areas (changepoint, MSTL, CV boundary) show high line coverage but have no tests with numerical assertions.
- Introducing a deliberate regression (e.g., negating a sign in MASE computation) does not cause any test failure.

**Phase to address:** Code-correctness and coverage baseline phase. The coverage baseline must be accompanied by an assertion-density audit, not just a percentage report.

---

### Pitfall E2: Coverage on Happy Paths Only, Missing the Error and Edge-Case Branches

**What goes wrong:**
Standard test suites cover the well-formed input path thoroughly but leave error branches untested. The existing gap inventory (CONCERNS.md) already documents this for changepoint metrics (malformed breakpoints), WASM integration (NaN/Inf from JS), and ensemble voting (empty model registry). If coverage is measured only on success paths, coverage percentage is high while the library can still panic on user-visible inputs.

**Why it happens:**
Error paths are harder to construct in tests. Proper error testing requires manufacturing invalid `TimeSeries` objects or simulating numerical failure conditions. The `.unwrap()` and `.expect()` calls (~4,768 instances) are often on paths that are only reached by unusual inputs.

**How to avoid:**
- The coverage measurement must include adversarial test cases: constant series, very short series (n=2), series with exactly one non-zero value (intermittent), series with Inf/NaN values, zero-length series.
- Each `ForecastError` variant must have at least one test that causes it to be returned and asserts the variant type.
- The `validate_series_complete()` skip pattern (some `fit()` paths skip it, per ARCHITECTURE.md anti-patterns) must be caught by tests: call `fit()` on a NaN-containing series and assert `ForecastError::MissingValues`, not a panic.
- Use cargo's `--include-only` coverage filter to specifically measure error branch coverage as a separate metric from happy-path coverage.

**Warning signs:**
- Any `panic!` or `.unwrap()` hit during a property-based fuzzing run (proptest) is a coverage gap by definition.
- `ForecastError::SingularMatrix` or `ForecastError::ConvergenceFailure` variants have no tests that produce them.
- WASM integration test only calls `fit()` on valid, well-formed data from JS.

**Phase to address:** Code-correctness and coverage baseline phase; input-robustness suite phase.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `.unwrap()` in internal helpers | Avoids `?` propagation boilerplate | Panic on unexpected edge cases; ~4,768 instances already in codebase | Never in public API paths; acceptable in truly infallible private helpers with documented invariants |
| NaN as error signal (entropy) | Avoids `Result` return type in numerical code | Silent propagation into aggregations; impossible to distinguish "error" from "computed NaN" | Never — convert to `Result<f64, ForecastError>` |
| Single aggregate coverage % | Simple CI gate | Hides uncovered error branches and assertion-free tests | Only as a floor (not a ceiling) to prevent gross neglect |
| Same window scheme for all series types | Simpler harness code | Bias in metric aggregation for short/seasonal series | Not acceptable in a benchmark that claims per-category accuracy |
| Shared MASE denominator for all series | Simpler metric implementation | Undefined/infinite values for intermittent/constant series | Never without a guarded fallback |
| Capture baseline in CI | Automated | Noisy; environment-dependent; baselines drift without cause | Only for ordering/regression tests, never for absolute timing claims |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| A1: k-fold leakage | Accuracy-harness design | Code-review fold construction; assert `train_end < test_start` in every fold |
| A2: Wrong CV window scheme | Accuracy-harness design | Document scheme choice; verify min 2 seasonal cycles per fold |
| A3: MASE denominator collapse | Accuracy-harness design (metric layer) | Confirm no NaN/Inf in any aggregate metric table |
| A4: Marginal-only coverage evaluation | Statistical-methodology validation | Coverage must be stratified by horizon, volatility bucket, and series length |
| A5: In-sample metrics reported as accuracy | Accuracy-harness design | All metrics labeled with evaluation source; no fitted-residual metrics in accuracy tables |
| B1: Debug build benchmarks | Compute-benchmark harness | CI asserts `--release`; add debug-assertion guard in bench main |
| B2: CI timing noise | Compute-benchmark harness | Baselines captured locally; CI uses ordering tests only |
| B3: Allocator measurement variance | Memory/WASM-size measurement | Fresh process per bench series; use DHAT not RSS |
| B4: Warm-up misconfiguration | Compute-benchmark harness | Document cold vs. warm profile per benchmark group |
| B5: Parallel vs. single-thread conflation | Cross-reference benchmark | Separate sections for single-thread and parallel comparison |
| C1: Preprocessing mismatch | Cross-reference benchmark | Pre-run checklist of transforms for both sides |
| C2: Hyperparameter mismatch | Cross-reference benchmark | Fixed-model comparison before auto-model comparison |
| C3: Cherry-picked datasets | Cross-reference benchmark | Use only pre-specified vendored competition datasets |
| C4: Mismatched horizon aggregation | Accuracy-harness design | Metric schema defined before any numbers are computed |
| D1: Monomorphization WASM bloat | Memory/WASM-size measurement | `twiggy top` analysis; size CI gate ±5% |
| D2: No-Rayon WASM confusion | Compute-benchmark harness | Separate native-parallel and WASM-single-thread baselines |
| D3: getrandom/JS crypto misconfiguration | WASM binding integration tests | `wasm-bindgen-test` must exercise stochastic code paths |
| D4: Debug WASM binary as baseline | Memory/WASM-size measurement | CI asserts `wasm-pack build --release`; check binary size plausibility |
| E1: Coverage % without assertions | Code-correctness and coverage baseline | Assertion-density audit; mutation score alongside line coverage |
| E2: Happy-path-only coverage | Code-correctness and coverage baseline; Input-robustness suite | Every `ForecastError` variant tested; proptest on fragile areas |

---

## "Looks Done But Isn't" Checklist

- [ ] **Accuracy harness:** Metrics are labeled with evaluation source (out-of-sample walk-forward, not fitted residuals). Verify by checking whether the harness code calls `fitted_values()` anywhere.
- [ ] **MASE implementation:** Denominator is guarded against zero. Verify by running harness on a constant series and checking the output is "undefined" rather than NaN or Infinity.
- [ ] **WASM size baseline:** Captured with `wasm-pack build --release`, not `--dev`. Verify by comparing `.wasm` size > 10 MB indicates debug mode.
- [ ] **Criterion baselines:** Stored with `--save-baseline` on a quiet machine, not in CI. Verify the commit timestamp and hostname match a known quiet build machine.
- [ ] **Coverage baseline:** Accompanied by an assertion-density audit, not just a percentage. Verify by searching for test functions with zero `assert!` / `assert_eq!` / `proptest!` calls.
- [ ] **Cross-reference comparison:** Includes both single-threaded and parallel configurations explicitly documented. Verify by checking that `RAYON_NUM_THREADS=1` and `n_jobs=1` appear in the benchmark script.
- [ ] **Conformal coverage:** Reported stratified by horizon and volatility bucket, not only as a single aggregate. Verify by checking the coverage output shape.
- [ ] **getrandom in WASM:** Stochastic code paths tested via `wasm-bindgen-test` before claiming WASM feature-completeness. Verify by checking that bootstrap and conformal WASM tests exist.

---

## Sources

- Hyndman & Koehler, "Another look at measures of forecast accuracy" (MASE definition and intermittent demand caveats): https://robjhyndman.com/papers/mase.pdf
- OpenForecast, "Don't use MAE-based error measures for intermittent demand": https://openforecast.org/2025/01/21/don-t-use-mae-based-error-measures-for-intermittent-demand/
- Retzlaff et al., "Testing Marginal and Conditional Coverage in Conformal Prediction for Non-Stationary Time Series": https://proceedings.mlr.press/v266/retzlaff25a.html
- Berkeley, "The limits of distribution-free conditional predictive inference": https://www.stat.berkeley.edu/~ryantibs/papers/limits.pdf
- Criterion.rs docs (warmup, noise, baseline comparison): https://bheisler.github.io/criterion.rs/book/user_guide/command_line_output.html
- rustwasm.github.io, "Shrinking .wasm Size": https://rustwasm.github.io/book/reference/code-size.html
- getrandom docs (wasm32-unknown-unknown and wasm_js feature): https://docs.rs/getrandom/latest/getrandom/
- getrandom GitHub Issue #267 (where to enable js feature): https://github.com/rust-random/getrandom/issues/267
- arxiv, "Cherry-Picking in Time Series Forecasting": https://arxiv.org/pdf/2412.14435
- arxiv, "TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting": https://arxiv.org/html/2403.20150
- Hyndman, "A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction": https://robjhyndman.com/papers/cv-wp.pdf
- techbuddies.io, "Top 7 Rust Custom Allocators Mistakes That Kill Performance": https://www.techbuddies.io/2026/04/02/top-7-rust-custom-allocators-mistakes-that-kill-performance/
- anofox-forecast codebase: `.planning/codebase/CONCERNS.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/PROJECT.md`

---
*Pitfalls research for: forecasting-library benchmarking and statistical validation (Rust, WASM)*
*Researched: 2026-08-09*
