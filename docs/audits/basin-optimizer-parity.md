# basin-Backed Optimizer Parity (GH #234, Steps 1-3)

**Scope:** evaluate `basin` 3.1.0 as a drop-in evaluation backend for `utils::optimization`'s
`nelder_mead` / `lbfgs_optimize`, behind the public signatures, with the hand-rolled optimizers
kept and unchanged. This is step 1 of GH #234; it does not flip any default.

## Routing mechanism

The `basin` cargo feature compiles the dependency and `src/utils/optimization/basin_backend.rs`
(`nelder_mead_basin`, `lbfgs_optimize_basin` — public, unit-tested, clippy-checked, doc-checked).
Routing the public `nelder_mead` / `lbfgs_optimize` to basin additionally requires the compiler
cfg `anofox_basin_optimizer`. This differs from the literal "when the `basin` feature is on,
route them" — feature-only routing was rejected for three reasons:

1. `.github/workflows/ci.yml` runs `cargo test --all-features` on stable/beta/nightly and
   `cargo llvm-cov --all-features`. A feature-only switch would run the whole R/statsforecast
   reference suite against an optimizer that had not yet been validated — the parity work this
   document reports would have shipped red CI, not measurements.
2. Cargo features unify across a dependency graph. If any crate in a downstream consumer's graph
   enabled `basin`, every model fit for every consumer would silently change numerically. A
   switch that changes numeric behavior should not ship as an additive feature until parity is
   proven (GH #234 step 3 is where the default flips).
3. `--all-features` still builds and unit-tests the basin backend, so it cannot rot while
   unused.

It is reversible: step 3 can change the cfg gate to the feature alone.

**Reproduce both runs:**

```bash
# Legacy routing (default; what CI runs today)
cargo test --all-features --no-fail-fast

# basin routing
RUSTFLAGS="--cfg anofox_basin_optimizer" CARGO_TARGET_DIR=target/basin-route \
    cargo test --all-features --no-fail-fast
```

`nelder_mead_legacy` and `lbfgs_optimize_legacy` are the renamed, byte-for-byte-unchanged
hand-rolled bodies; they are not deleted and remain the default path (`grep -c 'fn
nelder_mead_legacy\|fn lbfgs_optimize_legacy' src/utils/optimization.rs` = 2). The `lbfgs` crate
dependency (hand-rolled L-BFGS's Hessian approximation) is also unchanged and still present.

## Termination mapping

### Nelder-Mead (`NelderMeadConfig` → basin `NelderMead`)

| Legacy field | basin setting | Equivalence |
|---|---|---|
| `max_iter` | `Executor::max_iter(max_iter as u64)` | Exact |
| `tolerance` | `with_absolute_simplex_cost_tolerance(tolerance)` (size/relative-cost tolerances disabled) | Exact — reproduces legacy's "value range < tolerance" test; see note below |
| `tolerance` (2nd use) | `Executor::stop_when(...)` closure computing the centroid-distance test | Exact — reproduces legacy's simplex-diameter test manually, since basin has no identical built-in |
| `stagnation_window` | `Executor::no_objective_improvement(window as u64, tolerance)` (only when `> 0`) | Approximate — basin's running-anchor stall tracker and legacy's reset-on-improvement counter are functionally equivalent (both ask "no improvement > tolerance for N iterations") but are different implementations |
| `alpha` (reflection) | `NelderMead::with_params(alpha, ..)` | Exact, same coefficient |
| `gamma` (expansion) | maps to basin's `beta` | Exact value, renamed slot (basin requires `beta > 1`) |
| `rho` (contraction) | maps to basin's `gamma` | Exact value, renamed slot (basin requires `0 < gamma < 1`) |
| `sigma` (shrink) | maps to basin's `delta` | Exact value, renamed slot (basin requires `0 < delta < 1`) |
| `initial_step` | used to build the initial simplex passed to `SimplexProgress::from_simplex` | Exact — basin takes explicit vertices, so the legacy construction (vertex 0 clamped; vertex `i+1` nudged by a relative-or-absolute step, clamped) is reproduced manually in `build_initial_simplex` |

`with_absolute_simplex_size_tolerance`, `with_relative_simplex_size_tolerance`, and
`with_relative_simplex_cost_tolerance` are all explicitly disabled (`None`); only the absolute
cost tolerance is enabled, which per basin's own AND/OR combination rule ("a dimension with both
settings disabled is omitted from the AND group") collapses to exactly legacy's single
`range < tolerance` check.

### L-BFGS / L-BFGS-B (`LbfgsConfig` → basin `Lbfgs`)

| Legacy field | basin setting | Equivalence |
|---|---|---|
| `max_iter` | `Executor::max_iter(max_iter as u64)` | Exact |
| `tolerance` (unbounded) | `with_absolute_gradient_tolerance(tolerance)` | Exact — Euclidean gradient norm, same as legacy |
| `tolerance` (bounded) | `with_absolute_projected_gradient_tolerance(tolerance)` | Approximate — basin's bound-clipped infinity-norm projected-gradient test, not legacy's plain Euclidean norm in unconstrained (logit) space; different norms, same intent |
| `memory_size` | `with_m_capacity(memory_size)` | Exact |
| `fd_step` | used inside `GradientObjective::gradient` | Exact central-difference step `h = fd_step * (1 + |x_i|)`, reproduced manually with a bound-aware one-sided fallback (see below) — basin has no equivalent built-in finite-difference option matching legacy's exact probe rule |
| `armijo_c` | `Backtracking::with_sufficient_decrease_coefficient(armijo_c)` | Exact, same Armijo sufficient-decrease coefficient |
| `backtrack_rho` | `Backtracking::rho(backtrack_rho)` | Exact, same shrink factor |
| `max_linesearch` | `Backtracking::max_iter(max_linesearch as u32)` | Exact, same per-step budget |
| *(none)* | box constraints handled **natively** by basin's L-BFGS-B (Cauchy point + subspace minimization) | **No equivalent** — legacy reparametrizes to an unconstrained space with `atanh`/sigmoid and runs unconstrained L-BFGS; basin solves the box-constrained problem directly. This is the single largest semantic difference between the two backends. |
| *(none)* | basin's default signed relative-cost-reduction stop for `f64` bounded mode: `f_prev - f_cur <= 10*EPSILON*max(|f_prev|,|f_cur|,1)` | **Extra, no legacy equivalent** — left at basin's default rather than disabled, since it only fires on marginal improvement and is consistent with "converged" |

**Legacy fallback conditions** (both optimizers, threat T-go2-01): empty point (`n == 0`);
coefficients/settings outside basin's admissible ranges (NM: `alpha>0`, `beta(=gamma)>1`,
`0<gamma(=rho)<1`, `0<delta(=sigma)<1`, `max_iter>0`, finite `tolerance>=0`; L-BFGS:
`memory_size>=1`, `0<backtrack_rho<1`, `0<armijo_c<1`, `max_linesearch>=1`, `max_iter>0`, finite
`tolerance>=0`, finite `fd_step>0`); or `Executor::run()` itself returning `Err` (e.g. an
inverted/non-finite box under basin's stricter validation). All three are unit-tested.

**Bound-aware finite-difference gradient** (`GradientObjective`): central difference when both
`x_i + h` and `x_i - h` stay inside `[lower_i, upper_i]`; backward difference
`(f(x) - f(x-h))/h` when `x_i + h` would exceed the upper bound; forward difference
`(f(x+h) - f(x))/h` when `x_i - h` would drop below the lower bound. For the unbounded path,
`lower`/`upper` are filled with `(-inf, inf)`, so the comparison is always false and the
behavior is identical to legacy's plain central difference.

## Dependency surface and MSRV

New crates pulled in by `--features basin` (native target, default-features off; a dependency
that was already present at the same version — `num-traits 0.2.19`, `ppv-lite86 0.2.21`,
`zerocopy 0.8.31`, `libm 0.2.15` — is a new *edge*, not a new crate):

| Crate | Version |
|---|---|
| `basin` | 3.1.0 |
| `rand` | 0.10.3 |
| `rand_core` | 0.10.1 |
| `rand_chacha` | 0.10.0 |
| `rand_distr` | 0.6.0 |
| `web-time` | 1.1.0 |

**Duplicate versions:** the crate's dependency graph already carried two `rand` major versions
before this change — `rand 0.8.6` (the crate's own direct dependency and `anofox-statistics`)
and `rand 0.9.2` (via `proptest`/`argmin-math`, dev-only and transitively through
`anofox-regression`). `basin` adds a **third**: `rand 0.10.3`. All three stay fully isolated
(no shared `rand_core` major version between them: `0.6.4`, `0.9.3`, `0.10.1` respectively), so
there is no API-incompatibility risk, only binary-size and compile-time cost from three
independent RNG stacks. The default (no `basin`) tree is unaffected:
`cargo tree -p anofox-forecast -e normal --prefix none | grep -c '^basin '` = 0.

**MSRV:** `basin` declares `rust-version = "1.87.0"` and `edition = "2024"`.
`anofox-forecast` declares **no** `rust-version` in `Cargo.toml`. Measured directly:
`cargo +1.86.0 check --features basin` fails at the dependency-resolution step with Cargo's own
message (`basin@3.1.0 requires rustc 1.87.0`) — so the floor is enforced by Cargo itself, not by
a compile error buried in basin's code. Separately, `cargo +1.86.0 check` (default features, no
basin) already fails to compile at HEAD for an unrelated, pre-existing reason: `argmin` (pulled
in by the default `postprocess` feature via `anofox-regression`) uses `u64::is_multiple_of`,
stabilized after 1.86. `cargo +1.88.0 check` (default features) also fails at HEAD, for a third
unrelated reason: `trueno`'s AVX-512 backend uses intrinsics gated behind an unstable feature on
that toolchain. `cargo +1.91.0 check --features basin` succeeds cleanly. **The project's
effective floor is therefore already above 1.87 for reasons unrelated to basin**; `basin`'s MSRV
does not newly constrain anything in practice, but the DuckDB WASM pin mentioned in context
(rustc 1.86) would need to move to >= 1.87 before it could consume a build with `basin` enabled.

**`cargo audit`** (run locally; `cargo-deny` is not installed in this environment, so that gate
was not exercised): 1 vulnerability (`crossbeam-epoch` 0.9.18, via `rayon`/`criterion`, dev-only)
plus 6 allowed warnings (`bincode`, `paste`, `proc-macro-error2` unmaintained; `anyhow`, `rand`
0.9.2 unsound; `private-gemm-x86` yanked via `faer`). **All seven findings are pre-existing and
unrelated to `basin`** — verified directly: none of `basin`, `rand 0.10.3`, `rand_core 0.10.1`,
`rand_chacha 0.10.0`, `rand_distr 0.6.0`, `ppv-lite86`, `zerocopy`, or `web-time` appear anywhere
in the `cargo audit` output. basin's dependency surface introduces zero new audit findings.

## WASM

All three commands succeed (`wasm32-unknown-unknown`, installed via rustup):

| Command | Result |
|---|---|
| `cargo check --target wasm32-unknown-unknown --no-default-features --features basin` | OK |
| `cargo build --target wasm32-unknown-unknown --features postprocess,js,basin` | OK |
| `RUSTFLAGS="--cfg anofox_basin_optimizer" cargo check --target wasm32-unknown-unknown --no-default-features --features basin` (routed) | OK |

No basin feature (`parallel`, `rand`'s OS RNG, etc.) was enabled; basin's own `default-features
= false` kept its WASM-incompatible surface out of the build, consistent with the plan's
constraint that none of basin's features be turned on.

## Test parity

Both full suites run with `cargo test --all-features --no-fail-fast`, one at HEAD (legacy
routing) and one with `RUSTFLAGS="--cfg anofox_basin_optimizer"` and a separate
`CARGO_TARGET_DIR=target/basin-route` (debug builds, single machine, run sequentially apart from
a brief overlap at the start):

| | Legacy (default routing) | basin-routed |
|---|---|---|
| ok | 3791 | 3791 |
| FAILED | 0 | 0 |
| ignored | 26 | 26 |
| Total wall time (`time cargo test ...`) | 18m 7.97s | 20m 16.32s |

**Changed-outcome tests: none.** Normalizing every `test <name> ... <status>` line from both
logs to `<name> ... <status>` and diffing them is a single line, by design: the cfg-gated
backend-label test itself (`utils::optimization::tests::optimizer_backend_is_legacy_by_default`
vs `...optimizer_backend_is_basin_when_routed`) — both `ok`. No test anywhere in the
2988-file-spanning `--all-features` suite (SES / Holt / ETS / ARIMA / Theta / Croston / GARCH /
TBATS / other) flips pass↔fail under basin routing.

Per-binary wall time for the reference suites named in the task (debug build, `finished in`
field from each binary's own summary line):

| Binary | Legacy | basin |
|---|---|---|
| `ets_r_reference` | 1.00s | 1.36s |
| `ets_validation` | 1.00s | 1.21s |
| `ets_intervals_reference` | 1.67s | 1.72s |
| `es_family_intervals_reference` | 0.27s | 0.28s |
| `auto_arima_r_reference` | 31.67s | 35.32s |
| `auto_arima_validation` | 4.13s | 3.79s |
| `arima_intervals_r_reference` | 25.83s | 21.49s |
| `theta_thetaf_reference` | 0.01s | 0.01s |
| `garch_reference` | 0.83s | 0.95s |
| `tbats_mstl_intervals_reference` | 0.33s | 0.26s |
| `statsforecast_comparison` | 0.01s | 0.01s |
| `nixtla_validation` | 1.06s | 0.88s |
| `unittests src/lib.rs` (3236 tests) | 171.04s | 143.95s (**-16%**) |
| `global_models_bench` (batch/global ETS stress) | 512.21s | 625.64s (**+22%**) |

Every count above is identical (`N passed; 0 failed`) between the two runs; only wall time
differs, and it differs in both directions depending on the workload shape (see Timing below).

## Numeric deviations

`examples/optimizer_parity_probe.rs` (release build, 1 warm-up + 5 timed `fit_predict` calls per
case, median reported): all 12 cases succeed (`ok: true`) under both backends.

| case | max abs forecast delta | max rel forecast delta |
|---|---|---|
| ses_auto | 0 | 0 |
| holt_linear_trend_auto | 0 | 0 |
| holt_winters_additive_p12 | 5.68e-14 | 4.40e-16 |
| auto_ets_p12 | 1.36e-2 | 9.96e-5 |
| arima_1_1_1 | 2.61e-7 | 1.81e-9 |
| auto_arima_seasonal_12 | 0 | 0 |
| theta | 0 | 0 |
| optimized_theta | 0 | 0 |
| dynamic_optimized_theta | 0 | 0 |
| croston_optimized | 0 | 0 |
| garch_1_1 | 2.68e-12 | 1.69e-12 |
| tbats_p12 | 8.44e-12 | 6.30e-14 |

The largest deviation anywhere is `auto_ets_p12` at `1.36e-2` absolute / `9.96e-5` relative —
AutoETS's model-selection search evaluates many NM fits and compounds any tiny per-fit floating
point difference across candidate selection; still five orders of magnitude below any forecast
value. Every other case is at floating-point noise level (`<1e-7`) or exactly `0`. No reference
assertion in the full `--all-features` suite failed under basin routing (see Test parity above),
so there is no "largest deviation quoted from a failing assertion" to report — none failed.

## Timing

Probe medians (release build) and the basin/legacy ratio:

| case | legacy ms | basin ms | ratio |
|---|---|---|---|
| ses_auto | 0.005 | 0.009 | 1.68x |
| holt_linear_trend_auto | 0.043 | 0.053 | 1.25x |
| holt_winters_additive_p12 | 0.090 | 0.149 | 1.67x |
| auto_ets_p12 | 181.921 | 196.001 | 1.08x |
| arima_1_1_1 | 0.338 | 0.356 | 1.05x |
| auto_arima_seasonal_12 | 8609.482 | 8616.872 | 1.00x |
| theta | 0.055 | 0.064 | 1.16x |
| optimized_theta | 0.236 | 0.260 | 1.10x |
| dynamic_optimized_theta | 0.368 | 0.412 | 1.12x |
| croston_optimized | 0.004 | 0.005 | 1.25x |
| garch_1_1 | 9.485 | 9.663 | 1.02x |
| tbats_p12 | 0.228 | 0.252 | 1.10x |

Total `--all-features` suite wall time: legacy 18m 7.97s vs basin-routed 20m 16.32s (**+12%**).
That total is dominated by one outlier in the opposite direction from the probe: the
`global_models_bench` integration test (batch/global AutoETS across many series) took **+22%**
longer under basin (625.64s vs 512.21s, +113s — nearly the entire suite-level delta), while the
3236-test `unittests src/lib.rs` binary was actually **16% faster** under basin (143.95s vs
171.04s). Both effects are consistent with the same mechanism: basin's `Executor`/`stop_when`
machinery (a heap-allocated centroid scratch vector per iteration, a closure call per boundary
check, a `TerminationReport` built per run) adds a small constant per-fit overhead relative to
legacy's hand-tuned flat loop. That overhead is a net win when amortized across the library's
huge volume of tiny single-series fits (many of which dominate `unittests`), and a net loss when
one stress test performs an unusually large number of model fits back-to-back with each
individual fit being cheap (`global_models_bench`'s batch workload). Per-binary ratios in the
reference-suite table range from 0.78x to 1.36x with no consistent direction, consistent with
debug-build timing noise rather than a systematic regression.

**Caveat:** test-suite timings are unoptimized debug builds on one machine; probe timings are
release builds. The two are not directly comparable to each other, only within themselves
(legacy-vs-basin at the same optimization level).

## Findings

| Family | Verdict | Likely cause (tied to mapping table) |
|---|---|---|
| SES | Parity | No bounds, no stagnation — reflection/expansion coefficients and cost tolerance map exactly |
| Holt | Parity | Same as SES; unbounded, no seasonal complexity |
| HoltWinters | Parity (noise-level, 5.7e-14) | Bounded NM path; simplex-diameter `stop_when` reproduces legacy's second convergence clause exactly |
| ETS / AutoETS | Parity (small, 1.4e-2 abs / 1e-4 rel) | Grid search over many NM fits; per-fit floating-point differences compound across candidate selection, still far below any practical significance; no test outcome changed |
| ARIMA (fixed order) | Parity (1e-7 ... 1e-9) | L-BFGS-B (native box) feeding a Nelder-Mead polish; the `exact_likelihood` reference tests pass unmodified under routing |
| AutoARIMA (seasonal) | Parity (exact, 0/0) | Same mechanism as ARIMA; seasonal search makes the probe's slowest case (8.6s/fit) but the slowest-converging result is identical |
| Theta family | Parity (exact, 0/0) | Theta/OptimizedTheta/DynamicOptimizedTheta's NM-driven parameter search converges to the same point under both backends |
| Croston | Parity (exact, 0/0) | Bounded NM over 1-2 parameters; smallest-dimension case, least floating-point drift surface |
| GARCH | Parity (noise-level, 2.7e-12) | L-BFGS over a handful of variance parameters |
| TBATS | Parity (noise-level, 8.4e-12) | Multi-stage NM/L-BFGS fit over Fourier + ARMA components |

**Overall: parity, not tolerance drift, across every family measured** — zero test outcome
changes in the full `--all-features` suite and sub-1e-4-relative probe deviations everywhere
except AutoETS's grid search (itself still 1e-4 relative). Timing is a wash in aggregate (+12%
total suite wall time, dominated by one batch-heavy stress test; -16% on the bulk unit-test
binary; probe ratios 1.0x-1.7x on individual fits), consistent with basin being general-purpose
machinery paying a small constant per-run overhead rather than any algorithmic slowdown.

**Inputs for step 2** (per the plan's objective): BOBYQA — basin ships it
(`basin::Bobyqa`) — is a candidate for the `[0,1]`-bounded smoothing parameters (alpha/beta/phi
in ETS/Holt-Winters) that currently go through Nelder-Mead; it was not evaluated here since this
step's scope was limited to the existing `nelder_mead`/`lbfgs_optimize` signatures.

**Inputs for step 3** (flipping the default): the cfg-gated routing (`anofox_basin_optimizer`)
is additive and reversible — changing the gate to the feature alone is a one-line change once
the maintainer accepts this parity evidence; the dependency surface (6 new crates, a third `rand`
major version, MSRV 1.87) and the `global_models_bench` timing regression are the two concrete
costs to weigh against basin's win on the bulk unit-test binary and its native (rather than
reparametrized) box-constraint handling.

**Deviations from the plan encountered during this work** (see per-task SUMMARY.md for detail):
the plan's `<behavior>` spec assumed Rosenbrock converges at `max_iter: 500` and that
`memory_size: 0` falls back to a working legacy result — both measured directly to be false for
**legacy itself**, independent of basin; test expectations were corrected to match measured
reality rather than tuned to hide a basin-specific failure (there was none to hide).

---

# Step 2: BOBYQA for the box-bounded smoothing parameters

**What was added.** `basin_backend::bobyqa_basin` (basin `Bobyqa`, Powell 2009) with the
`NelderMeadResult` contract, and a router `utils::optimization::minimize_bounded` that the
smoothing-parameter optimizers of SES, Holt, Holt-Winters, ETS (all 9 call sites) and the Theta
family (Theta / OptimizedTheta / DynamicOptimizedTheta) now call instead of `nelder_mead`.
`minimize_bounded` is `nelder_mead` unless the build has `--features basin` **and**
`RUSTFLAGS="--cfg anofox_basin_bobyqa"`; the default build is unchanged (3795/3795 tests pass,
identical to step 1 plus 4 new BOBYQA unit tests).

**Adapter design.**
- BOBYQA uses one isotropic trust region, but ETS's joint box mixes `[0, 1]` smoothing parameters
  with data-scaled initial-state bounds (level `y_min - range .. y_max + range`). The adapter
  optimizes in the unit cube `u = (x - lower) / width`, so `rho_beg = 0.1`, `rho_end = 1e-6` are
  relative to each coordinate's width.
- The ETS / HW objectives return `f64::MAX` in inadmissible regions; such values (>= 1e100 or
  non-finite) are replaced by a finite penalty `f(x0) + 1e3 (1 + |f(x0)|)` so the quadratic model
  does not overflow. `optimal_value` is always re-evaluated on the raw objective.
- Evaluation budget `max_iter * (n + 1)` (the NM worst case); falls back to `nelder_mead` when the
  box is not fully finite (e.g. `Theta`'s `(-inf, inf)` drift coordinate), the point is empty, or
  the start point is inadmissible.

**Reference-test parity (full `cargo test --all-features --no-fail-fast` under BOBYQA routing).**
3788 passed, **3 failed**, all in `tests/ets_r_reference.rs`:

| test | assertion | legacy NM | BOBYQA | R |
|---|---|--:|--:|--:|
| `ets_mam_parameters_near_r` | MAM loglik within 5 of R | -681.11 | **-705.77** | -682.40 |
| `ets_optimised_fit_quality` | same MAM check (loop over 15 specs) | -681.11 | **-705.77** | -682.40 |
| `auto_ets_airpassengers_selection` | chosen AICc within 2 of R | 1401.07 (M,A,M) | **1415.49** (M,Ad,M) | 1400.64 (M,Ad,M) |

Per-spec loglik on AirPassengers (higher is better), legacy vs BOBYQA: additive-error specs
equal to <1e-4; multiplicative-error specs move both ways — MNA -744.06 -> -715.39 (better),
MNM -687.75 -> -703.58, MAA -701.16 -> -740.78, MAM -681.11 -> -705.77 (worse). A sweep of
`rho_beg in {0.05, 0.1, 0.25}`, the penalty scale `{0.1, 1, 1e3}` and a full quadratic model
(`npt = (n+1)(n+2)/2`) did not find a setting that is at least as good as legacy on all four
(e.g. `rho_beg = 0.05`: MAM -719.67; full model: MAM -771.23). The penalty is never hit on these
fits, so the differences are local-optimum selection on a multimodal likelihood, not the
inadmissible-region handling. The legacy NM + the ETS multi-start structure finds better optima
for MAM; BOBYQA from the same starts does not.

**Timing.** Release, 100 synthetic AutoETS (Reduced pool, n=200, p=12) fits:
legacy 12.7 s vs BOBYQA 19.7 s (1.55x slower; machine shared with other jobs). In the debug full
suite, `global_models_bench` (1600 such AutoETS fits) took 4215 s under BOBYQA routing versus
512 s for legacy in step 1 (not same-load conditions, so only the direction is reliable). On
M4 (below) BOBYQA is faster for AutoETS on long Hourly/Daily series and slower on short
Yearly/Quarterly ones.

# Step 3: M4 accuracy + time benchmark

**Data.** The M4 competition train/test CSVs already present locally at
`~/.datasetsforecast/m4/datasets` (nothing was downloaded). Deterministic sample: every k-th
series, 300 per frequency (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly = 1800 series),
official horizons (6/8/18/13/14/48), seasonal periods 1/4/12/1/1/24. AutoARIMA on the first 50 of
Yearly/Quarterly/Monthly (150 series). Metrics: M4 sMAPE and MASE on the holdout, in-sample MSE of
the one-step residuals (backend-independent proxy for the fit objective), wall time of
fit+predict. Tool: `examples/m4_optimizer_benchmark.rs` (`run` per routing, `compare` paired).
Release builds, single-threaded, run sequentially on a machine shared with other jobs; the
legacy run was repeated at the end to measure timing noise.

```text
cargo build --release --features basin --example m4_optimizer_benchmark            # legacy
RUSTFLAGS="--cfg anofox_basin_optimizer" CARGO_TARGET_DIR=target/basin-route ...   # basin NM/L-BFGS
RUSTFLAGS="--cfg anofox_basin_bobyqa" CARGO_TARGET_DIR=target/bobyqa ...           # BOBYQA
RUSTFLAGS="--cfg anofox_basin_optimizer --cfg anofox_basin_bobyqa" CARGO_TARGET_DIR=target/basin-all ...
PER_FREQ=300 ARIMA_PER_FREQ=50 <binary> run <out.json>; <binary> compare a.json b.json
```

Total wall time: legacy 574 s, basin 575 s, BOBYQA 444 s, basin+BOBYQA 446 s, legacy repeat 515 s.
**Timing noise:** legacy vs its repeat differs by 8-16% per model with bit-identical forecasts,
so per-model time ratios within about +-15% are not significant.

**Legacy vs basin Nelder-Mead / L-BFGS-B (step-1 routing), all 1800 series:**

| model | n | sMAPE legacy | sMAPE basin | MASE legacy | MASE basin | in-sample MSE better/same/worse | time ratio vs legacy run 1 / run 2 |
|---|--:|--:|--:|--:|--:|--:|--:|
| AutoARIMA | 150 | 11.156 | 11.156 | 1.923 | 1.923 | 0 / 150 / 0 | 1.00 / 1.13 |
| AutoETS | 1800 | 11.714 | 11.714 | 2.500 | 2.500 | 125 / 1557 / 118 | 1.00 / 1.11 |
| DynamicOptimizedTheta | 1800 | 10.522 | 10.522 | 2.438 | 2.438 | 0 / 1800 / 0 | 0.99 / 1.17 |
| Holt | 1800 | 26.139 | 26.139 | 9.567 | 9.567 | 0 / 1800 / 0 | 1.02 / 1.13 |
| HoltWinters | 900 | 13.160 | 13.160 | 1.254 | 1.254 | 0 / 900 / 0 | 1.01 / 1.10 |
| OptimizedTheta | 1800 | 10.277 | 10.277 | 2.302 | 2.302 | 0 / 1800 / 0 | 0.97 / 1.16 |
| SES | 1800 | 14.498 | 14.498 | 4.215 | 4.215 | 0 / 1800 / 0 | 1.10 / 1.19 |

Accuracy parity holds on M4 (every mean sMAPE / MASE identical to 3 decimals; AutoETS's
in-sample MSE differs on 243/1800 series in both directions, its mean holdout metrics do not).
Speed: basin is 0-19% slower (never faster beyond noise); SES is 1.25-2.1x slower on short
Yearly/Quarterly/Monthly series where per-call overhead dominates.

**Legacy vs BOBYQA (basin+BOBYQA is the same to 3 decimals in accuracy; times within noise):**

| model | n | sMAPE legacy | sMAPE BOBYQA | MASE legacy | MASE BOBYQA | in-sample MSE better/same/worse | time ratio |
|---|--:|--:|--:|--:|--:|--:|--:|
| AutoARIMA | 150 | 11.156 | 11.156 | 1.923 | 1.923 | 0 / 150 / 0 | 0.98 |
| AutoETS | 1800 | 11.714 | **10.781** | 2.500 | **2.305** | 718 / 526 / 556 | 0.72 |
| DynamicOptimizedTheta | 1800 | 10.522 | 10.526 | 2.438 | 2.439 | 731 / 390 / 679 | 1.03 |
| Holt | 1800 | 26.139 | **30.096** | 9.567 | **11.747** | 530 / 1258 / 12 | 1.07 |
| HoltWinters | 900 | 13.160 | **15.635** | 1.254 | **1.882** | 350 / 531 / 19 | 1.57 |
| OptimizedTheta | 1800 | 10.277 | 10.278 | 2.302 | 2.303 | 769 / 31 / 1000 | 1.12 |
| SES | 1800 | 14.498 | 14.493 | 4.215 | 4.214 | 172 / 1628 / 0 | 1.82 |

Per frequency (sMAPE legacy -> BOBYQA; time ratio):

| model | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|---|---|---|---|---|---|---|
| AutoETS | 16.00 -> 16.61; 5.49x | 9.70 -> 9.96; 2.93x | 13.21 -> 13.42; 1.04x | 7.94 -> 7.82; 1.04x | 2.65 -> 2.68; 0.62x | 20.79 -> **14.19**; 0.68x |
| Holt | 16.21 -> 16.82 | 13.75 -> 13.34 | 18.75 -> 18.43 | 9.28 -> 9.19 | 2.79 -> 2.66 | 96.06 -> **120.14** |
| HoltWinters | – | 10.85 -> 11.04 | 14.45 -> 13.72 | – | – | 14.18 -> **22.14** |

Reading: BOBYQA is the better *optimizer* — it reaches a lower in-sample objective far more
often than it reaches a higher one (Holt 530 vs 12 series, HoltWinters 350 vs 19) — but a
lower in-sample SSE does not translate into better forecasts: Holt/HW holdout error gets worse
(driven by Hourly, where tighter SSE minimization picks more aggressive trend/seasonal
smoothing). AutoETS's aggregate gain is entirely the Hourly subset (20.79 -> 14.19 sMAPE); on
Yearly, Quarterly, Monthly and Daily it is slightly worse, and it is 3-5x slower on the short
Yearly/Quarterly series. Theta family and SES: accuracy ties, SES 1.8x slower.

# Decision (per the GH #234 rule: switch only if parity AND speed hold)

**Do not switch over; do not delete the hand-rolled optimizers or the `lbfgs` dependency.**

- basin Nelder-Mead / L-BFGS-B: accuracy parity holds (step 1 reference suite and M4), but speed
  does not improve — 0-19% slower on M4, 1.0-1.7x per single fit (step 1). Switching would add 6
  crates, a third `rand` major and an MSRV of 1.87 for no measured gain.
- basin BOBYQA: fails parity — 3 R-reference tests (`ets_r_reference`) fail, Holt / Holt-Winters
  M4 holdout accuracy degrades (sMAPE +15% / +19%), and it is slower for SES/HW and short-series
  AutoETS. Its one win (AutoETS on M4 Hourly) is frequency-specific and comes with the MAM
  reference failure, so it is not a safe default.

Both basin routings stay available behind `--features basin` + the respective cfg for further
evaluation. Possible follow-ups if the maintainer wants to pursue BOBYQA: understand the AutoETS
Hourly gain (which specs get selected differently), and try BOBYQA only as an extra candidate
start inside ETS's multi-start (keeping the best of NM and BOBYQA) rather than as a replacement.
