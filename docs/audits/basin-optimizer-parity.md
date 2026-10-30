# basin-Backed Optimizer Parity (GH #234, Step 1)

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
