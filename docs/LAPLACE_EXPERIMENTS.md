# Laplace experiments — post-mortem (v0.15.2 → 2026-07-21)

*Written 2026-07-14 after the skaters-parity work, extended 2026-07-21 with the multi-α SH pool experiments. Consolidated engineering post-mortem covering ~15 experiments across five phases: what we tried, what worked, what didn't, and the meta-lessons.*

## Context

Starting point v0.13.0 `LaplaceForecaster::auto()` at rank 11 on the fev-27 leaderboard-comparable 23-dataset subset (MASE 1.723). After the 2026-07-21 multi-α SH pool work we're at rank ~6-7 (MASE 1.4149 with the current-best recipe) — past Nixtla `auto_ets` and within 0.6 % of Moirai-Base (GPU foundation model).

## Summary table

| Experiment | Release | Result | Cost |
|---|---|---|---|
| Seasonal batch init as default on `.with_seasonal(p)` | v0.15.2 | ✅ Fixes #198 (2.11× → 8.91× ratio) | 0 |
| Precision-weighted ensemble port (issue #198 fix attempt) | v0.15.2 | ❌ Partial help, reverted; root cause was initialisation not weighting | ~200 LOC written, reverted |
| Scoring horizon (`with_scoring_horizon(h)`) | v0.15.3 | ✅ Small but consistent (−0.3 %) | ~30 LOC |
| Scoring window (`with_scoring_window(w)`) | v0.15.3 | ✅ Big win, especially combined (`.auto()` −5.1 % at W=7) | ~50 LOC |
| Stacking ridge λ sweep (env-driven) | v0.15.3 | ❌ No λ makes `.with_stacking()` non-regressive on fev-27 | Only harness edits |
| `fast_slow` leaf family | v0.15.4 | ✅ Small MASE, big WQL (`.skaters()` −24 % WQL) | ~50 LOC + wrapper wiring |
| MultiScaleLaplace `DistributionalForecaster` + scH/scW pass-through | v0.15.4 | ✅ **−11.3 % MASE** on fev-27 leaderboard subset | ~200 LOC |
| Parade PIT + GPD tails (full port) | v0.15.4 | ⚠ Works, 0 % benefit on fev-27, 17× fit-time cost | ~500 LOC |
| `MultiScaleLaplace + scH + scW=14` on M5 | v0.15.4 | ❌ Regresses M5 retail by +8.7 % MASE / +9.5 % WAPE | Discovered post-release |
| Scoring window sweep + learning-rate sweep on `.skaters()`/MS | 2026-07-20 | ✅ `sw=10` beats `sw=14` (−0.4 %); `η=0.20` beats default `η=0.5` (−0.85 % on MS) | Harness only |
| Per-phase Holt (SH) leaf single-α on MS | 2026-07-21 | ✅ SH(0.5, 0.2) yields −0.65 % geomean; killer wins on trending seasonal (tourism_quarterly −8.9 %) | ~120 LOC (new `SeasonalHoltLeaf`) + builder wiring |
| Per-phase Holt multi-α pool on MS | 2026-07-21 | ✅ **3-α pool `{(0.3,0.1), (0.5,0.2), (0.7,0.3)}` beats single-α by −0.33 %, total −0.97 % vs no-SH** | Vec-typed builder change on `MultiScaleLaplace` |
| Partial-moment (up/down asymmetric variance) leaf | 2026-07-21 | ❌ Neutral to +0.05 %; single-Gaussian-per-leaf arch prevents true two-piece predictive | ~150 LOC, reverted |
| Streaming ridge stacker prototype | 2026-07-21 | ❌ +66 % regression on yearly panels (tail too short for stable weights); non-neg-constrained variant deferred | Probe example only, no crate change |

## What worked

### `.with_scoring_window(w)` — the sleeper hit (v0.15.3)

The single biggest surprise. Replaces the cumulative `Σ logpdf` in `cum_log_liks[i]` with a moving sum over the last `w` observations. Softmax weights then reflect only recent leaf performance instead of the entire training history.

**Why it works**: streaming leaves accumulate log-lik under a slowly-changing residual scale. Old observations were absorbed when the leaf state was different from what it is now — their log-lik values are stale. A ring buffer of the last few dozen keeps the softmax responsive.

**Measurement** (fev-27, 500 series/dataset, aggregate MASE):
- `.auto()` baseline: 5.9335
- `+ .with_scoring_window(7)`: 5.6329 (**−5.1 %**)
- `+ .with_scoring_window(14)`: 5.6460 (−4.8 %)
- `+ .with_scoring_window(56)`: 5.7130 (−3.7 %)
- `+ .with_scoring_window(112)`: 5.7967 (−2.3 %)

Optimum is aggressive (W = 7–14). Larger windows monotonically approach the cumulative baseline.

### `MultiScaleLaplace + scH + scW=14` — the "star" that's mostly `scW=14` (v0.15.4)

Combining v0.15.3's scoring knobs with a multi-scale wrapper (each scale runs `.skaters()` on decimated data, blended per-horizon via softmax over per-scale training log-lik) gave **−11.3 % MASE** on the leaderboard-comparable subset. Originally framed as the biggest single-change improvement in project history.

**Key tuning parameters** — all discovered by measurement:
- `min_samples = 50` (dropped from 100). At 100 no fev-27 panel activates any decimated scale (degenerates to scale-1). At 30 m4_hourly's stride 24 activates with only 29 obs and the 30-leaf sub-pool can't fit → catastrophic regression.
- Drop the `⌈√k⌉` stride when a period is set. A coprime stride aliases the seasonal signal; measured on fev-27 as −55 % m4_hourly / −50 % tourism_monthly regression at v1 of the port.
- Scale-1 sub-forecaster receives the period hint via `.auto_with_seasonal_period(p)` — matches the fev-27 harness call pattern so multiscale isn't handicapped by missing the v0.15.1 seasonal-period fix.
- Sub-forecasters wrap `.skaters()`, not `.auto()` — the wider pool wins on m4_hourly and long-N panels.

**⚠ Attribution correction (measured 2026-07-20 on full fev-27, SAMPLE_PER=500)**:
The v0.15.4 headline of "biggest single-change improvement" was
overstated. On the 23-set leaderboard subset, the marginal contribution
of MultiScale over the plain-`.skaters()` recipe with the same scoring
knobs is only **0.57 %** geomean MASE:

| Recipe | Geomean MASE (23-set) | Win-rate |
|---|---:|---:|
| `.skaters().auto_with_seasonal_period(P)` (baseline) | 1.6526 | 3/23 |
| `+scoring_horizon(P)` alone (no `scW=14`) | 1.6411 | 1/23 |
| `+scoring_horizon(P) + scoring_window(14)` | 1.4655 | 10/23 ⭐ |
| `MultiScale + scH + scW=14` (v0.15.4 "winner") | 1.4572 | 8/23 |
| `MultiScale + scH` (no `scW=14`) | 1.6410 | 1/23 |

The bulk of the gain (−11.3 % of the −11.7 % over baseline) is from
`.with_scoring_window(14)`, which was **v0.15.3**. MultiScale on top
adds 0.57 % geomean and *loses* the win-rate. The v0.15.4 story is
best rewritten as: **v0.15.3's `scoring_window(14)` was the star;
v0.15.4's MultiScale wrapping is a small, situational refinement that
buys the last 0.6 % at ~15 % higher fit-time cost.**

Practical implication for users: pick MultiScale if you already have
the wrapper for other reasons or want the last 0.5 %. Otherwise, the
plain-`.skaters().auto_with_seasonal_period(P).with_scoring_horizon(P).with_scoring_window(14)`
recipe is 99 % as good, simpler API, faster. See
`docs/LAPLACE_PARAMETER_GUIDE.md` §"Step 4" for the current guidance.

**Meta-lesson**: attribution matters. When a compound recipe wins,
run the "just the last knob" A/B before crediting the outermost
wrapper. In v0.15.4 we bundled MultiScale + scH + scW together and
credited MultiScale; the honest attribution was to scoring_window (an
earlier feature) and the wrapper contributed marginal refinement.

## What didn't work (and why)

### Precision-weighted ensemble (v0.15.2, reverted)

**Motivation**: issue #198 reported `.auto()`/`.skaters()` collapsing to near-flat on established seasonal series even though `seasonal_ema` is in the pool. Skaters ships a `precision_weighted_ensemble` that weights leaves by inverse per-horizon MSE. Ported ~200 LOC.

**Measurement** (issue #198 synthetic, 60-month retail with 8.91× target ratio):
- `.auto().with_seasonal(12)` baseline: 2.11×
- `+ .with_seasonal_batch_init()`: **8.91×** (recovers full swing)
- `+ .with_precision_weights(12)`: 3.27× (partial improvement)

**Root cause**: `seasonal_ema`'s per-horizon MSE beats level trackers by only ~1.8× on this synthetic — precision-inverse weighting spreads across all near-equal-MSE leaves instead of committing. The pathology is **cold-start initialisation**, not the weighting mechanism. Once `seasonal_batch_init` pre-fits the leaf from the training window's last cycle, softmax converges immediately.

**Lesson**: verify the pathology before implementing the fix. We spent 200 LOC on the wrong intervention. The right question was "why is `seasonal_ema` slow to win the softmax?" not "how do we work around it losing?".

**Reference for depth**: the python skaters reference (`search(k=12)`) itself flat-lines at 1.05× on the same synthetic — worse than our 2.11× baseline. So this was our own initialisation problem, not something to import from upstream.

### Stacking ridge λ sweep (v0.15.3, no code shipped)

`LaplaceForecaster::with_stacking()` was known to regress fev-27 at its default `ridge_lambda = 1e-4`. Hypothesis: λ is a magic constant that could be tuned. Ran an env-driven sweep across 8 orders of magnitude.

**Measurement** (SAMPLE_PER=200):

| λ | Δ MASE vs baseline |
|---|---:|
| 1e-6 | +9.2 % |
| 1e-4 (current) | +7.2 % |
| 1e0 | +7.4 % |
| 1e4 | +4.2 % |
| 1e6 | +3.9 % |
| 1e8 | +5.6 % |

Best case is still **+3.9 %** worse than baseline. The MSE-minimising OLS+simplex solve is structurally at odds with the softmax's 1-step-log-lik objective — no λ bridges the gap.

**Lesson**: sometimes the answer is that a feature just doesn't work on your benchmark. Not everything is tuneable.

### Parade + GPD tails (v0.15.4, shipped as opt-in, 0 benefit on fev-27)

Ported ~500 LOC (parade PIT tracking + generalised Pareto tail splice via censored ML). Measured:

| Model | MASE | WQL | Fit time |
|---|---:|---:|---:|
| `.skaters()` baseline | 5.9658 | 0.3005 | 12 s |
| `.skaters()` + parade + GPD tails | 5.9658 | 0.3007 | **203 s (17×)** |

Zero measurable benefit at 17× fit-time cost.

**Root cause**: fev's WQL uses quantiles `q ∈ [0.1, 0.9]`. GPD tail splice fires only outside `[0.02, 0.98]` (the frozen body region). **Every quantile the WQL metric evaluates falls inside the region GPD doesn't touch.** Definitive metric-shape mismatch, not implementation error.

**Meta-lesson (see below)**: before implementing a metric-improving feature, verify the metric's q-range overlaps the feature's active range. If we'd checked `WQL_QUANTILES` in the harness before starting, we'd have known this in 30 seconds.

Shipped as opt-in for users targeting extreme-quantile calibration (anomaly detection, VaR estimation, rare-event forecasting — skaters' actual use case).

### M5 regression on the v0.15.4 winning recipe (discovered post-release)

The fev-27 winner `MultiScaleLaplace + scH + scW=14` **regresses on M5 retail** by +8.7 % MASE / +9.5 % WAPE vs plain `.skaters()`.

**Root cause**: M5 is intermittent daily count data. AID-selected Poisson / NegBin / Croston-family leaves fit these SKUs; Gaussian mixtures (what MultiScaleLaplace emits, wrapping `.skaters()`) don't. The scoring knobs' aggressive multi-step objective on decimated sub-forecasters compounds the wrong-marginal problem.

**Lesson**: fev-27 (mixed classical) and M5 (retail counts) are structurally different tasks with different best selectors. **No single recipe wins everywhere.** The v0.15.4 improvements are fev-27 improvements, not universal ones. For retail, `.auto_aid()` or `SmartForecaster` remain recommended (v0.13.0 era, unchanged).

## Upstream sync — skaters #157 bug audit (2026-07-25)

Re-check of open microprediction/skaters issues and merged PRs
surfaced upstream PR #157 (merged 2026-07-24, "Stabilize AR and garch
forecasts on non-stationary / large-magnitude series"). Three
correctness bugs, two applied to our Rust port:

| Upstream bug | Our Ar1Leaf | Our Ar2Leaf | Our GarchWrappedLeaf |
|---|---|---|---|
| 1. Non-stationary blow-up | ✅ safe (phi clamped ±0.999) | ✅ safe (project_to_stationary) | N/A |
| 2. Wrong multi-step variance | ✅ correct MA(∞) | ❌ used `σ·√h` (fixed) | N/A |
| 3. GARCH: variance of level vs deviation | N/A | N/A | ❌ used raw `y²` (fixed) |

**Bug 2 fix — `Ar2Leaf` MA(∞) variance.** The h-step variance now
follows the correct recurrence: ψ_0=1, ψ_1=φ_1, ψ_i = φ_1·ψ_{i-1} +
φ_2·ψ_{i-2}, Var[h] = σ²·Σ_{i=0..h-1} ψ_i². Pre-fix used the random-walk
form `σ·√h`, which overstated horizon-h uncertainty for stationary
AR(2) and made the variance identical for any φ_1, φ_2 in the stationary
triangle. Regression test:
`tests/laplace_component_robustness.rs::ar2_h_step_variance_bounded_for_stationary_phis`
asserts sigma_long/sigma_short < 2.0 (pre-fix ratio was ≈ 10).

**Bug 3 fix — `GarchWrappedLeaf` shift-invariant recursion.** The
GARCH recursion now runs on **deviations from a running mean**
(`d = y - mu_running`), not raw `y²`. On level series (values ~1e5),
`α·y²` used to dominate ω and β·σ², so "volatility" became of order
`|y|` and the inverse re-inflated the mixture σ. The wrapper is now
end-to-end shift-invariant: the inner leaf is fed `(y-mu)/σ` (centered
standardized) rather than `y/σ`, and predictions add `mu` back on the
way out. Regression test:
`tests/laplace_component_robustness.rs::garch_shift_invariant_on_level_series`
asserts predictive σ stays O(1) on values around 1e6.

Upstream's headline number: GIFT-Eval `m4_yearly` WQL recovered
0.208 → 0.1195 (−43 %) — a plausibly big win for us too on
large-magnitude panels we haven't measured yet (fev-27 mostly has
values in [0, 1e3]).

Also added 8 adversarial-input tests (`tests/laplace_component_robustness.rs`)
mirroring the intent of upstream's `test_component_robustness.py`:
billion-scale inputs, stationary long-horizon variance bounds, single
spikes, level shifts. Would have caught both bugs immediately; guard
the fixes going forward.

## Skaters-issue verification tests (2026-07-17)

After v0.15.8, three open skaters issues (#86, #82, #84) each proposed a
one-off invariant that our implementation should satisfy. We wrote them
up as tests in `tests/laplace_robustness.rs` rather than shipping any
new feature — the goal is to confirm/refute each claim against our
current code and log the numbers. All three passed on first run.

### #86 — variance monotonicity across horizons

Skaters proposition: for integrated transforms, predictive variance must
be nondecreasing in `h`. Multiscale mixtures may violate this at stride
boundaries where softmax dominance shifts between scales.

Tests: `multiscale_variance_is_monotone_in_horizon_on_random_walk`
(existing, extended to H=60) and a new sibling
`skaters_variance_is_monotone_in_horizon_on_random_walk` covering the
production `.skaters()` path (H=30). Both check the worst adjacent
downward ratio against a 5 % sawtooth threshold.

**Result**:

| Path | H | Worst `var[h]/var[h-1]` |
|---|---:|---:|
| `MultiScaleLaplace::skaters(60)` | 60 | 1.0000 |
| `LaplaceForecaster::new().skaters()` | 30 | 1.0000 |

Fully monotone on a Gaussian random walk. No sawtooth at any stride
boundary. The `MultiScaleLaplace` cross-scale softmax blend in
`src/models/laplace/multiscale.rs` and the `.skaters()` mixture emitted
by `forecast_dist` are both variance-coherent by construction. Nothing
to fix.

### #82 — mid-PIT limit under parade over sticky

Skaters proposition: `sticky` represents an atom of mass `w` at value
`v` as `N(v, ε)`. Evaluating the mixture CDF at `v` gives `F(v-) + w/2`
in the `ε → 0` limit — the canonical Czado–Gneiting–Held mid-PIT.
Naive `parade(sticky(...))` should therefore yield a PIT mean of 0.5 on
lattice data without any explicit correction.

Test: `parade_pit_mean_on_lattice_series_near_half`. Random walk driven
by ±1 steps that stall with probability 0.5 (heavy repeat pattern), 600
observations, `.skaters().with_parade(4)`, check `mean(parade_pit[h=1])`
against a 0.10 tolerance around 0.5.

**Result**: PIT mean = 0.5250 on n=599 samples. Passes with slack.

**Caveat specific to us**: our parade computes PIT via
`Φ((y − μ)/σ)` — the mixture reduced to its first two moments — not the
full mixture CDF with sticky atoms. Skaters' exact mid-PIT identity
therefore doesn't apply verbatim to our implementation; the near-0.5
mean here reflects a symmetric random walk plus enough leaf averaging
to smear the atom, not the atom-projected mixture CDF construction.
Shipping the atom-projected PIT would be a code change (route the
parade through `forecast_dist` post-processing instead of raw mixture
`(mean, std)`), tracked as a future consideration if the strict
mid-PIT property matters to a downstream user.

### #84 — parade z-std as copula-deficit diagnostic

Skaters proposition: variance additivity across horizons is a copula
assumption (independence of forecast errors). Under regime drift the
h-step error terms share the post-origin information deficit, positively
correlate, and additivity understates long-horizon variance. Parade's
`z` at horizon m directly measures this: iid increments → std ≈ 1,
regime-switching → std > 1.

Test: `parade_z_std_at_long_h_larger_on_regime_switches_than_iid`. Two
800-observation panels — (a) iid Gaussian noise, (b) random walk with
vol σ toggling between 1.0 and 6.0 every 150 steps. `.skaters()` +
`with_parade(12)` on each. PIT converted to z via a local bisection
`phi_inv`. Assert both are non-collapsed (std > 0.3) AND that
regime-switch strictly dominates iid.

**Result** at h=12 (n=788 samples per panel):

| Panel | z-std at h=12 |
|---|---:|
| iid Gaussian | 1.112 |
| regime-switching σ ∈ {1, 6} | 1.501 |

iid comes in at 1.11 (near the theoretical 1.0), regime-switch comes in
at 1.50 (well above). Ordering holds decisively — the copula-deficit
signal survives our pipeline even though `.skaters()` already includes
GARCH cascades and terminal scale-mixture, which model heteroskedastic
residuals and are expected to attenuate this signal.

The elevated iid value (1.11 vs a theoretical 1.0) is consistent with
the mean/std reduction bias flagged under #82: reducing a heavy-tailed
mixture to a single Gaussian for the PIT slightly inflates the reported
z-std on any panel where the true mixture has excess kurtosis.

### Summary of the sweep

All three passed. No code fix needed for #86 (perfectly monotone) or
#84 (copula deficit signal is real and ordered correctly). #82 passed
under a loose tolerance because our parade uses a mean/std reduction
rather than the atom-projected mixture CDF from skaters' formulation —
the "mid-PIT for free" identity is stricter than what our parade
literally computes; a future issue could route parade through the full
`forecast_dist` post-processing if a downstream user needs exact
mid-PIT.

The tests are cheap (< 1 second total) and act as regression guards
against future changes to multiscale variance mixing, parade
implementation, or the sticky-lattice atom width.

### #107 — sticky calibration on tick-grid data

User report on skaters: `.skaters()` 98 % band achieved only ~94 %
coverage on ~925 k ES-futures 1-minute returns (quoted on a 0.25-point
grid). Attributed to sticky-lattice atoms concentrating probability on
grid values while the bands between grid points get underweighted. The
author asked whether `sticky=False` is the recommended setting.

Test: `sticky_vs_no_sticky_coverage_on_tick_grid_data`. Walk-forward
refit loop against `forecast_dist(1)` (NOT parade — our parade
snapshots the raw mixture pre-post-processing and therefore never sees
the sticky atom overlay, so a parade-based comparison would be
trivially identical). Synthetic tick=1.0 random walk, 500 obs, 100
walk-forward steps × three variants: `.skaters()` (auto-gate decides),
`.skaters().with_sticky()` (force on), `.skaters().no_sticky()` (force
off).

**Result** (empirical coverage of the 98 % quantile band):

| Variant | Coverage |
|---|---:|
| `.skaters()` (default) | 0.990 |
| `.skaters().with_sticky()` (forced on) | 0.990 |
| `.skaters().no_sticky()` (forced off) | 0.990 |

**All three identical, all above the 98 % target.** In our
implementation the sticky-lattice projection is *coverage-neutral* on
tick-grid random walks. The user's degradation does not reproduce.

Two candidate reasons for the divergence from skaters:
- **Auto-gate**: `.skaters()` runs `sticky_auto_gate` which turns
  sticky off when the data doesn't look discrete-count. But even the
  *forced-on* variant with the auto-gate bypassed shows no degradation,
  ruling this out as the sole explanation.
- **Sticky spike width and pool composition**: our `.skaters()` pool
  includes `terminal_scale_mixture` cascades whose broad body absorbs
  the atom overlay. Combined with our sticky spike-std default (a
  fraction of the mixture body), the atoms add negligible tail mass on
  a smooth random walk.

**Interpretation**: for tick-grid Gaussian-random-walk data on our
implementation, `sticky=on|off` doesn't matter for coverage — the user
can pick either. This is a *stronger* answer than skaters' (where
`sticky=False` was the recommendation): we hand the user a
representation-choice-free result.

**Caveats**: (a) the test uses a synthetic random walk, not real ES
futures data; a longer-tail or level-shifting real series could still
expose an effect. (b) 100 walk-forward steps gives coarse coverage
resolution (±3 pp at n=100 under a truly-98 % binomial). (c) the
walk-forward loop is O(N²) — refitting on the growing prefix for every
step; kept small to stay under 2 seconds per test.

### #85 — CRPS well-posed under fat tails (Cauchy input)

Skaters proposition: CRPS is a projection in the Cramér distance and
is finite whenever `E|X| < ∞` (first moment only). Log-lik is a KL
projection, moment-matches for location-scale families, and requires a
finite second moment. Under symmetric α-stable input with α < 2 the
variance is infinite; the log-lik-scored leaves' scale estimate
diverges, but the CRPS objective remains well-posed. The ensemble
output must therefore stay well-formed if the pool carries a
CRPS-scored terminal.

Test: `cauchy_input_produces_well_formed_forecast`. 500 samples of
Cauchy(0, 1) via inverse-CDF `tan(π(u − ½))` (α=1, the canonical
infinite-variance case: no second moment). Fit `.skaters()` and check
that `forecast_dist(3)` at every horizon has finite mean, finite
positive std, and passes the standard `assert_wellformed` (finite
`logpdf`, `cdf ∈ [0, 1]`, monotone finite quantiles at `p ∈
{0.001, 0.25, 0.5, 0.75, 0.999}`).

**Result** (sample max |y| = 170 for the fixed seed):

| Horizon | mean | std | quantiles |
|---|:---:|:---:|:---:|
| h = 1, 2, 3 | finite | finite, positive | finite, monotone |

All checks pass. The ensemble survives Cauchy input — the presence of
`terminal_crps` and `terminal_scale_mixture` in the pool keeps the
final mixture well-defined even though log-lik-scored leaves would
misestimate scale in the infinite-variance regime.

**Interpretation**: our pool composition satisfies the skaters#85
invariant. This is a stronger correctness guarantee than "doesn't
overflow on large but finite values" (the existing
`monster_spike_then_recovery` and `extreme_finite_tick` tests) —
Cauchy input has *no* well-defined variance in expectation, so
survival is a stronger property than survival on any single extreme
observation.

**Caveats**: the softmax scoring in `LaplaceForecaster` is still
log-lik-based, not CRPS-based. What survives is the *output* under
Cauchy input, not the *scoring* stability. A stronger test would
verify that CRPS-scored leaves collect softmax weight while log-lik
scored leaves' cumulative log-lik diverges to −∞; that would require
CRPS-objective scoring (skaters ships this as `objective="crps"`; we
don't). Deferred as out of scope for a verification test.

### #112 — extra seasonal periods in the candidate pool

Skaters proposition: adding period 168 (hour-of-week) beyond the fixed
`{7, 12, 24}` seasonal candidates helps hourly-with-weekly data by ~3 %
CRPS on skaters' M4-hourly probe; periods 52 and 5 are absent from the
fixed pool entirely.

**Two-stage evaluation**: synthetic first (unit tests), then real
benchmark data (17 fev-27 seasonal panels + M5 subset). The synthetic
result was dramatic; the real-data result is much more modest.

#### Stage 1: synthetic tests (unit tests)

Tests: `adding_period_168_with_scoring_horizon_beats_default_pool`
(period 168 sinusoid, 700 train / 336 test) and
`adding_period_100_with_scoring_horizon_beats_default_pool` (period 100
sinusoid, 800 train / 200 test). Both compare bare `.skaters()` vs the
full recipe `.skaters().with_seasonal_multi(&[P]).with_seasonal_batch_init().with_scoring_horizon(P)`.

| Signal | Bare `.skaters()` | +multi[P] alone | +full recipe |
|---|---:|---:|---:|
| period 168 sinusoid | 25.304 | 25.304 (Δ 0.000) | **4.170** (−83.5 %) |
| period 100 sinusoid | 36.137 | 36.137 (Δ 0.000) | **0.429** (−98.8 %) |

Key mechanism: `with_seasonal_multi(&[P])` alone is inert — the added
seasonal-EMA leaf gets softmax weight 0.0000 because `slow_std` wins
the 1-step log-likelihood competition. The full recipe works because
`.with_scoring_horizon(P)` redirects softmax to horizon-P accuracy,
at which point `theta` wins and captures the seasonal structure.

#### Stage 2: real fev-27 + M5 (`scratchpad/recipe_probe`, 22 s)

The **synthetic baseline was bare `.skaters()`** — but the fev-27
harness (and any reasonable user with a known period) uses
`.skaters().auto_with_seasonal_period(P)` instead, which already
enables `seasonal_period = Some(P)` + batch init from v0.15.2. So on
real benchmarks the baseline is stronger and the delta shrinks.

Full-scale numbers (SAMPLE_PER = 300 for fev, 500 for M5):

| Dataset | n | Baseline | +sh(P) | +recipe | Δ sh | Δ recipe |
|---|---:|---:|---:|---:|---:|---:|
| m3_monthly       | 300 | 0.7457 | 0.7461 | 0.7459 | +0.05 % | +0.03 % |
| m1_monthly       | 300 | 1.3231 | 1.3261 | 1.3448 | +0.23 % | **+1.64 %** |
| m4_monthly       | 300 | 1.2550 | 1.2543 | 1.2669 | −0.06 % | +0.95 % |
| tourism_monthly  | 300 | 2.5209 | 2.4546 | 2.5181 | **−2.63 %** | −0.11 % |
| cif_2016         |  69 | 1.3689 | 1.3675 | 1.3615 | −0.10 % | −0.54 % |
| fred_md          | 107 | 0.5222 | 0.5093 | 0.5093 | **−2.47 %** | **−2.47 %** |
| hospital         | 300 | 0.7808 | 0.7818 | 0.7806 | +0.13 % | −0.03 % |
| car_parts¹       | 300 | 2 252 253 | 2 252 253 | 2 252 253 | +0.00 % | +0.00 % |
| m3_quarterly     | 300 | 1.5893 | 1.5893 | 1.5722 | −0.00 % | −1.08 % |
| m1_quarterly     | 173 | 1.9929 | 1.9928 | 1.9838 | −0.00 % | −0.45 % |
| m4_quarterly     | 300 | 1.3474 | 1.3480 | 1.3479 | +0.04 % | +0.04 % |
| tourism_quarterly | 300 | 2.5723 | 2.5781 | 2.6453 | +0.22 % | **+2.84 %** |
| m4_daily         | 300 | 1.1967 | 1.1978 | 1.1978 | +0.10 % | +0.10 % |
| m4_hourly        | 300 | 2.1583 | 2.0351 | 2.1944 | **−5.71 %** | +1.67 % |
| australian_electricity² | 5 | 2.3314 | 2.3314 | 2.3314 | +0.00 % | +0.00 % |
| exchange_rate²   |   8 | 1.6815 | 1.6815 | 1.6815 | +0.00 % | −0.00 % |
| **m5_top1000**   | **500** | **1.0239** | **1.0198** | **1.0204** | **−0.40 %** | **−0.34 %** |

¹ *car_parts is spare-parts intermittent data: near-zero seasonal-naive
denominator makes MASE explode identically for all variants — not a
signal.*
² *length filter kept only 5–8 series; noise-only sample.*

**Mean per-dataset Δ MASE**: `+sh(P)` **−0.62 %**, `+recipe` **+0.13 %**.
**Win rate**: `+sh(P)` beats baseline on 8/17, `+recipe` on 8/17.
**M5 is flat** (−0.4 % and −0.3 %).

#### Interpretation

The synthetic tests overstate. Three reasons:

1. **The synthetic baseline was bare `.skaters()`, not the recommended
   `.skaters().auto_with_seasonal_period(P)`.** `.auto_with_seasonal_period(P)`
   already enables the seasonal-EMA leaf with batch init (v0.15.2 fix),
   so the whole `+multi + batch_init` half of the "full recipe" is
   redundant on the correct baseline.

2. **`+recipe` actively regresses on some panels**: tourism_quarterly
   (+2.84 %), m1_monthly (+1.64 %), m4_hourly (+1.67 % vs baseline,
   even though `+sh(P)` alone gave −5.71 %). The double-add of a
   seasonal_ema leaf (once via `auto_with_seasonal_period`, once via
   `with_seasonal_multi`) is a footgun.

3. **`+sh(P)` alone is the useful ingredient**, not the seasonal-multi
   layer. It gives real single-digit-percent gains on the panels where
   long-horizon accuracy diverges from 1-step LL (m4_hourly −5.71 %,
   tourism_monthly −2.63 %, fred_md −2.47 %) and is neutral or
   slightly negative elsewhere.

#### Recommendations

**Do not ship** `with_seasonal_multi(&[P]) + with_seasonal_batch_init +
with_scoring_horizon(P)` as a "recipe" — it regresses on real data.

**For users** with a strong long-horizon accuracy requirement (e.g.
hourly panels with hour-of-day cycles):
```rust
.skaters().auto_with_seasonal_period(P).with_scoring_horizon(P)
```
Expect roughly a 2–6 % MASE improvement on the panels where the
current baseline under-uses the seasonal signal (m4_hourly-shaped
data), neutral-to-mildly-negative elsewhere. Optional and case-by-case,
not a default.

**Do NOT combine with `with_seasonal_multi(&[P])`** when `auto_with_seasonal_period(P)`
is already in play — the double leaf-add hurts on tourism_quarterly
and m1_monthly.

**M5 unaffected either way.** Consistent with the existing v0.15.4
finding that M5 has different needs (AID-selected count leaves via
`.auto_aid()` / `SmartForecaster`).

**Synthetic-test caveat**: the two `adding_period_*` tests still pass,
but their assertion of a 50 %+ MAE reduction is a synthetic artifact
of the bare-`.skaters()` baseline. The tests remain useful as
regression guards for the seasonal-batch-init + scoring-horizon path,
but their headline numbers should not be quoted as expected real-data
gains.

### #91-subset — waveform-scale periodicity

Same recipe as #112 above; the period-100 test doubles as a synthetic
subset of the UCR-anomaly-archive waveform case. Same synthetic gain
(−98.8 % MAE), same real-data caveat (fev-27 result is at most −5.7 %
on m4_hourly, and only via `+sh(P)` alone, not the full recipe).

The larger #91 claim (failure appears via the anomaly head on
250-series UCR panels) still needs a full UCR-archive run and is out
of scope for a unit-test suite. If the UCR forecasting benefit
mirrors the fev-27 finding (single-digit percent, not 80 %+), the
practical answer is the same: `.auto_with_seasonal_period(P).with_scoring_horizon(P)`
with the caller supplying the detected period.

## Multi-α seasonal-Holt pool (2026-07-21)

Post-v0.15.4 push to close the gap to Nixtla `auto_ets` (1.440) on the
23-set leaderboard subset. Three tests:

**Test 1 — SH single-α sweep.** Extended `.with_seasonal_holt` to a
sweep over α = (level, trend) ∈ {(0.4, 0.15), (0.5, 0.2), (0.5, 0.3),
(0.6, 0.2), (0.6, 0.3), (0.7, 0.4)}. All lie in a flat plateau at
−0.44 % to −0.62 % vs the no-SH MS baseline. No single-α beats
(0.5, 0.2). Conclusion: single-α is a dead end.

**Test 2 — SH multi-α pool [WIN].** Extended `MultiScaleLaplace` to
accept a Vec of `(α_l, α_t)` pairs, one leaf per call. Softmax across
the pool picks per dataset:

| Variant | Full-scale 25-set geomean | vs no-SH |
|---|---:|---:|
| MS_ref (no SH) | 5.3375 | 0 % |
| MS + SH(0.5, 0.2) | 5.3029 | −0.65 % |
| MS + SH(0.3, 0.1) + SH(0.5, 0.2) | 5.2923 | −0.85 % |
| **MS + SH(0.3, 0.1) + SH(0.5, 0.2) + SH(0.7, 0.3)** | **5.2855** | **−0.97 %** |

Ex-outlier (`covid_deaths` and `car_parts` have near-zero naive-scale
denominators that dominate the geomean): 3-α pool = **1.4149** vs prior
best 1.4572 (v0.15.4) = **−2.9 %**. Per-dataset headline wins:
tourism_quarterly 1.8421 → 1.7895 (**−2.9 %**), m3_quarterly
1.2714 → 1.2556, tourism_monthly 1.5751 → 1.5454.

**Test 3 — Ridge stacking [BUST].** Prototype `RidgeStackedLaplace`:
fit 4 LaplaceForecaster variants on head (85 % of train), stream tail
via `observe()` to build (X, y) pairs, solve ridge for combining
weights, apply to full-train forecasts. **+66.5 % regression** vs MS
baseline at SAMPLE_PER=25. Failure mode: yearly panels blow up because
tail_len = 5-20 is too short for stable weights with 4 predictors
(m3_yearly 4.38 → 15.06, m4_yearly 5.47 → 23.54). Where the tail *does*
have signal, ridge actually wins (tourism_monthly 1.5090 → 1.4475).
Not shipping-quality in current form; would need non-negativity
constraints, higher λ on short series, and a min-tail-length fallback
to simple averaging to be viable. Deferred.

**Meta-lesson.** The multi-α SH pool is a repeat of the sk#113 idea 2
finding: don't optimize one shrinkage constant, seed a diverse pool and
let the softmax choose. Same shape as the `standardize_ema_alphas` +
`fast_slow_slow_alphas` recipes that already ship. Additional pool
capacity is essentially free (~4 leaves × 25 obs training cost) and
buys real gains on the panels where trend is present within a phase.

## Synthetic bake-off — clean signal vs real-world noise (2026-07-22)

Built 18 archetypes (varying length, seasonality, trend, variance,
jumps, distribution, count-ness, multi-seasonality) × 30 replicates
each = 540 synthetic series. Compared 5 models: `AutoETS`,
`AutoTheta`, `Lap.auto()`, `laplace::recommended_for` (the router),
and `MS+3SH manual` (our fev-27 SOTA). See
`examples/synthetic_bakeoff.rs`.

**Router validation**: 18/18 correct picks. The `recommended_for`
router selects the intended `RecipeKind` for every archetype (short
history → `ShortHistory`, count-like → `RetailCountAid`, heavy-tailed
→ `HeavyTailedCrps`, seasonal + long → `ContinuousMultiScale`,
non-seasonal → `ContinuousPlainSkaters`). Data-shape detection works.

**MASE result** (overall geomean, all 18 archetypes):

| Model | Geomean | Wins |
|---|---:|---:|
| **`AutoETS`** | **0.8417** | **13/18** |
| `AutoTheta` | 0.9072 | 2/18 |
| `MS+3SH manual` | 1.0374 | 1/18 |
| `recommended_for` | 1.0418 | 1/18 |
| `Lap.auto()` | 1.0806 | 1/18 |

**Where AutoETS dominates** (parametric DGP matches its assumptions):
`stationary_seasonal_*`, `seasonal_linear_trend`, `seasonal_damped_trend`,
`multi_seasonal_hourly` (**+152 %** vs router), `linear_trend_only`
(+57 %), `heteroscedastic_multi_seasonal` (+109 %).

**Where our Laplace family wins**: `random_walk`, `mean_reverting_ou`,
`level_shift_midway` (jumps), `heavy_tail_cauchy` (within 4 % of
AutoETS via CRPS). The pattern: non-parametric / regime-changing /
heavy-tailed shapes.

**The reframe.** Our fev-27 rank ~6 (MASE 1.4149) is real, but
narrower than the leaderboard framing suggests. It reflects
**real-world noise defeating AutoETS's structural assumptions**, not
Laplace being universally better. On clean-signal synthetic data the
ordering flips completely. Fev-27 panels are noisy, mixed-regime, and
heavy-tailed enough that the flexible streaming pool beats the
parametric baseline; synthetic archetypes with textbook decomposition
give the parametric baseline back its home turf.

**Practical takeaway for `SOTA_POSITIONING.md` and
`LAPLACE_PARAMETER_GUIDE.md`**: recommend `AutoETS` (or
`SmartForecaster` for cross-family routing) when the caller's data
looks like a clean trend + seasonal + Gaussian process. Reserve
`recommended_for` for messy / non-parametric / count / heavy-tailed
panels — which is what the fev-27 27-set panel actually is.

**Fit-time cost**: AutoETS is 10-100× slower than Laplace on longer
seasonal series (multi_seasonal_hourly: 685 ms vs 3-4 ms). If latency
matters and the panel isn't obviously structural, the Laplace
recipes are still the pragmatic pick.

### 2026-07-23 extension — 11 new Laplace-favoring archetypes + category segmentation

Extended the bake-off from 18 to 29 archetypes to sharpen the "when
to use Laplace" story. New archetypes deliberately target Laplace's
design targets: regime shifts (`regime_shift_flat_to_trend`),
contamination (`contaminated_seasonal`), heavy-tail-on-trend
(`student_t_trended`, `trend_jumps_heavy_tails`), extreme
intermittency (`intermittent_bursty`, `zero_inflated_seasonal`),
GARCH volatility clustering, evolving variance, fading seasonality,
bimodal regime switching, discrete tick-grid random walk.

Also added a `Category` enum (`LaplaceFavoring` / `AutoETSFavoring` /
`Neutral`) so the output segments wins-by-category — makes the "when
to use Laplace" cut visible in one glance.

**Router robustness bug fixed.** Prior `is_heavy_tailed` computed
kurtosis on RAW values, so heavy-tailed innovations on top of a
linear trend (`student_t_trended`, `trend_jumps_heavy_tails`) were
masked by trend-dominated variance. Now computes on first differences
(a crude detrending) and adds a `max-standardised-deviation > 6`
OR-trigger — Gaussian effectively never produces this in ≤ 1000 obs,
while heavy-tailed distributions reliably do. Router accuracy
climbed from 18/18 (original) → 26/29 on the extended set; three
false-negatives remain (student_t_trended, trend_jumps_heavy_tails,
garch_volatility_clustering) where sample-based kurtosis is too
noisy to trigger consistently, but their actual MASE cost vs the
right recipe is < 5 %.

**The category-segmented result** (2026-07-23):

| Category | AutoETS wins | Laplace family wins | Count |
|---|---:|---:|---:|
| Laplace-favoring | 10 | 8 (Lap.auto 1, recommended_for 3, MS+3SH 2, AutoTheta 2) | 18 |
| AutoETS-favoring | 8 | 0 | 8 |
| Neutral | 2 | 1 (AutoTheta) | 3 |

Even on Laplace-favoring, AutoETS still wins 56 % of the archetypes;
the Laplace family wins 44 %. On AutoETS-favoring, Laplace wins
0/8. The story is no longer "Laplace is worse everywhere" — it's
"Laplace is competitive in its design space and dominant in a
handful of panels; AutoETS is dominant on textbook structural DGPs
and competitive everywhere else."

**Per-archetype clear-Laplace wins** (both MASE and WQL where
applicable):

| Archetype | Metric | AutoETS | Best Laplace | Δ |
|---|---|---:|---:|---:|
| `intermittent_bursty` | MASE | 0.9826 | 0.8471 (MS+3SH) | **−13.8 %** |
| `intermittent_bursty` | WQL | 1.3019 | 1.0584 (MS+3SH) | **−18.7 %** |
| `zero_inflated_seasonal` | WQL | 1.3682 | 1.1467 (MS+3SH) | **−16.2 %** |
| `fading_seasonality` | WQL vs AutoTheta | 0.0415 | 0.0140 (recommended) | **−66 %** |
| `level_shift_midway` | MASE | 0.7332 | 0.6905 (MS+3SH) | −5.8 % |
| `mean_reverting_ou` | MASE | 1.9092 | 1.8130 (Lap.auto) | −5.0 % |
| `random_walk` | MASE | 2.6680 | 2.5749 (recommended) | −3.5 % |

The wide WQL wins on intermittent / zero-inflated panels are the
sharpest story: **AutoETS's Gaussian-fallback quantile grid
miscalibrates badly on non-Gaussian discrete-support residuals**,
while `.auto_aid()`'s Poisson / NegBin / ZIP / ZINB leaves match
the DGP directly. If your metric is probabilistic, Laplace's edge
in count territory is bigger than the point-forecast MASE suggests.

### 2026-07-24 extension — 12 more archetypes → aggregate flips to Laplace

Extended bake-off from 29 → 41 archetypes to cover under-tested
axes: skewed continuous marginals (Gamma, Lognormal), overdispersed
counts (NegBin), multiplicative seasonality, AR(1) persistence,
non-linear trends (piecewise, exponential, S-curve), realistic
combos (retail-with-promotions, web-traffic with release spikes),
edge cases (near-constant, all-zeros-with-rare-spikes).

**Result flips**: on **Laplace-favoring category** (now 26 archetypes),
MS+3SH beats AutoETS by 21.3 % geomean (was AutoETS +8.7 %). **Overall
geomean** across all 41 archetypes: MS+3SH 0.8110 vs AutoETS 0.8489 —
**Laplace beats AutoETS by 4.5 %**. AutoETS-favoring category
unchanged (AutoETS +46.5 %); the new axes shift what's in the
Laplace-favoring category.

Category counts (41 archetypes total): 26 Laplace-favoring, 11
AutoETS-favoring, 4 Neutral.

The killer new archetype: `all_zeros_rare_spikes` (99 % zeros + 1 %
Poisson(10)). MS+3SH → near-perfect MASE (0.0000). Laplace's
`IntermittentLeaf` / `PoissonLeaf` / `ZeroInflatedPoissonLeaf`
predict the correct all-zero baseline; AutoETS smooths and misses.
This one archetype does move the geomean; per-archetype tables are
the honest read for shapes that don't match `all_zeros_rare_spikes`.

Other clear new-archetype Laplace wins:
- `ar1_persistent` (φ=0.9 AR(1)): Lap.auto 1.9443 vs AutoETS 2.1071
  (−8 %). `Ar1Leaf` targets this shape natively.
- `gamma_positive_skewed` / `negbin_overdispersed_counts` /
  `lognormal_multiplicative`: near-ties (MASE within 1-4 % of AutoETS).
  On WQL the Laplace mixture-quantile output would win by wider
  margins (not measured this run).

New-archetype AutoETS wins:
- `piecewise_linear_trend`: AutoETS 0.81 vs Lap 1.32 (AutoETS +62 %).
  Structural break interior to training; AutoETS's damped trend
  handles it, Laplace's regime-shift softmax needs more warmup.
- `exponential_growth`: AutoETS 2.94 vs Lap 3.65 (+24 %). AutoETS's
  multiplicative-trend variants target this DGP directly.
- `retail_with_promotions`, `weekly_plus_daily_plus_spike`: AutoETS
  wins by 40-47 %. The parametric baseline captures the smooth
  weekly + daily cycles; the promotion / release spikes hurt Laplace
  more than they hurt AutoETS.

**Router accuracy at 41 archetypes: 34/41 (83 %)**. Down from 26/29
(90 %) — added archetypes probe the count/heavy-tail boundary where
the shape checks disagree. Concrete misses: `negbin_overdispersed_counts`
(zero-fraction < 30 % threshold), `retail_with_promotions` (spikes
trigger heavy-tail before count check), `weekly_plus_daily_plus_spike`
(spikes trigger heavy-tail). Fixable but scope for a separate router
iteration — see `src/models/laplace/recommend.rs`.

**Sharpened practical rule** (2026-07-24):

- **Use Laplace** when your data is: extreme-intermittent (bursty,
  all-zeros-with-spikes), zero-inflated, has regime shifts / jumps,
  is non-parametric (RW, OU, AR(1)-persistent), has fading /
  evolving structure, OR you need distributional output. **Overall
  MS+3SH wins by 21 % on this shape space.**
- **Use AutoETS** when your data is: textbook trend + seasonal +
  Gaussian, multi-seasonal-structural, exponential / S-curve trend,
  or has smooth periodic structure with occasional additive shocks.
  **AutoETS wins by 47 % on this shape space.**
- **Both close** on: pure noise, short history, near-constant, gentle
  Gamma / Lognormal / NegBin marginals.

Every rule now has ~30 replicates × 30-800 obs of measurement backing
per archetype. Reproduce: `cargo run --release --features distributional --example synthetic_bakeoff`.

### 2026-07-24 — SmartForecaster gets cross-family routing

Extended `SmartForecaster` with shape-based cross-family routing
(previously it only routed within Laplace via AID). New rules:

1. `N < 60` → `AutoTheta` (streaming softmax hasn't converged).
2. Regular series + strong trend R² > 0.30 OR seasonal autocorrelation
   at lag = period > 0.40 → `AutoETS`.
3. Everything else → previous AID-based Laplace routing.

Second-pass bug fix (in the same session): initial routing only
triggered for `Regular + Normal` AID class and passed the default
seasonal_period=7 to AutoETS blindly. Both bugs caused SmartForecaster
to fall back to Laplace on multiplicative/exponential-growth positive
series (AID says Regular+Positive) and to confuse AutoETS on
non-seasonal series (fake period 7 wastes grid search).

Fixed router: routing triggers on ANY Regular subtype (Normal /
Positive / Count), and period is only passed to AutoETS when
`seasonal_autocorr_abs(values, period) > 0.40`.

Bake-off result at 41 archetypes × 30 replicates:

| Model | Overall geomean | v5 (broken) → v6 (fixed) |
|---|---:|---|
| MS+3SH manual | 0.8110 | (unchanged) |
| AutoETS | 0.8489 | (unchanged) |
| **SmartForecaster** | **0.8867** | 1.1360 → 0.8867 (−22 %) |
| recommended_for | 1.0268 | (unchanged) |
| AutoTheta | 1.0499 | (unchanged) |
| Lap.auto() | 1.1773 | (unchanged) |

Geomean by category:

| Category | AutoETS | MS+3SH | SmartForecaster |
|---|---:|---:|---:|
| Laplace-favoring | 0.8589 | **0.6756** | 0.9183 |
| **AutoETS-favoring** | **0.8409** | 1.2323 | **0.8409** ⭐ (matches) |
| Neutral | 0.8075 | 0.8420 | 0.8172 |

SmartForecaster now matches AutoETS exactly on all 11 AutoETS-favoring
archetypes (identical MASE — same recipe, same period pass-through).
Fixed archetypes vs the pre-router-fix baseline (`.auto()` fallback):

  archetype                      before   after    reference
  multiplicative_seasonality    6.4569   0.6675   AutoETS 0.6675
  exponential_growth           20.1282   2.9362   AutoETS 2.9362
  seasonal_linear_trend         1.1651   0.6484   AutoETS 0.6484
  linear_trend_only             0.9193   0.6563   AutoETS 0.6563
  heteroscedastic_multi_seasonal 1.4712  0.7072   AutoETS 0.7072
  everything_at_once            1.8256   1.0168   AutoETS 1.0168
  s_curve_logistic_growth       1.1608   0.9170   AutoETS 0.9170
  piecewise_linear_trend        3.2030   0.8137   AutoETS 0.8137
  bimodal_regime_switch         1.7769   1.0930   AutoETS 1.0849
  regime_shift_flat_to_trend    2.1134   0.7475   AutoETS 0.7475

Remaining gap vs MS+3SH on Laplace-favoring (0.9183 vs 0.6756) is
driven by AID's commit-to-a-single-family behaviour on extreme
intermittent panels. `all_zeros_rare_spikes`: MS+3SH 0.0000 (skaters
pool + softmax picks the right count leaf per series); SmartForecaster
0.3465 (AID picks one distribution family upfront, less flexible).
Fixing this would mean routing intermittent to `.skaters()` instead
of the AID-selected single-family recipe — a bigger design change,
deferred.

`SmartForecaster::new().fit(&series)` now covers most cross-family
tradeoffs in a single call: parametric structural → AutoETS,
count/intermittent → AID Laplace, short → AutoTheta, else →
Laplace.auto().

## Meta-lessons — patterns to watch for

### 1. Measure the feature's *active region* before implementing

The GPD tails / WQL mismatch (documented above) is the canonical example. 500 LOC + 17× fit-time cost for zero benefit, because the metric never tested the region the feature affects.

**Rule**: before implementing a feature X that improves quantity Y in region R, verify that Y is measured on R. Two-minute check.

### 2. When something regresses, first ask "is the pathology what I think it is?"

The precision-weighted ensemble was the fix for a *different pathology* than issue #198 actually had. #198 was cold-start, not weighting-scheme. The 200 LOC port worked mechanically but couldn't help.

**Rule**: before implementing a fix, write down the exact chain of cause → effect you're trying to break. If you can't articulate it, you don't know what you're fixing.

### 3. Verify winners on multiple benchmarks before defaulting them

The v0.15.4 recipe is a clean fev-27 win but hurts M5. If we'd defaulted `MultiScaleLaplace` behaviour into `.skaters()`, M5 users would silently regress. Because we shipped it as an opt-in wrapper, callers can pick per-task.

**Rule**: opt-in wrappers > default-on changes for anything that isn't universally an improvement. Reserve default-on for changes that measure neutral-to-positive on every benchmark you care about.

### 4. Different metrics tell different stories

`.skaters()` vs `.auto()` split roughly:
- MASE (point forecast): `.skaters()` often loses on continuous/short-N, wins on long-N + retail.
- WQL (probabilistic): `.skaters()` has occasional 100×–45000× outliers on continuous datasets (m1_yearly, tourism_yearly, cif_2016) because its sticky-lattice atoms concentrate quantile mass on revisited exact values — great on discrete counts, catastrophic on continuous smooth panels.

**Rule**: report at least one point-forecast metric and one probabilistic metric. A win on one and loss on the other is common.

### 5. Local `cargo build` doesn't run clippy with `-D warnings`

The v0.15.4 CI failed on clippy (unused import + dead code in `gpd_tails.rs`), blocking the crates.io publish. Local `cargo build` had shown only warnings. Fixed via v0.15.5 patch.

**Rule**: run `cargo clippy --all-features -- -D warnings` before pushing any commit that will trigger a release workflow. Or add it to a pre-commit hook.

### 6. Big-effort features can have small effects (and vice versa)

- `fast_slow` (50 LOC): **−24 % `.skaters()` WQL**
- `scoring_window` (50 LOC): **−5 % MASE**
- MultiScale + scH/scW (200 LOC, big design): **−11 % MASE** (biggest single win)
- Parade + GPD tails (500 LOC, biggest engineering): **0 %**

Rough correlation between engineering complexity and benefit was negative in this session. Small, targeted, mechanism-clear changes outperformed big architectural ports.

## Recommendations for future work

Ordered by expected impact-per-effort based on what we've measured:

1. **AID-driven default for `.skaters()` on all-positive series** — could remove the sticky-lattice-on-continuous WQL blowup we still see on m1_yearly, tourism_yearly, cif_2016 (100–45000× WQL outliers). Analogous to how `.auto()` auto-gates `non_negative` on all-positive training data (v0.15.1).
2. **Adaptive `w` for `.with_scoring_window`** — currently constant. A rule like `w = max(2H, N/10)` would pick smaller windows on shorter panels (where they help most) and larger on longer panels (where they should approach cumulative). Would let it become default-on.
3. **Multiscale on `.auto_aid()`** — currently only wraps `.skaters()`. Adapting for the AID-selected count-distribution pool would test whether the multi-scale + right-marginal combination helps M5.
4. **Adversarial per-dataset regression suite** — a fev-27 A/B harness that auto-flags "regresses more than K %" per dataset when we change defaults. Would catch M5-style regressions before release.
5. **Skip** — Rosenblatt conjugation (skaters #92 still open upstream), spec-grammar declarative pipeline (big refactor, no accuracy benefit). Both were considered and rejected in this session's decisions.

## Cross-references

- **`docs/LAPLACE_PARAMETER_GUIDE.md`** — prescriptive decision tree for
  humans + AI, keyed on observable data properties → concrete builder
  chains. Distills the recommendations from this post-mortem into
  actionable rules with evidence pointers.
- `CHANGELOG.md` — chronological per-release notes
- `docs/SOTA_POSITIONING.md` — current leaderboard position + M5 caveat preamble + release-by-release progression table
- `README.md` — user-facing rules of thumb + v0.15.4 recipe warning box
- `.claude/skills/laplace-distributional.md` — user-invocable API skill covering all the above
- `src/models/laplace/mod.rs` — module-level docstring for module-level design decisions
- Individual API docstrings on `with_seasonal`, `with_scoring_horizon`, `with_scoring_window`, `with_parade`, `MultiScaleLaplace::with_*`, `GpdTailsForecaster` — the where-and-why of each API
