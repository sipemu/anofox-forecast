# Laplace experiments — post-mortem (v0.15.2 → v0.15.5)

*Written 2026-07-14 after the skaters-parity work. This is a consolidated engineering post-mortem covering ~10 experiments across four releases: what we tried, what worked, what didn't, and the meta-lessons. Meant to save future contributors from re-running dead ends.*

## Context

The starting point was v0.13.0 `LaplaceForecaster::auto()` at rank 11 on the fev-27 leaderboard-comparable subset (MASE 1.723). By v0.15.5 we're at rank ~7-8 (MASE 1.4602 with the winning opt-in recipe) — competitive with Nixtla `auto_ets`. This document records how we got there.

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

### `MultiScaleLaplace + scH + scW=14` — the star (v0.15.4)

Combining v0.15.3's scoring knobs with a multi-scale wrapper (each scale runs `.skaters()` on decimated data, blended per-horizon via softmax over per-scale training log-lik) gave **−11.3 % MASE** on the leaderboard-comparable subset. Biggest single-change improvement in project history.

**Key tuning parameters** — all discovered by measurement:
- `min_samples = 50` (dropped from 100). At 100 no fev-27 panel activates any decimated scale (degenerates to scale-1). At 30 m4_hourly's stride 24 activates with only 29 obs and the 30-leaf sub-pool can't fit → catastrophic regression.
- Drop the `⌈√k⌉` stride when a period is set. A coprime stride aliases the seasonal signal; measured on fev-27 as −55 % m4_hourly / −50 % tourism_monthly regression at v1 of the port.
- Scale-1 sub-forecaster receives the period hint via `.auto_with_seasonal_period(p)` — matches the fev-27 harness call pattern so multiscale isn't handicapped by missing the v0.15.1 seasonal-period fix.
- Sub-forecasters wrap `.skaters()`, not `.auto()` — the wider pool wins on m4_hourly and long-N panels.

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

- `CHANGELOG.md` — chronological per-release notes
- `docs/SOTA_POSITIONING.md` — current leaderboard position + M5 caveat preamble + release-by-release progression table
- `README.md` — user-facing rules of thumb + v0.15.4 recipe warning box
- `.claude/skills/laplace-distributional.md` — user-invocable API skill covering all the above
- `src/models/laplace/mod.rs` — module-level docstring for module-level design decisions
- Individual API docstrings on `with_seasonal`, `with_scoring_horizon`, `with_scoring_window`, `with_parade`, `MultiScaleLaplace::with_*`, `GpdTailsForecaster` — the where-and-why of each API
