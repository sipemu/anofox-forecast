# LaplaceForecaster parameter guide

**Audience**: any caller — human or AI — that needs to pick builder-chain
parameters for `LaplaceForecaster` (and related distributional forecasters).

**How to use this document**: read the decision tree top-down. Each rule
is keyed on an **observable data property** (zero fraction, seasonal
period, series length, tick-grid signature) and hands back a **specific
builder chain**. Every recommendation cites the measurement that
justifies it — chase the reference in `docs/LAPLACE_EXPERIMENTS.md` or
the linked test/example if you need to verify the number.

Rules that are marked **[SETTLED]** are grounded in benchmark runs.
Rules marked **[HEURISTIC]** are informed defaults where the measurement
is thin — treat them as starting points, not settled truth.

Every builder mentioned exists in `src/models/laplace/forecaster.rs`.
The `.claude/skills/laplace-distributional.md` skill covers the API
surface; this guide picks *which* API to call.

---

## TL;DR — one-liner router

If you don't want to read the decision tree below, use the extracted
router. Given a `TimeSeries` and horizon it inspects data-shape and
returns a fit-ready forecaster keyed to the 2026-07-21 decision table:

```rust
use anofox_forecast::models::laplace::{recommended_for, recipe_for};
use anofox_forecast::models::{DistributionalForecaster, Forecaster};

let mut f = recommended_for(&series, /* horizon */ 12, /* period */ Some(12));
f.fit(&series)?;
let mixtures = f.forecast_dist(12)?;
// Log which recipe was picked, e.g. for debugging a regression:
eprintln!("recipe = {}", recipe_for(&series, Some(12)).name());
```

Rules the router applies (in order):

| # | Trigger | Route | Evidence |
|---|---|---|---|
| 1 | `N < 60` | `.auto()` (Laplace fallback; classical Theta/ETS often better outside this crate) | Streaming softmax needs > 60 obs to converge |
| 2 | integer-values > 95 % **and** zero-fraction > 30 % | `.auto_aid().auto_with_seasonal_period(P)` | M5 measured `.skaters()` = 0.9962 MASE vs MS = 1.0833 (+8.7 %); AID picks Poisson/NegBin/Croston |
| 3 | excess kurtosis > 5 | `.skaters().with_terminal_crps()` | Cauchy input stability, `tests/laplace_robustness.rs::cauchy_input_produces_well_formed_forecast` |
| 4 | continuous + `period ≥ 2` + `N ≥ 60` | `MultiScaleLaplace + scH + sw=10 + η=0.20 + 3α-SH pool` (the fev-27 winner) | fev-27 rank ~6 (1.4149 geomean) |
| 5 | continuous fallback | `.skaters() + scH(P) + sw=10 + η=0.20` (no MS) | +0.24 % vs row 4 but simpler; skips MS when period unknown or `N < 50/scale` |

Period detection is deliberately out of scope — you must supply `period`.
For a cross-family router (also picks between `AutoTheta` / `AutoETS`),
use [`SmartForecaster`](../src/models/smart.rs) instead.

Read the sections below when you want to override a rule or understand
*why* a given recipe wins.

### When to reach for Laplace vs `AutoETS`

Measured 2026-07-24 on `examples/synthetic_bakeoff.rs` — **41
archetypes × 30 replicates** covering length, seasonality (fixed /
multi / fading / multiplicative), trend (none / linear / damped /
piecewise / exponential / S-curve), variance (constant / evolving /
GARCH clustering / regime), jumps, distribution (Gaussian / Cauchy /
Student-t / Gamma / Lognormal), count (Poisson / NegBin /
zero-inflated / bursty / all-zeros-with-spikes), tick-grid
discreteness, real-world combos (retail-with-promotions,
web-traffic). Segmented by DGP category:

| Category | AutoETS | Best Laplace | Δ | Count |
|---|---:|---:|---:|---:|
| **Laplace-favoring** (non-parametric / count / regime / heavy-tail) | 0.8589 | **0.6756** (MS+3SH) | **Laplace wins by 21.3 %** ⭐ | 26 |
| AutoETS-favoring (textbook trend + seasonal + Gaussian) | **0.8409** | 1.2323 | AutoETS wins by 46.5 % | 11 |
| Neutral (pure noise, short history, low-var, moderate var-shift) | 0.8075 | 0.8187 | AutoETS by 1.4 % | 4 |

**Overall geomean across all 41 archetypes: MS+3SH 0.8110 beats
AutoETS 0.8489 by 4.5 %.** This is the mirror image of the 18-archetype
snapshot — extending coverage to the shapes Laplace was actually
designed for (extreme intermittency, promotion spikes, non-parametric
persistence) tips the aggregate the other way.

Big caveat: on `all_zeros_rare_spikes` MS+3SH scores near-zero MASE
because Laplace's `IntermittentLeaf` / `PoissonLeaf` / `ZeroInflatedPoissonLeaf`
predict the correct all-zero baseline while AutoETS smooths and
misses. This one archetype does move the geomean; look at per-archetype
tables when your data doesn't match that shape.

**Per-archetype signals are what actually matter for the recipe choice**:

**Reach for Laplace when your data looks like these — clear wins:**

| Archetype | Metric | AutoETS | best Laplace | Δ |
|---|---|---:|---:|---:|
| `intermittent_bursty` | MASE | 0.9826 | 0.8471 (MS+3SH) | **−13.8 %** |
| `intermittent_bursty` | WQL | 1.3019 | 1.0584 (MS+3SH) | **−18.7 %** |
| `zero_inflated_seasonal` | WQL | 1.3682 | 1.1467 (MS+3SH) | **−16.2 %** |
| `level_shift_midway` | MASE | 0.7332 | 0.6905 (MS+3SH) | −5.8 % |
| `mean_reverting_ou` | MASE | 1.9092 | 1.8130 (Lap.auto) | −5.0 % |
| `random_walk` | MASE | 2.6680 | 2.5749 (recommended) | −3.5 % |
| `fading_seasonality` | WQL vs AutoTheta | 0.0415 | 0.0140 (recommended) | **−66 %** |

Pattern: **intermittent counts, zero-inflation, structural breaks,
non-parametric mean-reversion, fading / evolving structure**.
Distributional metrics (WQL) win by wider margins than point (MASE)
because AutoETS's Gaussian-fallback quantiles miscalibrate on the
non-Gaussian residuals these DGPs produce.

**Use `AutoETS` when your data looks like these — clear losses:**

| Archetype | AutoETS MASE | `recommended_for` MASE | Δ |
|---|---:|---:|---:|
| `multi_seasonal_hourly` | 0.7453 | 1.8898 | **+152 %** |
| `heteroscedastic_multi_seasonal` | 0.7072 | 1.4840 | +109 % |
| `linear_trend_only` | 0.6563 | 1.0345 | +57 % |
| `everything_at_once` | 1.0168 | 1.3758 | +35 % |
| `stationary_seasonal_short` | 0.7930 | 0.9780 | +23 % |

Pattern: **textbook trend + seasonal + Gaussian noise**. When the DGP
matches AutoETS's structural assumptions, the parametric baseline
wins by a wide margin — and Laplace's flexibility becomes overhead.

**Why our fev-27 rank ~6 doesn't contradict this.** Fev-27 panels
are almost all intermittent-count, real-world noisy, mixed-regime,
heavy-tailed — the exact shape space of the *Laplace-favoring*
category above. On clean-signal synthetic archetypes, AutoETS still
dominates. Our win on real data is real, but narrower than the
overall leaderboard framing suggests.

For a cross-family router that also picks `AutoETS` when the data
looks structural, use [`SmartForecaster`](../src/models/smart.rs).

---

## Step 1 — Pick the top-level selector

There are three streaming-distributional selectors plus one point-only
router. Pick by data type.

| Signal from your data | Selector | Why |
|---|---|---|
| Continuous / economic series (M-competition, FRED, tourism) | `.auto()` | 10-leaf per-heuristic pool; strongest on the fev-27 leaderboard-comparable subset (MASE 1.65) [SETTLED — `docs/SOTA_POSITIONING.md`] |
| Retail SKU / demand counts with **> 40 % zeros** | `SmartForecaster::new()` (or `.auto_aid()` if you need the `Vec<GaussianMixture>` interface) | Detects intermittency via `zero_fraction > 0.4`, routes to Croston-family + non-negative clamp. `.skaters()` and `.auto()` both regress here [SETTLED — `docs/LAPLACE_EXPERIMENTS.md` §"What didn't work"] |
| Non-intermittent count data with strong seasonality (electricity, hourly demand, wide count panels) | `.skaters()` | 30+ leaf pool with sticky lattice + terminal scale-mixture. Wins on m4_hourly (baseline MASE 2.16), `australian_electricity`. Slower fit |
| **Short-history** panels (N < 100) | Not `LaplaceForecaster` — use `AutoTheta` or `AutoETS` | Streaming leaves need warmup; softmax hasn't converged at N < 100 [SETTLED — `.claude/skills/laplace-distributional.md` §"When NOT to use"] |
| You only need a point forecast (no quantiles) | `AutoTheta` / `AutoETS` | Simpler, faster, no `distributional` feature gate |

**If unsure between `.auto()` and `.skaters()`**: prefer `.auto()`. It's
faster, matches `.skaters()` on most panels, and doesn't drag in the
sticky-lattice-on-continuous WQL blowup that `.skaters()` still has on
short yearly panels (m1_yearly, cif_2016, tourism_yearly) unless you
explicitly `.no_sticky()`.

---

## Step 2 — Commit to a seasonal period

If you know your period upstream (e.g. calendar-driven: 7 for daily,
12 for monthly, 24 for hourly, 168 for hour-of-week), USE it. This is
worth 5-10 % MASE on seasonal panels.

There are three APIs and they compose subtly. **Only use one of the
first two** (they overlap):

```rust
// Option A — recommended when you know the period AND you're using .auto() or .skaters():
LaplaceForecaster::new().skaters().auto_with_seasonal_period(P)
LaplaceForecaster::new().auto().auto_with_seasonal_period(P)
//                    ^^^^ enables seasonal_period = Some(P)
//                         adds seasonal-EMA(P) leaf
//                         does NOT default batch init (regression-safe for growing amplitude)

// Option B — when you want batch init on:
LaplaceForecaster::new().auto().with_seasonal(P)
//                              ^^^^^^^^^^^^^ same as A but ALSO enables batch init
//                              use when: stable amplitude, known period, N >= 2·P

// Option C — DO NOT use in combination with A or B:
LaplaceForecaster::new().auto().with_seasonal_multi(&[P])
//                              ^^^^^^^^^^^^^^^^^^^ adds a SECOND seasonal-EMA leaf
//                              use ONLY for genuinely multi-periodic data (e.g. &[7, 365])
//                              on daily-with-weekly-and-annual
```

### Rules

- **[SETTLED]** If period is known and amplitude is stable → `.with_seasonal(P)`. Batch init closes the softmax cold-start (issue #198, v0.15.2 fix).
- **[SETTLED]** If period is known but amplitude is **growing** or seasonal **phase is shifting** → `.with_seasonal(P).no_seasonal_batch_init()`. Batch init misleads the softmax with a stale last-cycle prior in these cases (issue #195).
- **[SETTLED]** Multiple periods (e.g. daily data with weekly AND annual) → `.with_seasonal_multi(&[7, 365])`. Never call this with a period already committed via `with_seasonal` or `auto_with_seasonal_period` — the double-add regresses tourism_quarterly by +2.8 % and m1_monthly by +1.6 % [SETTLED — `docs/LAPLACE_EXPERIMENTS.md` §"#112"].
- **[HEURISTIC]** Auto-detect period → let `.auto()` do it. Only override with `auto_with_seasonal_period(P)` when you have external knowledge (calendar structure the ACF can't see).

---

## Step 3 — Scoring knobs (opt-in, meaningful gains)

Two knobs replace the softmax scoring objective. Both are opt-in
because they compose non-trivially with pool composition and can hurt
some panels.

### `.with_scoring_window(w)`

Replace the cumulative `Σ logpdf` softmax with a sliding-window sum of
the last `w` observations.

- **[SETTLED]** Use `w = 7` or `w = 14` on `.auto()` for fev-27-style
  seasonal panels. **−5 % geomean MASE** [`docs/LAPLACE_EXPERIMENTS.md`
  §"What worked"].
- **[SETTLED]** Do NOT use with default larger values (`w = 56` or
  `w = 112`) — regression relative to the aggressive `w = 7`.
- **[HEURISTIC]** For M5 / retail counts: skip. Streaming intermittent
  leaves need long history to accumulate their state.

### `.with_scoring_horizon(H)`

Score softmax on H-step log-likelihood instead of 1-step. Redirects
softmax to horizon-H accuracy.

- **[SETTLED]** Use on seasonal panels where 1-step LL under-weights
  the seasonal leaf. Concrete win: **m4_hourly −5.71 % MASE** with
  `.skaters().auto_with_seasonal_period(24).with_scoring_horizon(24)`
  [`scratchpad/recipe_probe`, full-scale run 2026-07-17].
- **[SETTLED]** Also helps: tourism_monthly (−2.63 %), fred_md (−2.47 %).
- **[SETTLED]** Neutral to slightly negative on: m3_monthly, m4_monthly,
  m4_daily, m5. Do not add reflexively.
- **[SETTLED]** M5 unaffected (−0.4 %). Do not add.
- **[HEURISTIC]** Pair with `.with_scoring_window(14)` on `.auto()` for
  the v0.15.3 recommended recipe.

### Combined recipe (fev-27 winning `.auto()` chain)

```rust
LaplaceForecaster::new()
    .auto()
    .with_seasonal(P)                // if period known
    .with_scoring_horizon(P)         // match softmax to seasonal period
    .with_scoring_window(14)         // recent-window softmax
```

MASE gain: **−5.1 % on fev-27** vs plain `.auto()` [SETTLED — v0.15.3
release notes].

---

## Step 4 — MultiScale, tails, and calibration (opt-in, situational)

### `MultiScaleLaplace::skaters(H)` + tuned knobs + SH pool — best fev-27 config

Wraps `.skaters()` at multiple decimation strides ({1, period, k}) and
adds a per-phase Holt (SH) pool at scale 1. Best-known fev-27 geomean MASE.

- **[SETTLED]** Use when: continuous / economic panels + horizon-focused
  MASE optimization. **Geomean MASE 1.4149 on the 23-set leaderboard
  subset** (measured 2026-07-21). Prior v0.15.4-shipped recipe (no SH,
  sw=14, η=0.5) was 1.4572 — this recipe is a further **−2.9 %**.
- **[SETTLED]** DO NOT use for M5 / retail counts — **+8.7 % MASE
  regression** [`docs/LAPLACE_EXPERIMENTS.md` §"M5 regression"].
- **[SETTLED]** Requires N > 50 per activated scale; drops decimated
  scales otherwise.
- **[HEURISTIC]** Pass the period via `.with_period(P)` when known.

```rust
let mut m = MultiScaleLaplace::skaters(H)
    .with_scoring_horizon()
    .with_scoring_window(10)       // 2026-07-20 sweep: 10 beats 14
    .with_learning_rate(0.20)      // 2026-07-20 sweep: 0.20 beats 0.5 default
    .with_seasonal_holt(0.3, 0.1)  // multi-α SH pool: softmax picks
    .with_seasonal_holt(0.5, 0.2)  // per dataset
    .with_seasonal_holt(0.7, 0.3);
if period >= 2 { m = m.with_period(period); }
```

Sweep evidence (2026-07-21):
- SH single-α: best at `(0.5, 0.2)` giving −0.65 % vs no-SH. Wider alpha
  sweep {(0.4,0.15), (0.6,0.2), (0.6,0.3), (0.7,0.4)} shows a flat
  plateau — no single α beats (0.5, 0.2) alone.
- SH multi-α pool: 2-α `{(0.3,0.1), (0.5,0.2)}` gives −0.85 %; 3-α with
  `(0.7,0.3)` added gives **−0.97 %** vs no-SH (**−0.33 %** vs single).
  The softmax picks the right α per dataset (tourism_quarterly picks
  the aggressive one, m4_monthly picks the mild one).
- Killer per-dataset wins: tourism_quarterly 1.8421 → 1.7895 (**−2.9 %**),
  m3_quarterly 1.2714 → 1.2556, m4_quarterly 1.3146 → 1.3051.

### Simpler alternative: `.skaters() + scoring_horizon(P) + scoring_window(10) + SH pool`

If you don't want the MultiScale wrapper, the same knobs on plain
`.skaters()` are 99 % of the gain.

```rust
LaplaceForecaster::new()
    .skaters()
    .auto_with_seasonal_period(P)
    .with_scoring_horizon(P)
    .with_scoring_window(10)
    .learning_rate(0.20)
    .with_seasonal_holt(P, 0.3, 0.1)
    .with_seasonal_holt(P, 0.5, 0.2)
    .with_seasonal_holt(P, 0.7, 0.3)
```

- **[SETTLED]** Same tier as MultiScale; small MultiScale margin
  (≤ 0.6 % geomean, situational per-panel).
- **[SETTLED]** Same M5 rule applies — do not use for retail counts.
- **[HEURISTIC]** Recommended default for fev-27-shaped work when
  simplicity matters more than the last MultiScale margin.

### `.with_parade(k) + GpdTailsForecaster` — extreme quantiles

- **[SETTLED]** Use ONLY when your metric evaluates quantiles outside
  `[0.02, 0.98]` (anomaly detection, VaR, rare-event forecasting).
- **[SETTLED]** DO NOT add for fev-style WQL (`q ∈ [0.1, 0.9]`) —
  measured 0 % benefit at **17× fit-time cost** [`docs/LAPLACE_EXPERIMENTS.md`
  §"Parade + GPD tails"].
- **[HEURISTIC]** `.with_parade(k)` alone (no GPD) is standalone useful
  for per-horizon PIT diagnostics.

### `.with_calibration()` / `.with_per_horizon_calibration(k)`

- **[HEURISTIC]** Enable when you observe systematic PIT deviation from
  Uniform in a validation window. Cheap, no measured regression.

---

## Step 5 — Sticky lattice

`.skaters()` enables sticky by default with an auto-gate that turns it
off on continuous smooth panels. Rules for overriding:

| Data character | Setting | Evidence |
|---|---|---|
| Discrete counts (M5 SKU sales, dominick, exchange_rate) | `.skaters()` (leave default; auto-gate keeps it on) | [SETTLED] Big LL / MASE wins on these panels |
| Continuous smooth yearly / low-N (m1_yearly, tourism_yearly, cif_2016) | `.skaters().no_sticky()` | [SETTLED] Sticky degrades WQL by **100×-45000×** on these panels — the atoms concentrate quantile mass on spurious repeated values [`docs/LAPLACE_EXPERIMENTS.md` §"What didn't work" |
| Tick-grid financial data (futures on 0.25 grid) | Either — measured coverage-neutral in our impl | [SETTLED] Walk-forward 98 % band coverage identical (0.990) across default / forced-on / forced-off on synthetic tick-grid random walks [`tests/laplace_robustness.rs::sticky_vs_no_sticky_coverage_on_tick_grid_data`, skaters#107 verification] |
| Everything else | Leave the auto-gate to decide | [HEURISTIC] Default `.skaters()` picks correctly on most panels |

---

## Recipes for common panel types

Ready-to-use builder chains, ordered by common panel shape.

### Monthly economic (period 12) — the M3/tourism/fred class

Best-known fev-27 recipe (2026-07-21):

```rust
let mut m = MultiScaleLaplace::skaters(H)
    .with_period(12)
    .with_scoring_horizon()
    .with_scoring_window(10)
    .with_learning_rate(0.20)
    .with_seasonal_holt(0.3, 0.1)     // multi-α SH pool — softmax
    .with_seasonal_holt(0.5, 0.2)     // picks per dataset
    .with_seasonal_holt(0.7, 0.3);
```
Expect: rank ~6 on fev-27 (1.4149 geomean), passes Nixtla `auto_ets`.

Lighter alternative if you don't want the MultiScale wrapper:
```rust
LaplaceForecaster::new()
    .skaters()
    .auto_with_seasonal_period(12)
    .with_scoring_horizon(12)
    .with_scoring_window(10)
    .learning_rate(0.20)
    .with_seasonal_holt(12, 0.3, 0.1)
    .with_seasonal_holt(12, 0.5, 0.2)
    .with_seasonal_holt(12, 0.7, 0.3)
```

### Hourly with daily cycle (period 24) — M4-hourly / electricity

```rust
LaplaceForecaster::new()
    .skaters()
    .auto_with_seasonal_period(24)
    .with_scoring_horizon(24)      // measured -5.71 % on m4_hourly
```

If you *also* have weekly hour-of-week structure (period 168), the
current API doesn't help much — see `docs/LAPLACE_EXPERIMENTS.md` §"#91"
for the measured negative result. Skip `with_seasonal_multi(&[168])`.

### Retail count SKUs (period 7, heavy zeros)

```rust
SmartForecaster::new()             // point-only; auto-routes AID
    .with_seasonal_period(7)

// OR if you need the distributional interface:
LaplaceForecaster::new()
    .auto_aid()                    // AID-selected count leaf pool
    .auto_with_seasonal_period(7)
```

Do NOT add `with_scoring_horizon` or `MultiScaleLaplace` here (regresses).

### Financial returns / volatility-clustered

```rust
LaplaceForecaster::new()
    .skaters()
    .with_terminal_crps()          // CRPS objective — well-posed under fat tails
```

`.skaters()` already includes GARCH cascades and `terminal_crps`.
Cauchy input stability verified in
`tests/laplace_robustness.rs::cauchy_input_produces_well_formed_forecast`
(skaters#85). If you're doing tail risk, add `.with_parade(H) + GpdTailsForecaster`.

### Yearly / very short panels (N < 60, no seasonal period)

```rust
// Not a LaplaceForecaster job. Use:
AutoTheta::new()
// or:
AutoETS::new()
```

Streaming distributional selectors need > 60 obs to converge softmax.
Below that, classical statistical methods dominate.

---

## Anti-patterns (measured to hurt)

- `.skaters().auto_with_seasonal_period(P).with_seasonal_multi(&[P])`
  → double-adds seasonal-EMA(P) leaf. **Regresses tourism_quarterly +2.84 %,
  m1_monthly +1.64 %** [SETTLED].
- `MultiScaleLaplace::skaters(H)` on M5 retail
  → **+8.7 % MASE / +9.5 % WAPE regression** [SETTLED].
- `.with_scoring_window(112)` or larger
  → approaches cumulative baseline; loses the sleeper-hit gain of `w = 7-14`.
- `.with_parade(k) + GpdTailsForecaster` for fev-style WQL evaluation
  → 0 % benefit at 17× fit-time cost [SETTLED, metric-shape mismatch].
- `.skaters()` on m1_yearly / tourism_yearly / cif_2016 without `.no_sticky()`
  → **100×-45000× WQL blowup** [SETTLED].
- `.with_stacking()` at any `ridge_lambda`
  → best-case still **+3.9 % MASE regression** on fev-27 [SETTLED].
- Adding `.with_holt(0.3, 0.1, 0.9)` to `.skaters()`
  → **+1.5 % geomean MASE regression** on fev-27 [SETTLED, docs/ACCURACY_AUDIT.md].
- Any recipe that "works on synthetic but wasn't measured on real data"
  → treat as unproven until fev-27 + M5 numbers exist [meta-lesson].

---

## Debugging a bad forecast

If a fitted model produces a suspicious forecast, check in this order:

1. **Inspect leaf weights** — use `Inspectable::explanation(&f)` (or
   `debug_leaf_predictions()` for hidden internals). If one leaf owns
   > 95 % of softmax weight and it's the wrong one, the pathology is
   softmax scoring, not the leaf pool.
2. **Verify batch init on seasonal panels** — `.with_seasonal(P)` should
   have `seasonal_batch_init` on by default; confirm with
   `.debug_leaf_predictions()` — the seasonal-EMA leaf's std should be
   larger than the cold-start value if batch init fired.
3. **Check for the sticky-on-continuous WQL blowup** — if WQL is
   100×+ larger than MASE ratios suggest, the sticky atoms are firing
   on spurious repeated values. Add `.no_sticky()`.
4. **Check the parade PIT if you're using it** — parade snapshots the
   raw mixture (mean, std), not the sticky-atom-projected mixture.
   If your interpretation depends on the full mixture CDF, the parade
   diagnostic is only a lower bound of miscalibration.

---

## Reference: measured evidence sources

- `docs/LAPLACE_EXPERIMENTS.md` — post-mortem of ~15 experiments across
  v0.15.2 → v0.15.5 releases + verification tests for open skaters issues
  (#82, #84, #85, #86, #107, #112). The primary source for "why we
  ship / don't ship X".
- `docs/SOTA_POSITIONING.md` — release-by-release fev-27 rankings, M5
  caveat preamble.
- `docs/ACCURACY_AUDIT.md` — smaller-scale ablation studies.
- `tests/laplace_robustness.rs` — 16 tests covering adversarial input,
  determinism, and skaters-issue verifications. Reading a test docstring
  is the fastest way to understand a rule's evidence.
- `scratchpad/recipe_probe/` (transient) — the source of the full-scale
  seasonal-recipe #112 results reported here. Not in the repo — the
  numbers are in `LAPLACE_EXPERIMENTS.md` §"#112".

## When to update this guide

Any change to a `LaplaceForecaster` default or the `.auto()` /
`.skaters()` pool composition should trigger a corresponding rule
update here. New measurements that would shift a "SETTLED" rule to a
different builder chain should update both the recipe section and the
evidence source pointer.

If you're an AI reading this: **do not** invent builder chains not
documented here without running a benchmark first. The negative-result
list ("Anti-patterns") is longer than the positive-result list because
most compositions we tried didn't work.
