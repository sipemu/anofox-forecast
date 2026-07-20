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

### `MultiScaleLaplace::skaters(H)` — best fev-27 config (marginal over `+sw14`)

Wraps `.skaters()` at multiple decimation strides ({1, period, k}).
Best-known fev-27 geomean MASE.

- **[SETTLED]** Use when: continuous / economic panels + horizon-focused
  MASE optimization + you want the last 0.6 % of accuracy.
  **Geomean MASE 1.4572 on the 23-set leaderboard subset** (reproduced
  2026-07-20; documented value 1.4602 from v0.15.4 release).
- **[SETTLED]** DO NOT use for M5 / retail counts — **+8.7 % MASE
  regression** [`docs/LAPLACE_EXPERIMENTS.md` §"M5 regression"].
- **[SETTLED]** Requires N > 50 per activated scale; drops decimated
  scales otherwise.
- **[HEURISTIC]** Pass the period via `.with_period(P)` when known.

```rust
let mut m = MultiScaleLaplace::skaters(H)
    .with_scoring_horizon()
    .with_scoring_window(14);
if period >= 2 { m = m.with_period(period); }
```

**⚠ Attribution note (measured 2026-07-20)**: on the 23-set leaderboard
subset, MultiScale's marginal contribution over the plain-`.skaters()`
recipe with the same scoring knobs is only **0.57 %** geomean MASE
(1.4572 vs 1.4655). The bulk of the historical −11.3 % v0.15.4 headline
gain is actually from `.with_scoring_window(14)` (v0.15.3), not from
the MultiScale wrapping. See the lighter alternative below.

### Simpler alternative: `.skaters() + scoring_horizon(P) + scoring_window(14)` — 99 % of MultiScale's gain

Same accuracy tier, ~15 % faster fit, no MultiScale wrapping, no per-scale
sub-forecaster book-keeping.

```rust
LaplaceForecaster::new()
    .skaters()
    .auto_with_seasonal_period(P)
    .with_scoring_horizon(P)
    .with_scoring_window(14)
```

- **[SETTLED]** Geomean MASE **1.4655** on the 23-set leaderboard
  subset — 0.57 % behind MultiScale, but wins **10/23** datasets vs
  MultiScale's 8/23. Directly competitive.
- **[SETTLED]** Total fit time 9.6 s vs MultiScale's 11.5 s on the same
  panel (~15 % lighter). No MultiScale complexity.
- **[SETTLED]** Same M5 rule applies — do not use for retail counts.
- **[HEURISTIC]** Recommended default for fev-27-shaped work when
  simplicity matters more than the last 0.6 %.

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

```rust
LaplaceForecaster::new()
    .auto()
    .with_seasonal(12)
    .with_scoring_horizon(12)
    .with_scoring_window(14)
```
Expect: MASE competitive with AutoETS. Add `MultiScaleLaplace` wrapping
for another 5-10 % on longer panels.

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
