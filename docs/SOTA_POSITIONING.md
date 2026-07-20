# anofox-forecast SOTA positioning

*Updated 2026-07-14 for v0.15.4 (skaters-parity port: fast_slow + multiscale + parade + GPD tails). Datasets and metrics from autogluon/fev's Chronos-benchmark classical panel and our M5 200-series head-to-head. Reproducible via `cargo run --release --features distributional --example fev_benchmark` and `examples/m5_skaters_vs_autoets.rs`.*

## ⚠ Read this first — different benchmarks want different recipes

**The v0.15.4 winning recipe (`MultiScaleLaplace + scH + scW=14`) is a fev-27 win but REGRESSES on M5 retail.** Measured on M5 200-series:

| Model | mean MASE | panel WAPE | fit time |
|---|---:|---:|---:|
| `Laplace + skaters()` | **0.9962** | **0.6868** | 0.8 s |
| MultiScale + scH + scW=14 (v0.15.4 fev-27 winner) | 1.0833 (**+8.7 %**) | 0.7521 (**+9.5 %**) | 1.1 s |

Selector by data type:

| Data type | Recipe |
|---|---|
| **Continuous / M-competition / economic (fev-27-like)** | `MultiScaleLaplace::skaters(H).with_scoring_horizon().with_scoring_window(14)` |
| **Retail SKU / demand / count data** | `LaplaceForecaster::new().auto_aid()` or `SmartForecaster::new()` (with `postprocess` feature) |
| **Short-history (N < 100)** | `AutoTheta` / `AutoETS` from `crate::models::exponential` / `crate::models::theta` |

Root cause of the M5 regression: M5 is intermittent count data where AID-selected Poisson / NegBin / Croston-family leaves are the right marginals — Gaussian mixtures aren't. MultiScaleLaplace wraps `.skaters()` (Gaussian pool), which is off the right selector for this data. **The two benchmark stories are structurally different tasks.**

## v0.15.4 headline — MultiScaleLaplace closes the gap to Nixtla classical

The best config on the fev-27 leaderboard-comparable 23-dataset subset is now:

```rust
MultiScaleLaplace::skaters(H)
    .with_scoring_horizon()          // v0.15.3 — match softmax scoring depth to horizon
    .with_scoring_window(14)         // v0.15.3 — moving-window LL instead of cumulative
```

| Rank | Model | MASE | Tier |
|---:|---|---:|---|
| 🥇 1 | Tirex | 1.351 | Foundation (GPU) |
| 🥈 2 | TimesFM-2.0 | 1.354 | Foundation (GPU) |
| 🥉 3 | Nixtla `auto_theta` | 1.362 | Classical (CPU) |
| 4 | **this crate: `AutoTheta`** | **1.381** | Classical (CPU) |
| 5 | Chronos-Bolt-Base | 1.393 | Foundation (CPU-optimized) |
| 6 | Moirai-Base | 1.423 | Foundation (GPU) |
| 7 | Nixtla `auto_ets` | 1.440 | Classical (CPU) |
| **~7-8** | **this crate: `MultiScaleLaplace + scH + scW=14` (v0.15.4)** | **1.4572** | **Classical (CPU), streaming shell** |
| **~7-8** | **this crate: `.skaters() + scH(P) + scW=14` (v0.15.3)** | **1.4655** | **Same tier, simpler recipe (2026-07-20 A/B)** |
| 8 | this crate: `AutoETS` | 1.525 | Classical (CPU) |
| 9 | Seasonal Naive | 1.665 | Baseline |
| ~10 | this crate: `LaplaceForecaster::auto()` (v0.15.4) | 1.6457 | Streaming |
| ~10 | this crate: `LaplaceForecaster::skaters()` (v0.15.4) | 1.6554 | Streaming |

**We moved up ~3 ranks from v0.13.0 (was rank 11 at 1.723). Now competitive with Nixtla `auto_ets` (1.440) and ahead of our own `AutoETS` (1.525).**

**⚠ Attribution note (measured 2026-07-20)**: MultiScale's marginal
contribution over the plain-`.skaters()` recipe with the same scoring
knobs is only **0.57 %** geomean MASE (1.4572 vs 1.4655). Wins the
geomean by a hair, loses the win-rate (8/23 vs 10/23). The bulk of the
v0.15.4 −15.3 % headline is really v0.15.3's `scoring_window(14)`; the
MultiScale wrapping added on top is a small, situational refinement,
not the "biggest single-change improvement" the original narrative
suggested. Users who want the last 0.6 % of accuracy or already have
MultiScale in their stack — use it. Users who want the simpler API and
15 % lower fit time — use the plain-`.skaters()` recipe. See
[`docs/LAPLACE_PARAMETER_GUIDE.md`](LAPLACE_PARAMETER_GUIDE.md) §"Step 4"
for the full breakdown.

Progression by release on the same 23-set geomean MASE:

| Release | `.auto()` | winning opt-in config | Δ vs v0.13.0 |
|---|---:|---:|---:|
| v0.13.0 (baseline) | 1.723 | — | — |
| v0.15.1 (#195 fixes) | 1.6457 | — | −4.5 % |
| v0.15.3 (scH + scW) | 1.6457 | `.skaters()` + scH + scW=14: **1.4655**† | **−15.0 %** |
| v0.15.4 (multiscale) | 1.6457 | `MultiScaleLaplace + scH + scW=14`: **1.4572**† | **−15.5 %** |

† *Both remeasured 2026-07-20 on the same 23-set subset with SAMPLE_PER=500.
Original v0.15.4 release-time value was 1.4602; the 2026-07-20 rerun
gets 1.4572 (matches within 0.2 %). Original v0.15.3 value was 1.4994;
the difference now to 1.4655 is that the v0.15.3 measurement did not
include the `.auto_with_seasonal_period(P)` baseline that we now use
universally.*

## Historic benchmarks (v0.13.0 → v0.15.0 era)

## Headline — TWO benchmarks, TWO positions

The extended `LaplaceForecaster` shipped in #180 (with `.skaters()`: full fixed candidate pool, sticky lattice, terminal scale-mixture, XGBoost-shrunk softmax) is a **different** engine from the pre-#180 `.auto()`. It has to be measured separately.

### Benchmark A — M5 200-series retail (H=28, our benchmark)

| model | mean MASE (↓) | panel WAPE (↓) | fit time |
|---|---:|---:|---:|
| **`LaplaceForecaster::new().skaters()`** | **0.9775** ⭐ | **0.6628** ⭐ | **0.6 s** |
| AutoETS | 1.0652 | 0.6950 | 5.6 s |
| `LaplaceForecaster::new().auto()` | 1.1878 | 0.7958 | 0.2 s |

**`.skaters()` beats AutoETS on both accuracy metrics AND is 9.3× faster.** On retail count data, the extended engine now dominates the classical baseline that had been top-ranked here.

Reproduce: `SAMPLE_SIZE=200 cargo run --release --features distributional --example m5_skaters_vs_autoets`.

### Benchmark B — fev Chronos-benchmark 27-panel (mixed classical)

On the 27-dataset fev classical panel (500 series/dataset, MASE with fev-canonical seasonal-naive scaling, geomean across 19 datasets shared with the fev leaderboard):

| rank | model | MASE | tier |
|---|---|---|---|
| 🥇 1 | Tirex | 1.351 | Foundation (GPU) |
| 🥈 2 | TimesFM-2.0 | 1.354 | Foundation (GPU) |
| 🥉 3 | fev `auto_theta` (Nixtla) | 1.362 | Classical (CPU) |
| 4 | **`AutoTheta` (this crate)** | **1.381** | **Classical (CPU) — 1.4 % behind Nixtla** ✅ |
| 5 | Chronos-Bolt-Base | 1.393 | Foundation (CPU-optimized) |
| 6 | Moirai-Base | 1.423 | Foundation (GPU) |
| 7 | fev `auto_ets` (Nixtla) | 1.440 | Classical (CPU) |
| 8 | **`AutoETS` (this crate)** | **1.525** | **Classical (CPU) — 5.9 % behind Nixtla** |
| 9 | Seasonal Naive | 1.665 | Baseline |
| **10** | **`LaplaceForecaster::skaters()`** (post-fix) | **~1.6 est.** | **Streaming — see nuanced breakdown below** |
| **11** | **`LaplaceForecaster::auto()`** | **1.723** | **Classical (CPU) — mixed-panel warmup penalty** |

**On the aggregate the picture is mixed** — `.skaters()` doesn't uniformly dominate `.auto()` on this panel. It **wins big on retail / demand / discrete-count datasets** and **wins short-history yearly datasets** where the wider pool + shrunk softmax help warmup, but **loses to `.auto()` and classical on smooth M-competition monthly/quarterly** where sticky-lattice atoms don't help.

### Per-dataset MASE — where `.skaters()` beats `.auto()` and classical

| dataset | AutoTheta | Laplace+auto | **Laplace+skaters** | verdict |
|---|---:|---:|---:|---|
| **m5** | 1.150 | 1.253 | **1.045** | 🥇 skaters beats classical (−9 % vs AutoTheta) |
| **exchange_rate** | 1.841 | 1.893 | **1.682** | 🥇 skaters beats classical (−9 %) |
| m1_yearly | 3.824 | 5.260 | **3.864** | skaters cuts 27 % vs `.auto()`; matches AutoTheta |
| m3_yearly | 2.876 | 3.777 | **3.107** | skaters cuts 18 % vs `.auto()` |
| m4_yearly | 3.965 | 5.254 | **4.136** | skaters cuts 21 % vs `.auto()` |
| tourism_yearly | 2.778 | 3.365 | **3.362** | skaters matches `.auto()`; both lose to classical |

### Per-dataset MASE — where `.skaters()` still loses to classical

| dataset | AutoTheta | **Laplace+skaters** (post-fix) | gap |
|---|---:|---:|---|
| m3_monthly | 0.731 | 0.772 | +6 % |
| m3_quarterly | 1.138 | 1.494 | +31 % |
| m1_monthly | 1.082 | 1.373 | +27 % |
| m4_monthly | 1.210 | 1.337 | +10 % |
| m4_quarterly | 1.121 | 1.308 | +17 % |
| tourism_monthly | 1.677 | 2.344 | +40 % |
| tourism_quarterly | 1.668 | 3.304 | +98 % |
| m4_hourly | 2.517 | **2.329** | **−7 %** ⭐ (was +119 %) |
| cif_2016 | 1.019 | 1.347 | +32 % |
| hospital | 0.764 | 0.784 | +3 % |
| dominick | 0.861 | 0.966 | +12 % (but −23 % vs `.auto()`) |

### Impact of the two fev-27 fixes (post-#180 follow-up)

|                        | before fixes | after fixes | Δ MASE |
|------------------------|-------------:|------------:|-------:|
| geomean MASE           |       6.6237 |  **6.0847** |  −8 % |
| **m4_hourly** MASE     |        5.509 |   **2.329** | **−58 %** ⭐ |
| **aus_electric** MASE  |       10.060 |   **2.560** | **−75 %** ⭐ |
| m4_daily MASE          |        1.191 |       1.109 |  −7 % |
| m4_weekly MASE         |        2.915 |       2.723 |  −7 % |
| tourism_monthly MASE   |        2.559 |       2.344 |  −8 % |
| m1_yearly MASE         |        3.864 |       4.554 |  +18 % (Fix B slight over-fit) |
| m4_yearly MASE         |        4.136 |       4.359 |  +5 %  |

**Fix B** (seasonal-diff drift compounding) delivered the m4_hourly + australian_electricity turnaround. **Fix A** (sticky horizon decay at 0.05 / step) is too gentle to collapse the WQL blowup on short-horizon continuous panels — atoms still dominate at ≥ 70 % weight through h=12.

### The sticky-lattice WQL cost on continuous data

Sticky-lattice atoms concentrate quantile mass on revisited exact values. On discrete-count panels this is a big win (M5, dominick, exchange_rate all show tight WQL). **On continuous smooth panels the atoms are misplaced and WQL blows up:**

- `m1_yearly` WQL: `AutoTheta` 0.157 vs `Laplace+skaters` **281.9** (1800× worse)
- `tourism_yearly` WQL: `AutoTheta` 0.235 vs `Laplace+skaters` **23.8** (100× worse)
- `cif_2016` WQL: `AutoTheta` 0.143 vs `Laplace+skaters` **55405** (pathological)

**Implication:** for callers on continuous / smooth panels, `.skaters()` needs a `.no_sticky()` variant. Sticky is not a universal win — it's specifically a retail-count technique. Follow-up to expose the toggle cleanly.

### Fit time on the fev-27 panel

| model | total fit (s) | avg / series-fit |
|---|---:|---:|
| AutoETS | 1093.6 | ~120 ms |
| AutoTheta | 63.6 | ~7 ms |
| Laplace+skaters | 11.7 | ~1.3 ms |
| Laplace+auto | 7.1 | ~0.8 ms |
| SmartForecaster | 2.0 | ~0.2 ms |

**`.skaters()` is ~5× faster than AutoTheta and ~90× faster than AutoETS on the fev classical panel** — even where it loses on accuracy, the compute story is decisive.

### Benchmark C — vs `skaters.laplace` (Python reference)

100 M5 series × 90 rolling one-step preds on first-differenced counts, both sides `sticky=True` and `objective="likelihood"`:

| metric | Rust `.skaters()` | skaters.laplace | Δ |
|---|---:|---:|---:|
| **Log-likelihood (nats/pred, ↑)** | **+0.914** | −0.488 | **+1.402** (we win) |
| **CRPS (↓)** | **0.847** | 0.869 | **−0.021** (we win) |
| **LL win rate (per-series)** | **60 / 100** | 40 / 100 | |
| **CRPS win rate (per-series)** | **56 / 100** | 44 / 100 | |
| **Wall time** | 11.9 s | 70.4 s | **5.9× faster** |

`exp(1.40) ≈ 4.06×` more density mass at the actual observed value than skaters. Sticky-lattice atoms sit under a tighter terminal-mixture density here, so on M5's zero-inflated first-differences the atoms are more concentrated on the modal outcome (0). Full picture in `bench/skaters-laplace-bakeoff`.

## Two clear findings

### 1. Our classical `AutoETS` / `AutoTheta` are competitive with Nixtla reference

- **`AutoTheta`**: 1.381 vs Nixtla 1.362 — within 1.4 %, well inside sampling variance.
- **`AutoETS`**: 1.525 vs Nixtla 1.440 — 5.9 % gap. Small.

For general-purpose forecasting on economic / retail / mixed panels, our internal classical baselines are **Nixtla-quality**.

### 2. On retail data, `.skaters()` now beats AutoETS

The pre-#180 comparison had AutoETS at the top of the retail slot (M5 full-30k, median MAE 0.728). Post-#180, `.skaters()` on M5 200-series wins on **both MASE and WAPE at 9.3× the speed**. The fixed skaters-shaped candidate pool + sticky-lattice projection + terminal scale-mixture together add up to a real point-forecast improvement, not just a distributional one.

### 3. `.auto()` still sits at seasonal-naive level on the fev mixed panel

The old, heuristic `.auto()` selector loses to `AutoTheta` by 25 % geomean MASE on the fev 27-panel. Root cause: **warmup penalty on short-history panels**. The new `.skaters()` engine avoids this by having a much wider fixed pool (~30 candidates) with cheap XGBoost-shrunk softmax updates that stay adaptive during warmup — but that trade wasn't tested end-to-end on fev yet.

## Where `.auto()` loses on fev — by training length

Per-dataset MASE vs. `AutoTheta` (pre-#180 `.auto()`):

| panel | N | H | Laplace.auto() | AutoTheta | Δ |
|---|---|---|---|---|---|
| m3_monthly | ~450 | 18 | 0.816 | 0.731 | +12 % |
| hospital | ~72 | 12 | 0.780 | 0.764 | +2 % |
| nn5_weekly | ~726 | 8 | 0.963 | 0.969 | −1 % (win) |
| m4_daily | ~2820 | 14 | 1.169 | 1.113 | +5 % |
| fred_md | ~700 | 12 | 0.615 | 0.566 | +9 % |
| m3_yearly | ~30 | 6 | 3.493 | 2.876 | +21 % |
| m1_yearly | ~30 | 6 | 4.931 | 3.824 | +29 % |
| m4_yearly | ~30 | 6 | 4.674 | 3.965 | +18 % |
| tourism_yearly | ~30 | 4 | 3.257 | 2.778 | +17 % |
| m3_quarterly | ~65 | 8 | 1.418 | 1.138 | +25 % |
| tourism_quarterly | ~130 | 8 | 2.554 | 1.668 | +53 % |
| m4_hourly | ~700 | 48 | 4.559 | 2.517 | +81 % |
| dominick | ~100 | 8 | 1.447 | 0.861 | +68 % |

Sorted by training length:

- **N > 300** — `.auto()` is competitive (within ±5–15 %).
- **N = 100–300** — starts losing (+5 to +25 %).
- **N = 50–100** — larger losses (+25 to +80 %). Cold start dominates.
- **N < 50** — consistent +15 to +30 % loss.

The wider `.skaters()` pool + XGBoost-shrunk updates should compress this significantly, but full re-measurement is pending.

## Design principle — this is architectural, not a bug

`LaplaceForecaster` is a **streaming per-observation** design by construction:

- Each leaf maintains state that updates on `observe(y)`. State converges after ~30-50 observations depending on the leaf's smoothing rate.
- The leaf softmax maintains cumulative log-likelihood per leaf and reweights via softmax.
- With `.auto()` (small hand-picked pool), the softmax is easier to peak wrong on early observations. With `.skaters()` (30+ candidates, η = 0.5 shrunk updates, log-clamp), the ensemble stays adaptive longer.

Classical closed-form fitters (`AutoETS`, `AutoTheta`) solve their parameters over the full training window in one shot. That closes the warmup penalty on VERY short panels (N < 50) but doesn't help on retail-scale data.

**This is a trade-off:**

| property | Classical | `LaplaceForecaster::skaters()` |
|---|---|---|
| Fit efficiency on long series | O(N) per iteration, needs re-solve if data grows | O(1) per new observation |
| Cold-start on short series (N < 50) | direct optimum from N observations | ~30-obs warmup, mitigated by wider softmax pool |
| Distributional output | none (point + parametric intervals only) | full Gaussian mixture, per-horizon, with sticky-lattice atoms on discrete-repeat data |
| Streaming updates | requires refit | native |
| Retail count data (M5) | 5.6 s / 200 series, MASE 1.065 | **0.6 s / 200 series, MASE 0.978** |
| CPU speed | ms-to-seconds per fit | sub-millisecond per fit |

**Use `LaplaceForecaster::skaters()` when:** your series is retail / demand / integer-count with adequate history (N ≥ 100 typical), OR you need distributional output (mixture density, quantiles, per-h calibration), OR you're on continuous data where AutoETS was previously the classical top.

**Use `AutoTheta` / `AutoETS` when:** your series is very short (N < 50), you just need point forecasts, or you're on the M-Competition-style classical panels where they are competitive with the best foundation models — the fev re-measurement is still pending.

**Use `LaplaceForecaster::auto()` when:** you need the smallest, cheapest ensemble (~5-10 candidates), and you know the data class fits its heuristic gates (e.g. clear seasonality, adequate history).

## Comparison to foundation models

Foundation models (Chronos-Bolt, TimesFM-2.0, Tirex, Moirai) beat classical by ~2–3 % MASE on the fev panel. That gap corresponds to their **cross-series pretraining data advantage** — a foundation model trained on ~100k-1M series has learned an implicit prior over data-generating processes that is finer-grained than what any classical parametric family (ETS, ARMA) can express.

Mathematically, this is amortized Bayesian inference under a learned prior; see `TabPFN` (Hollmann et al. 2022, 2025) for the cleanest formulation. This is not accessible to a per-series streaming classical model without pretraining infrastructure.

Whether `.skaters()` closes any part of that gap on the fev panel is still an open question — the M5 result suggests the sticky-lattice + tighter density might help on retail-count series specifically, but foundation models are trained across many more data classes.

## Where `LaplaceForecaster` shines

Empirical wins on our M5 benchmarks:

### M5 full 30k (pre-#180 numbers — `.auto_aid()` / `SmartForecaster` for AID-driven demand)

| model | MAE (median, 30k series) | fit time | vs. AutoETS |
|---|---|---|---|
| AutoETS | 0.728 | 916 s | — |
| `Laplace + auto_aid` | 0.734 | 22 s | +0.8 % MASE, ~42× faster |
| `SmartForecaster` | 0.735 | 11 s | +1.0 % MASE, ~82× faster |

### M5 200-series (post-#180 numbers — `.skaters()` for the full extended engine)

| model | mean MASE | panel WAPE | fit time |
|---|---:|---:|---:|
| **`Laplace + skaters()`** | **0.978** | **0.663** | **0.6 s** |
| AutoETS | 1.065 | 0.695 | 5.6 s |

For **retail SKU / demand forecasting**, both routes now beat classical:
- `.auto_aid()` / `SmartForecaster` — matches classical on point accuracy at ~40-80× speed, with domain-aware family selection.
- `.skaters()` — **beats classical on both point accuracy and speed**, with full distributional output.

## Reproduce

```bash
# fev 27-panel (~90 min for the numbers in Benchmark B)
cargo run --release --features distributional --example fev_benchmark

# M5 200-series skaters vs AutoETS (Benchmark A, ~10 s)
SAMPLE_SIZE=200 cargo run --release --features distributional --example m5_skaters_vs_autoets

# skaters.laplace bakeoff (Benchmark C, ~1.5 min)
BUILDER=skaters SAMPLE_SIZE=100 PRED_STRIDE=10 MAX_T=1200 BURN_IN=300 \
  cargo run --release --features distributional --example m5_bakeoff_export
python3 scripts/m5_laplace_bakeoff.py --burn-in 300
```

Datasets fetched from the Monash Time Series Forecasting Archive (Zenodo), autogluon/chronos_datasets (HuggingFace), and the M5 competition CSV.

## What was tried and DIDN'T work (documented so we don't re-try)

### STL-decomposition leaf (auto-enabled)

Ported an `StlDecompLeaf(period)` that runs STL on a rolling buffer and extrapolates linear trend + cyclic seasonal at forecast time. Auto-enabled it when `seasonality_strength > 0.30` and training length ≥ 3 × period.

**Result on fev-27 (SAMPLE_PER=500)**: NET REGRESSION.

| dataset | before STL auto | with STL auto | Δ |
|---|---:|---:|---:|
| **tourism_monthly** | 2.344 | **3.077** | **+31 %** ❌ |
| dominick | 0.966 | 1.024 | +6 % |
| m5 | 1.045 | 1.065 | +2 % |
| m4_monthly | 1.337 | 1.376 | +3 % |
| geomean MASE | **6.085** | **6.186** | **+1.7 %** ❌ |
| geomean WQL | **0.6695** | **0.7257** | **+8.4 %** ❌ |

The STL leaf's linear-trend extrapolation compounds badly at long horizons on short-history seasonal panels (150-300 obs, H=24). Same failure mode as the seasonal-diff drift compounding I fixed in Fix B — but harder to gate cleanly.

**Decision**: `StlDecompLeaf` remains available as opt-in via `LaplaceForecaster::new().with_stl(period)` for callers who verify it helps on their specific data. Not auto-enabled by any builder.

## Deferred / future work

- **Multi-scale improvements** — `MultiScaleLaplace::with_period(p)` adds period-aligned strides. Only helps on long time series where decimation still leaves ≥100 samples. Not auto-enabled in `.skaters()`. Effort to make it universally safe: needs a proper skaters-style likelihood-blend across eligible scales + warmup handling for decimated forecasters on short data.
- **Full autogluon/fev PyO3 bridge** — head-to-head submission to the [autogluon/fev leaderboard](https://huggingface.co/spaces/autogluon/fev-leaderboard). Effort: 3–5 days.
- **Foundation model in pure Rust** — TabPFN-style prior-fitted network trained on synthetic time-series priors. Effort: 2-3 months, adds `candle` / `burn` dependency.
- **Improve short-series behavior** — not through classical fallback (which we explicitly avoid — the shell should stay purely streaming), but through leaf-specific batch initialization tricks.
- **Full autogluon/fev PyO3 bridge** — head-to-head submission to the [autogluon/fev leaderboard](https://huggingface.co/spaces/autogluon/fev-leaderboard). Effort: 3–5 days.
- **Foundation model in pure Rust** — TabPFN-style prior-fitted network trained on synthetic time-series priors. Effort: 2-3 months, adds `candle` / `burn` dependency.
- **Improve short-series behavior** — not through classical fallback (which we explicitly avoid — the shell should stay purely streaming), but through leaf-specific batch initialization tricks.

## Defensible claims

- *"`anofox-forecast`'s `AutoTheta` matches Nixtla reference quality within 1.4 % MASE on the fev Chronos-benchmark classical panel."*
- *"On M5 200-series (H=28, weekly-period MASE), `LaplaceForecaster::skaters()` beats AutoETS by 8.2 % MASE / 4.6 % WAPE at 9.3× the fit-time speed."*
- *"On the fev classical panel, `LaplaceForecaster::skaters()` wins on the retail / demand slice (m5 −9 % vs AutoTheta, exchange_rate −9 %) and on short-history yearly panels vs `.auto()` (m1/m3/m4 yearly, 18–27 % improvement), but loses to classical on smooth M-competition monthly/quarterly (5-100 % gap). It's a specialist, not a universal replacement."*
- *"For short-history panels (`N < 50`), classical `AutoTheta` / `AutoETS` still outperform any streaming-based approach by 15–30 % MASE — a fundamental property of streaming per-observation designs, mitigated but not eliminated by `.skaters()`'s wider candidate pool."*
- *"On the M5 first-differenced bakeoff against `skaters.laplace` (both sides `sticky=True`), our port beats the reference by +1.40 nats/pred on LL and 0.021 on CRPS while running 5.9× faster."*
- *"Sticky-lattice atoms in `.skaters()` are a specialty tool: they concentrate quantile mass on integer / repeated values, which crushes WQL on continuous smooth panels (1800× worse on m1_yearly). A `.no_sticky()` variant is on the roadmap."*
- *"For long-history retail demand panels (M5, N > 1000 typical), `LaplaceForecaster::auto_aid()` matches classical MASE within 0.8 % while running 40× faster and providing native distributional output."*
- *"On the fev-27 panel, `.skaters()` runs ~5× faster than `AutoTheta` and ~90× faster than `AutoETS` — even where accuracy loses, the compute story is decisive."*
- *"Foundation model SOTA (Tirex, TimesFM-2.0) leads the best classical by ~2 % MASE on the fev panel — a data-scale advantage from pretraining, not an algorithm advantage."*
