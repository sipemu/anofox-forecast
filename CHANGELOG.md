# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0-alpha.4] - 2026-07-06

### Added

- **AR(2) leaf for `LaplaceForecaster`** (opt-in via `LaplaceForecaster::new().with_ar2(alpha_mean)` or `with_ar2_defaults()`). Uses **Yule-Walker over EMA-based autocovariance estimates** (`γ₀`, `γ₁`, `γ₂`) — more numerically stable than running centred products, and standard for streaming AR estimation. h-step forecast is recursive substitution into the AR(2) recursion.
- New public: `models::laplace::leaves::Ar2Leaf`.
- **Residual-slicing analysis in `examples/skaters_m5_benchmark.rs`** — computes per-series characteristics (`trend_strength`, `seasonality_strength`, `acf1`) and reports median MAE + Laplace-variant winrate in tercile buckets. Answers "where does leaf X help?" per characteristic. Landed in [#148](https://github.com/sipemu/anofox-forecast/pull/148); the benchmark now includes AR(2) variants (`Laplace+AR2`, `Laplace+AR2+S7`).
- **`THIRD_PARTY_NOTICES.md`** documenting `skaters` (MIT) attribution. Module doc-comment for `models::laplace` now points to it.

### Benchmark: AR(2) is a real improvement

Guided by the alpha-3 residual slicing (Laplace's MAE climbs 4.84 → 6.51 as ACF grows — AR(1) is under-fitting the longer-memory tail):

| variant | MAE (median) | MAE (mean) | vs. AutoETS wr | vs. plain wr | AR2-only vs. plain (high-ACF bucket) |
|---|---|---|---|---|---|
| Laplace (plain, 3 leaves) | 5.46 | 7.05 | 0.368 | – | – |
| Laplace + Holt | 5.54 | 7.20 | 0.334 | plain wins 82.5% | Holt wins 27.0% |
| **Laplace + AR2** | **5.37** | **6.90** | **0.385** | **AR2 wins 26.7%** | **AR2 wins 42.3%** |
| Laplace + seasonal7 | 5.15 | 6.70 | 0.466 | – | – |
| **Laplace + AR2 + seasonal7** | **5.09** | **6.64** | **0.477** | AR2+S7 wins ~90% | AR2+S7 wins 46.5% |

- AR2 improves median AND mean MAE aggregate — unlike Holt (which regressed both).
- On the high-ACF tercile (the alpha's biggest empirical weak spot), AR2 wins on 42% of series against plain — **1.6× Holt's helpfulness on the same segment**.
- `Laplace+AR2+S7` is the new best config: 47.7% winrate vs. AutoETS (up from 46.6% for S7-alone), median MAE 5.09 (5% below S7-alone).

### Attribution

The `LaplaceForecaster` design is inspired by [`microprediction/skaters`](https://github.com/microprediction/skaters) (MIT, Peter Cotton). See `THIRD_PARTY_NOTICES.md` for the full notice and license text. Empirical defaults (which leaves are on by default; leaf hyperparameters) are anofox-forecast's own — chosen from the M5 benchmark and may materially differ from skaters'.

### Notes

Alpha surface: additive behind the `distributional` feature. `LaplaceForecaster::new()` still produces the 3-leaf shell; Holt, AR(2), and seasonal are each an opt-in builder call.

## [0.12.0-alpha.3] - 2026-07-06

### Added

- **Damped-Holt leaf for `LaplaceForecaster`** (opt-in via `LaplaceForecaster::new().with_holt(alpha, beta, phi)` or the defaults shortcut `with_holt_defaults()`). Standard damped-trend recursion; `phi = 1.0` gives pure Holt, `phi ∈ (0.5, 1.0)` damps the trend. Sensible defaults: α=0.3, β=0.1, φ=0.98.
- New public: `models::laplace::leaves::HoltLeaf`.
- `examples/skaters_m5_benchmark.rs` extended to run all four Laplace variants: plain / +Holt / +seasonal7 / +Holt+seasonal7.

### Benchmark: Holt is opt-in because it *hurts* on M5 retail

The alpha-loop paid off exactly the way it's supposed to. Adding Holt to the default set (as the paper's leaf list suggests) regressed the mixture on M5 top-1000:

| variant | MAE (median) | MAE (mean) | vs. AutoETS wr | vs. plain wr |
|---|---|---|---|---|
| Laplace (plain, 3 leaves) | 5.46 | 7.05 | 0.368 | – |
| Laplace + Holt | 5.54 | 7.20 | 0.334 | plain wins 82.5% |
| Laplace + seasonal7 | 5.15 | 6.70 | 0.466 | – |
| Laplace + Holt + seasonal7 | 5.22 | 6.86 | 0.432 | seasonal-only wins 87.3% |

Retail sales series are mostly bounded / mean-reverting; Holt's noisy trend estimate steals softmax weight from the other leaves and drags the mixture down. On panels with genuine sustained trend the finding may reverse — hence the opt-in default.

**Decision rule.** Enable Holt when your series have persistent trend; leave it off for retail-style panels. The next PR will slice residuals by series characteristic (magnitude, autocorrelation, trend strength, seasonality strength) so this call gets made from data, not priors.

### Notes

Alpha surface: additive behind the `distributional` feature. `LaplaceForecaster::new()` still produces the 3-leaf shell; both Holt and seasonal are opt-in.

## [0.12.0-alpha.2] - 2026-07-06

### Added

- **Seasonal-EMA leaf for `LaplaceForecaster`** (opt-in via `LaplaceForecaster::new().with_seasonal(period)`). Maintains one EMA per phase `k ∈ 0..period`; the h-step forecast reads phase `(now + h - 1) mod period`. Unseen phases fall back to a global EMA so short-history fits don't emit `NaN`. Period is caller-supplied (no auto-detection). `with_seasonal(0)` / `with_seasonal(1)` is a no-op.
- New public: `models::laplace::leaves::SeasonalEmaLeaf`.
- `LaplaceForecaster::seasonal_alpha(alpha)` builder to override the seasonal leaf's smoothing rate (default 0.15).
- `examples/skaters_m5_benchmark.rs` (added in [#144](https://github.com/sipemu/anofox-forecast/pull/144)) now runs both the plain and `with_seasonal(7)` configurations.

### Benchmark

On the M5 top-1000 panel (999 non-intermittent series, 28-day horizon, weekly seasonality):

| model | MAE (median) | MAE (mean) | vs. AutoETS winrate | fit time |
|---|---|---|---|---|
| Laplace (plain) | 5.46 | 7.05 | 0.368 | 0.24 s |
| **Laplace+seasonal7** | **5.15** | **6.70** | **0.466** | **0.24 s** |
| AutoETS | 4.77 | 6.23 | — | 26.1 s |

One additional leaf closed ~40% of the MAE gap and +9.8 pp of winrate against AutoETS at zero fit-time cost. Laplace+seasonal7 beats plain Laplace on 91% of series — the leaf is doing real work.

### Notes

Alpha surface — additive behind the `distributional` feature. `LaplaceForecaster::new()` still produces the 3-leaf shell; the seasonal leaf must be requested explicitly.

## [0.12.0-alpha.1] - 2026-07-06

### Added

- **`LaplaceForecaster` — distributional forecasting shell (alpha, `distributional` feature).** Streaming, likelihood-weighted mixture over three cheap leaves (EMA, drift, AR(1)); emits a `GaussianMixture` per horizon rather than a point forecast. Inspired by the shell design in [`microprediction/skaters`](https://github.com/microprediction/skaters); the full skaters ensemble (Holt, fractional-differencing, seasonal, Yeo-Johnson, OU mean-reversion, CRPS-tuned terminal leaf) is intentionally deferred. New public surface:
  - `models::laplace::{Gaussian, GaussianMixture}` — distribution primitives with `mean`/`std`/`quantile`/`logpdf`/`cdf`.
  - `models::laplace::Leaf` — streaming per-horizon predictor unit.
  - `models::laplace::LaplaceForecaster` — implements `Forecaster` (point = mixture mean), `Inspectable` (`Explanation::Laplace(_)`), and the new `DistributionalForecaster` trait.
  - `models::laplace::DistributionalForecaster: Forecaster` — object-safe sibling trait exposing `forecast_dist(&self, horizon) -> Result<Vec<GaussianMixture>>`. Point forecasters remain on `Forecaster` alone; distributional forecasters implement both.
  - `models::LaplaceExplanation` — payload with `horizon_dists`, `leaf_weights`, `leaf_names`, `fitted_values`, `residuals`.
- The `Explanation` enum gained a `#[cfg(feature = "distributional")] Laplace(LaplaceExplanation)` variant. The #107 conformance suite covers it.

### Notes

- Alpha surface: the leaf set and mixing scheme may change before the stable `0.12.0`. No downstream code is broken by default builds — the module and enum variant only compile when `distributional` is enabled.
- Not recommended for price/return series (the empirical wins in the skaters paper are on non-price economic series).

## [0.11.0] - 2026-06-26

### Added

- **`Forecaster::explanation()` reads fitted params without re-fitting** (closes #136). The trait gained a default `fn explanation(&self) -> Result<Explanation>` (default returns `Err(InvalidParameter)`). Every model that already implements `Inspectable` now overrides it with a one-line delegate: `RegressionForecaster`, `AutoARIMA`, `AutoETS`, `AutoTheta`, `AutoTBATS`, `MFLES`, `MSTLForecaster`. Callers holding `Box<dyn Forecaster>` can now read `model.explanation()` directly — no second-trait bound, no downcast, no re-fit. Downstream `model_params` extraction was previously paying the entire per-series fit cost a second time (~178 s of a 185 s output stage on a 6,319-series / 7-method panel; projected to ~2.9 h on a 367k-series panel); the extra fit goes away once callers route through the trait method.
- The standalone `Inspectable` trait is kept as-is — the new `Forecaster::explanation` delegates to it, so existing direct uses continue to work. No deprecation.

### Migration

Additive change, backward-compatible. Existing code that imports `Inspectable` and calls `model.explanation()` on a concrete model may hit `E0034` (multiple methods named `explanation`); drop the `Inspectable` import — `Forecaster::explanation` is in scope wherever the model is fit, and delegates to `Inspectable`.

## [0.10.2] - 2026-06-24

### Fixed

- **Restore default `RegressionBackend::Dynamic` behaviour; make the v0.10.1 short-series OLS fast-path opt-in** (closes #137). v0.10.1 enabled the fast-path unconditionally and traded a real win — MAE-neutral, ~11× faster — for a downstream-measured **+24% rise in delivered median |bias|** on a 367k-series panel: short-series DynLM was the low-bias method (win-rate 24.8% on v0.10.0), and the fast path silently replaced it with OLS. Point accuracy (MAE) was unchanged, but bias regressed.

  v0.10.2 reverts `RegressionForecaster::dynamic` / `dynamic_smoothed` to the full IC-weighted fit on every series — bit-for-bit the same behaviour as v0.10.0. Callers who measured the perf/bias trade-off and want the fast path can opt in via the new `dynamic_fast` / `dynamic_smoothed_fast` constructors.

  The `RegressionBackend::Dynamic` variant gained a `short_series_ols_fast_path: bool` field (struct-literal callers must initialise it; convenience constructors handle it for you).

## [0.10.1] - 2026-06-24

### Performance

- **Short-series fast path on `RegressionBackend::Dynamic`** (closes #134). Below `max(60, 4 × (p+1))` observations the backend drops to a single `OlsRegressor` fit; long series take the full dynamic path. **Superseded by #137 in v0.10.2**: the fast-path is now opt-in via `dynamic_fast` / `dynamic_smoothed_fast` because the default-on behaviour shipped here caused a +24% delivered-bias regression at panel scale.

## [0.10.0] - 2026-06-21

### Changed

- **Matrix-free MinT solver for grouped hierarchies** (closes #130). The variance-weighted MinT path (`MinTraceVariance` and `MinTraceStruct`) was correct on grouped hierarchies after #124 but materialised the dense `M×M` normal matrix `S'W⁻¹S` — capping practical scale at ~10k leaves before OOM. On a 47,640-leaf site×part panel that's a 36 GB peak RSS spike; the corresponding 569,389-leaf full panel would need ~2.6 PB for the dense matrix alone.

  `min_trace_diagonal` now auto-switches based on M:

  - **M ≤ 1000** — dense Cholesky path (existing, ~8 MB at the threshold, factor reused across the horizon — fastest for small inputs).
  - **M > 1000** — conjugate gradient with Jacobi preconditioner, applying `A = S'W⁻¹S` as `Sᵀ·(W⁻¹·(S·p))` via sparse mat-vecs over the existing `sparse_s` ancestor lists. Per-iteration `O(nnz(S)) = O(M · depth)`. Memory `O(M + N + nnz(S))` — well under a GB for million-leaf hierarchies. The diagonal preconditioner reuses the dense path's `sts_diag` and costs nothing extra.

  Both paths produce identical output on small inputs; CG converges in ≤ depth + O(1) iterations under the diagonal preconditioner for the panel scales seen in practice. Tested at 40×40 = 1600 leaves crossing the auto-switch boundary; downstream consumer has the same approach scaling to 569k leaves.

  No API changes — the switch is internal to `min_trace_diagonal`. `MinTraceVariance` and `MinTraceStruct` continue to work exactly as before for small inputs and now handle large grouped panels too.

### Out of scope

- `min_trace_shrink` still builds an N×N sample covariance, and the dense `min_trace_ols` still allocates `M×M`. Both are tree-only methods where the scaling problem is smaller in practice; matrix-free variants are a possible future refinement if a real workload hits the wall.

## [0.9.3] - 2026-06-21

### Changed

- **M4-Daily SF baseline regenerated from `statsforecast 2.0.3`** (`tests/m4_daily_accuracy_regression.rs`). The v0.9.1 reference values were hand-copied from issue #64 and several were wrong — most notably D2085 (69.2 → 257.1, an inflated value that motivated the v0.9.2 per-series exemption). With the correct baseline:
  - **D2085 lands at 0.99×** — anofox is actually slightly better than SF, not a 3.68× outlier.
  - **`PER_SERIES_EXEMPTIONS` emptied** — no series need an exemption now; all 10 land below 1.75×.
  - **`AGGREGATE_TOLERANCE` retuned 1.15× → 1.30×**. The true SF mean (776.16) is lower than the wrong SF mean (858.59) because D2085 was inflating it; the same anofox numerator now gives a 1.222× ratio. D2172 (anofox 4628 vs SF 2670) is the biggest single-series gap, contributing ~25% of the aggregate. The new gate at 1.30× gives ~8pp future-regression headroom — v0.5.6's +37% regression would still land at ~1.55× and trip the gate.
- All three M4 accuracy gates remain active in CI; no `#[ignore]`d test remains.

## [0.9.2] - 2026-06-20

### Fixed

- **AutoARIMA short-series drift over-fitting** (closes #128). The `(S)ARIMA(p,1,q)(P,0,Q)` optimiser was unconditionally fitting a drift / intercept term on the differenced series. On long stable series the AICc verdict matches the forecast verdict; on short / regime-changed series the drift shaves a tiny amount off the in-sample residual variance (1–2 AICc units in its favour) but the integrated forecast then extrapolates the historical mean instead of the recent regime, walking away from the last training value at `+mean(diff)` per step. The M4-Daily benchmark (issue #64) caught it: D2085 (n=93) ran at 7.65× the SF baseline, D4047 (n=162) at 4.06×.

  Fix: when `d + cap_D == 1` and the model has any AR/MA terms, fit two variants — with-drift and without-drift — and compare AICc, matching R's `auto.arima` with one critical twist. On **short** series (`n < 200`) require drift to win by ≥ 2.0 AICc units (the cost of one extra parameter); on **long** series accept any strict improvement. This bifurcation reflects that the AICc verdict tracks the forecast verdict on long series but not on short ones. Wired into both `ARIMA::fit` and `SARIMA::fit`.

  M4-Daily results vs 0.9.1: D2085 dropped 529 → 255 MAE (7.65× → 3.68×), D4047 dropped 587 → 143 MAE (4.06× → 0.99×), every other series unchanged within float epsilon, aggregate ratio 1.189× → 1.105×.

### Changed

- **`tests/m4_daily_accuracy_regression.rs` gates tightened**:
  - `AGGREGATE_TOLERANCE`: 1.20× → 1.15×. Now lands at 1.105 with headroom.
  - Per-series 2× test promoted from `#[ignore]` to active CI. D4047 is now under the bound; D2085 is exempted with documented rationale (its test split ends with an unforecastable 6080 dip from an 8800 baseline at h=14, so even a perfect flat predictor would MAE ~257 ≈ 3.71× SF — a data-quality artifact, not a model gap).
  - New `PER_SERIES_EXEMPTIONS` table gives the test a structured escape hatch for data-quality artifacts.
  - All three M4 accuracy gates now active in CI.

## [0.9.1] - 2026-06-20

### Added

- **M4-Daily AutoARIMA accuracy regression tests** (closes #64). New integration suite at `tests/m4_daily_accuracy_regression.rs` runs `AutoARIMA::seasonal(7)` against 10 representative M4-Daily series (fixture at `tests/data/m4_outliers.json`) and asserts forecast accuracy stays within bounded multipliers of the `statsforecast 2.0.3` baseline. Three layered gates:
  - **Aggregate** (active in CI): `mean(anofox_mae) / mean(SF_mae) ≤ 1.20×`. Catches the v0.5.6-shaped distribution-wide regression that motivated this issue (currently 1.189×).
  - **Catastrophic** (active in CI): no series may exceed `SF_mae × 10`. Hard wall against silent quality cliffs.
  - **Per-series 2× tolerance** (`#[ignore]`d, documented): currently flags D2085 (~7.65×) and D4047 (~4.06×) — the documented short-series quality gap. The gate is wired and will fire on `cargo test -- --ignored`; un-ignore once short-series quality work catches up.
  - Cost: ~0.4 s in release build for all three tests; ARIMA fits run in series across 10 fixtures.

## [0.9.0] - 2026-06-20

### Added

- **Grouped / crossed hierarchies via `HierarchyTree::from_summing_matrix`** (closes #124). Supports Hyndman GTS-style hierarchies where each leaf belongs to multiple aggregate dimensions at once — e.g. `(site, part)` leaves rolling up to `site_total`, `material_total`, and `grand_total` simultaneously. Caller supplies the leaf→ancestor row lists directly; the constructor wires the leaf→aggregate parent edges and infers aggregate→aggregate edges via leaf-set containment (a single grand-total root + layered intermediates so `bfs_order` can walk the DAG cleanly). The variance-weighted `MinTraceVariance` and `MinTraceStruct` reconcilers operate on the resulting `sparse_s` directly and produce coherent forecasts where every aggregate (sites, materials, total) equals the sum of leaves under it.

### Changed

- **`Node.parent: Option<usize>` → `parents: Vec<usize>`** (internal). `HierarchyTree::new` no longer rejects multi-parent edges and accepts repeated `(parent, child)` edges naturally. The multiple-roots check remains — every hierarchy still needs exactly one grand-total node. Tree-mode reconcilers (`BottomUp`, `TopDown`, `MiddleOut`, dense `MinTraceOls` / `MinTraceShrink`) continue to work on strict trees; grouped hierarchies should use the variance-weighted MinT path.
- Four parent-walking sites (dense S construction in `MinTraceOls` / `MinTraceShrink`, sparse S construction in `MinTraceVariance` / `MinTraceStruct`, and `is_descendant_of`) refactored onto a shared `ancestors_of` helper that walks every parent edge transitively. Identical results for strict trees; correct transitive ancestor set for grouped DAGs.
- `bfs_order` now de-duplicates via a seen-set so each node appears exactly once even when reachable from the root via several paths.

## [0.8.7] - 2026-06-20

### Added

- **Levenbach STI_Class taxonomy** in `forecastability::sti_class` (closes #93). Classifies a monthly series into one of six categories — `Sit`, `Sti`, `Ist`, `Tsi`, `Its`, `Tis` — by ranking the relative magnitude of Seasonal / Trend / Irregular mean-squares from a two-way ANOVA without replication on a `years × months` grid. Returns `StiClassResult { class, sf_seas, sf_trnd, sif, tif }` with strength factors and F-statistics. Returns `None` on degenerate inputs (too few observations, non-finite values, zero variance, grid dims < 2). Designed for monthly granularity — sub-monthly use cases see ~93% land in a single trend-dominant class. Gated behind the `forecastability` feature. Reference: Levenbach (2025).

- **`RegressionForecaster::predict_with_exog_intervals(horizon, future_regressors, level)`** (closes #123). The trait already declared the method with a points-only default; this commit overrides it on `RegressionForecaster` to surface true OLS / WLS prediction intervals through the exog future-design matrix — mirroring the existing `predict_with_intervals` (no-exog) sibling. Unblocks orchestrator metalearners that were emitting NaN `forecast_q*` for ~42% of rows on regression-family methods at 569k-series panel scale. Recursive (lag-using) models fall back to point-only — same contract as `predict_with_intervals`, since direct OLS PIs don't apply to recursive forecasts.

## [0.8.6] - 2026-06-17

### Added

- **`TrendType::Logistic`** as a regression-design feature (closes #121). `RegressionFeatures::with_trend_component(TrendType::Logistic)` now fits a `LogisticTrend` (S-curve to capacity `K`) as a `"__logistic_trend"` column in the design matrix. Capacity defaults to `CapacityMode::Auto` (`max(window) × 1.5`), matching the issue's recommendation. Lets `RegressionForecaster` model ramp-up products that plateau — saturating extrapolation rather than runaway linear growth. Mirrors the existing `Exponential` plumbing across `TrendType`, `FittedComponentState`, `feature_names`, `classify_features`, `build_matrices`, and the design-matrix column emission.

- **`LogisticTrend` is now an `AutoTrend` candidate**. The default candidate list grows from `{Linear, Quadratic, Cubic, Exponential, TheilSen, PiecewiseLinear (auto-penalty)}` to also include `Logistic`. Doubles as a **saturating-series detector** — after `AutoTrend::fit_trend`, reading `selection_result().selected == "Logistic"` (or `"Logistic"` ranking ahead of `"Exponential"` in `scores`) flags an S-curve. Composes with the v0.8.5 auto-penalty `PiecewiseLinear` candidate: both are valid saturation detectors and the user gets both signals from one fit.

### Known limitation

- `LogisticTrend::CapacityMode::Auto` uses `K = max(window) × 1.5`. On series that haven't fully saturated, that under-estimates the true ceiling and biases the linearised fit. If you have prior knowledge of the cap, `LogisticTrend::new().with_capacity(K)` recovers the right shape. Joint MLE over `(K, r, t₀)` is a possible future refinement.

## [0.8.5] - 2026-06-16

### Added

- **`PiecewiseLinearTrend::with_auto_penalty()`** — per-series automatic PELT penalty selection via CROPS + elbow (Haynes et al. 2017). Drop-in replacement for hand-tuning the `penalty` knob: when enabled, `fit_trend` runs `Pelt::auto_detect` over a geometric range and picks the elbow where adding one more knot stops paying for itself. Default remains `auto_penalty = false` so existing callers are unaffected.
- **`AutoTrend`'s PiecewiseLinear candidate** now uses `.with_auto_penalty()`. Previously it always ran with `penalty = 10`, so PiecewiseLinear could lose to Quadratic on series where a different penalty would have made it the best fit. With per-series penalty selection it competes fairly in the AICc / BIC / Holdout bake-off.

### Fixed

- **Plain WLS backend no longer returns silent-NaN coefficients** (closes #119). On heavy-feature / small-effective-N designs (e.g. `wls_logistic` with `offset = 24` against ~20 categorical dummies on a short window), `WlsRegressor` could return a successful result whose coefficients were `NaN` — the NaN then propagated into fitted values and forecasts, which silently looked "successful" to downstream consumers and quietly dropped the method from honest backtests (orchestrator repro: 151,656 / 151,656 NaN fold rows). The `Wls` fit_to arm now walks `result.coefficients` and `result.intercept` after the solve; any non-finite entry not explained by `result.aliased` surfaces a clean `Err` pointing the caller at `wls_logistic_ridge` with λ > 0 (same recency weighting plus L2 regularisation). The legitimate pivoted-out aliased path is preserved.

## [0.8.4] - 2026-06-16

### Fixed

- **`WlsLogisticRidge` standard errors stay finite under heavy-feature / small-effective-N designs** (closes #117). The v0.8.3 SE block used the simple form `Cov = σ̂² · (XᵀWX + λI)⁻¹` and returned NaN for almost every series on the motivating panel-scale repro (≈20 features, logistic `offset = 24`). Two fixes in the same block:
  - **Switch to the ridge-adjusted sandwich covariance** `Cov = σ̂² · A⁻¹ · (Xfullᵀ W Xfull) · A⁻¹` where `A = Xfullᵀ W Xfull + λI_β`. This is the "more correct form when ridge λ>0 is active" called out in #115 — accounts for the shrinkage and stays meaningful when `Xfullᵀ W Xfull` alone is rank-deficient.
  - **Clamp the effective residual degrees of freedom** `df_eff = (Σ wᵢ − (p + 1)).max(1.0)`. The v0.8.3 code set `σ̂² = NaN` whenever `Σ wᵢ ≤ p + 1`, propagating NaN through every coefficient SE — exactly the regime where ridge was needed. Clamping to ≥ 1 produces conservative-but-finite SEs in those cases; no-op when there's headroom.

## [0.8.3] - 2026-06-16

### Fixed

- **`WlsLogisticRidge`: standard errors are now computed on the original-scale design** (closes #115). The v0.8.1 implementation built `RegressionResult` by cloning the augmented OLS fit's result and overwriting intercept / fitted / residuals / R² on the original scale — but never replaced `std_errors`, so the SE values surfaced via `RegressionExplanation` came from the centred + √W-scaled + Tikhonov-augmented design (wrong units, silently wrong/empty for downstream consumers). The fix recomputes `Cov(β̂) = σ̂² · (Xfullᵀ W Xfull + λI_β)⁻¹` inline using faer Cholesky, populating `result.std_errors` and `result.intercept_std_error` so the Inspectable surface returns finite, positive SEs aligned with the coefficients.

## [0.8.2] - 2026-06-15

### Fixed

- **`RegressionForecaster`: columns claimed by `exog_features` are now excluded from the raw exog block** (closes #113). Previously, declaring a column via `with_categorical("is_chp", …)` while leaving `.exog()` enabled inserted that column both as a raw numeric column *and* as its one-hot dummies, producing perfect collinearity and a singular design matrix that blew OLS up. The fix excludes any column referenced by an `exog_features` spec (Categorical / Lag / Rolling / Polynomial / Interaction) from the raw exog name list — letting callers declare *some* regressors categorical and others numeric, without the all-or-nothing `.no_exog()` workaround.

### Changed

- **`predict_with_exog` validation now covers all spec-claimed columns**. Columns referenced by `exog_features` entries (which the encoder reads via `future_at` at predict time) are now required in `future_regressors`. A missing claimed column surfaces a clear `InvalidParameter` error instead of silently producing zeros.

### Breaking (narrow)

- A caller who deliberately relied on `.exog().with_exog_lags("foo", &[1])` to produce both raw `foo` *and* `foo_lag1` columns no longer gets the raw column. The motivating Categorical case is the strictly-correct fix; for the lag/rolling cases this is a behaviour change. Workaround: declare derived features on a different column name, or accept the loss of raw column.

## [0.8.1] - 2026-06-15

### Added

- **Per-coefficient standard errors in `RegressionExplanation`** (closes #111). Two new owned fields on the `Regression` variant of `Explanation`:
  - `coef_std_errors: Vec<f64>` — aligned with `coefficients`; empty when the backend doesn't compute inference (Poisson, Tweedie, BLS, RLS).
  - `intercept_std_error: f64` — `NaN` when not available.
  Populated for the default linear backends (OLS, Ridge, ElasticNet, WLS — `RegressionOptions::compute_inference` is `true` by default). SEs under regularised / weighted backends are nominal (sandwich-style approximation) — fine for display but not exact frequentist intervals. Unblocks downstream coefficient forest plots and `model_params` parquet emission in anofox-orchestration.

- **`RegressionBackend::WlsLogisticRidge { offset, lambda }`** (closes #103). New backend that combines logistic recency weighting with L2 regularization in a single fit, solving the weighted Ridge normal equations `β = (Xᵀ W X + λI)⁻¹ Xᵀ W y` exactly. Convenience constructor `RegressionForecaster::wls_logistic_ridge(offset, lambda, features)`. Implementation: weighted centering by `ȳ_w` / `X̄_w`, sqrt(W) on centred data, Tikhonov `√λ · I_p` augmentation, OLS without intercept, then β + analytical intercept reconstructed on the original scale via a small `WlsLogisticRidgeFitted` wrapper. Fixes the catastrophic-extrapolation problem seen on heavy feature designs with small effective N (5-lag + Fourier + rolling features under `wls_offset: 30` — previously median MAE 669–1249 vs Ridge's 0.483; with combined fit, regularization stabilises the slopes).

- **`AutoForecast` candidate-set extension** (closes #105). Three new opt-in include flags on `AutoForecastConfig` and matching builder methods (`include_tbats`, `include_mfles`, `include_mstl`), defaulting to `false`. Each adds CV-fit cost so users opt in when they suspect their series benefits from complex / multiple seasonality (TBATS), smooth trend + Fourier (MFLES), or STL-style trend-cycle separation (MSTL). TBATS and MSTL silently skip when `seasonal_period` is unset (matches the existing seasonal-only convention).

## [0.8.0] - 2026-06-10

### Added

- **`Forecaster` decomposition + training-state accessors** (closes #106) — three new methods on the `Forecaster` trait expose primitive fit state previously only available through model-specific accessors. All three have default impls so existing implementors continue to compile.
  - `trend_component() -> Result<&[f64]>` — per-row trend (where defined; `Err(InvalidParameter)` otherwise).
  - `seasonal_component() -> Result<&[f64]>` — per-row seasonal contribution (sum of components for multi-seasonal models like MSTL).
  - `residual_component() -> Result<&[f64]>` — per-row residual; default delegates to `residuals()`. MSTL overrides to return the STL remainder.
  - `training_values() -> Result<&[f64]>` — the values the model was fit on.
  - `training_regressors() -> Option<&HashMap<String, Vec<f64>>>` — exogenous regressors retained at fit time.
  - Implemented natively on 11 auto-tuned models: `MSTLForecaster`, `MFLES`, `AutoARIMA`, `SimpleExponentialSmoothing`, `HoltLinearTrend`, `HoltWinters`, `SeasonalES`, `AutoETS`, `AutoTheta`, `DynamicTheta`, `OptimizedTheta`, `AutoTBATS`. `predict_with_exog` is also routed through a residual-Ridge shim (`src/utils/exog_shim.rs`, Cholesky-solved closed form) so models that don't natively support regressors can still accept them at fit time.
  - Cross-cutting conformance test at `tests/issue_106_decomposable_conformance.rs` validates the contract (trend + seasonal + residual ≈ training, with documented exceptions for AutoARIMA differencing).

- **`Inspectable` trait + `Explanation` enum** (closes #107) — typed sibling to the existing `Explainable` (forecast-decomposition) trait. Where `Explainable` decomposes a *forecast*, `Inspectable` packages the *fit*: a per-model snapshot of internal state and interpretable parameters.
  - `Inspectable::explanation(&self) -> Result<Explanation>` — object-safe, so `Box<dyn Inspectable>` works.
  - `Explanation` is an owned enum with seven variants: `Regression`, `Ets`, `Arima`, `Mfles`, `Theta`, `Tbats`, `Mstl`. Each carries the universal spine (`fitted_values`, `residuals`) plus model-specific scalars (e.g. ETS spec + α/β/γ/φ, ARIMA order + AIC/BIC + coefficients, Regression R² + feature names + coefficients + intercept, TBATS Box-Cox λ + selected config).
  - All variants are `Clone + PartialEq + Default`. Behind the `serde` feature: `Serialize + Deserialize` for caching, transport, and UI surfacing.
  - Implemented on `RegressionForecaster`, `MSTLForecaster`, `MFLES`, `AutoARIMA`, `AutoETS`, `AutoTheta`, `AutoTBATS`. Returns `Err(FitRequired)` before fit.
  - Cross-cutting conformance test at `tests/issue_107_inspectable_conformance.rs` covers per-model variant matching, spine invariants, object-safety, and serde-JSON serialization across every variant.

## [0.7.6] - 2026-05-31

### Added

- **`WeightStrategy::Logistic { offset }`** (closes #98) — sigmoid recency weights for `RegressionForecaster::wls(…)`. Weight at observation `i` of an `n`-row training set is `1 / (1 + exp(-((i as f64) - (n as f64 - offset))))`, so the last ~`offset` observations contribute near 1.0 and older ones decay through a single inflection point and plateau near 0. Convenience constructor `RegressionForecaster::wls_logistic(offset, features)` mirrors the existing `wls_decay`. Different shape from `ExponentialDecay`: smooth plateaus at both ends with a single inflection, rather than monotonic geometric decay.

## [0.7.5] - 2026-05-13

### Added

- **Full [`ruptures`](https://github.com/deepcharles/ruptures)-parity changepoint surface** in `src/changepoint/`. New trait-based API lives alongside the existing free-function one (`pelt_detect`, `PeltConfig`, `CostFunction` enum unchanged). Mirrors `ruptures` 1.1.9.
  - **Traits**: `Cost` (`fit` + `error(start, end)` + `min_size` + `name`), `Detector` (`fit` + `predict_pen` / `predict_n_bkps` / `predict_eps`), `DetectorResult` (segments + changepoints + n_changepoints helpers, terminal-`n` included matching ruptures convention).
  - **`Signal` carrier** for univariate / multivariate (`From<&[f64]>` for univariate ergonomics).
  - **Detectors** (6 of 6 from ruptures): `PeltDetector<C>` (PELT pruning), `DynpDetector<C>` (exact O(K·n²) dynamic programming), `BinsegDetector<C>` (greedy binary segmentation), `BottomUpDetector<C>` (agglomerative merge), `WindowDetector<C>` (sliding-window discrepancy), `KernelCpdDetector` (kernel changepoints — Linear / Rbf / Cosine, O(n²) gram cumsum).
  - **Cost functions** (10 of 10 from ruptures + 3 extras): `CostL1`, `CostL2`, `CostNormal`, `CostLinear` (multivariate OLS), `CostRank` (rank-then-L2), `CostMahalanobis` (user-supplied metric matrix), `CostAR` (autoregressive RSS via per-segment Cholesky), `CostRbf` (RBF kernel with median-heuristic γ), `CostCosine` (cosine kernel), `CostCLinear` (continuous piecewise linear); plus `CostPoisson`, `CostMeanVariance`, `CostCusum`, `CostLinearTrend` retained from v0.7.x.
  - **Metrics** (3 of 3 from `ruptures.metrics`): `precision_recall(truth, pred, margin)`, `hausdorff(a, b)`, `randindex(truth, pred, n)`.
- **Parity validation against `ruptures==1.1.9`**:
  - `scripts/generate_ruptures_fixtures.py` — Python script (pinned to ruptures 1.1.9, numpy 2.4.6) emits JSON fixtures recording the input signal, parameters, and ruptures-computed breakpoints + total cost. Version header lets the Rust test refuse on drift.
  - 5 committed fixtures under `tests/data/ruptures_fixtures/` covering Pelt + Dynp + Binseg + BottomUp + Window with L2 cost.
  - `tests/changepoint_ruptures_parity.rs` (gated on the `serde` feature): loads each fixture, runs the Rust port with identical parameters, and asserts exact breakpoint match for the deterministic detectors (Pelt, Dynp, Binseg, BottomUp), ±2·jump tolerance for Window, and total cost matching to 1e-6 relative tolerance across all five.
- 69 new unit tests + 5 parity integration tests; lib suite 2843 → 2912 (+69) under `--all-features`.

## [0.7.4] - 2026-05-07

### Added

- **Rolling-feature kinds**: `RollingStatKind` gains `Quantile { tau }`, `Range`, `Iqr`, `Skew`, `Kurt`, `Slope` (OLS slope), `Rank` (fractional rank of latest value), `ZScore`, `CountAbove { threshold }`, `CountBelow { threshold }`. Convenience builders on `RegressionFeatures`: `with_rolling_quantile`, `with_rolling_range`, `with_rolling_iqr`, `with_rolling_skew`, `with_rolling_kurt`, `with_rolling_slope`, `with_rolling_rank`, `with_rolling_zscore`, `with_rolling_count_above`, `with_rolling_count_below`.
- **`EventDistanceFeature`** — new `RecursiveFeature` impl for *steps-since-last* / *steps-until-next* arbitrary event indices (holidays, promos, regime markers). Added `EventDistanceMode` enum and `RegressionFeatures::with_event_distance(events, mode)` builder. Trait extended with default-impl `compute_predict_at(recent, step_h, n_train, out)` so position-aware features can resolve absolute indices at predict time; existing `RecursiveFeature` impls (incl. `RollingFeature`) are unaffected.
- **Lag-of-exog and rolling-of-exog** in `RegressionFeatures`: new `ExogFeatureSpec` enum and `with_exog_lags(col, &[k])` / `with_exog_rolling(col, window, lag, kind)` builders. Predict-time resolution combines a stored exog tail with caller-supplied future regressors.
- **MI-based feature selection** (gated behind `forecastability`): `rank_features_mi(features, target, k)` and `select_features_mi(features, target, k_neighbours, top_k)` in `features::selection` — uses the existing KSG1 estimator from `forecastability::knn_mutual_information`.
- **Polynomial and interaction exog features** (closes #81): new `ExogFeatureSpec::Polynomial { col, degree }` and `ExogFeatureSpec::Interaction { col_a, col_b }` variants. Builders `RegressionFeatures::with_exog_polynomial(col, degree)` (emits `col^2..col^degree`) and `with_exog_interaction(col_a, col_b)` (emits `col_a * col_b`). Captures nonlinear and multiplicative effects in linear regression backends.
- **Categorical encoding for exog columns** (closes #82): new `CategoricalStrategy` enum (`OneHot { drop_first }`, `Ordinal`, `Count`, `Target { smoothing }`) and `ExogFeatureSpec::Categorical { col, categories, strategy }`. Builder `RegressionFeatures::with_categorical(col, categories, strategy)`. Categories are pre-declared for deterministic shape; unseen codes at predict time fall back to baseline / unused-index / 0 / grand mean. `Count` and `Target` are flagged `FeatureSafety::DataDependent` so CV folds know to refit.
- **Yeo-Johnson transform** (closes #83): new `YeoJohnsonTransform` (auto- and fixed-λ) plus standalone `yeo_johnson` / `inv_yeo_johnson` / `yeo_johnson_lambda` / `yeo_johnson_auto` helpers in `transform::yeo_johnson`. Yeo-Johnson generalises Box-Cox to handle zeros and negatives — useful for returns, deltas, and residual-style series where `BoxCoxTransform` would error. Reference: Yeo & Johnson (Biometrika 2000).
- **Cross-series (panel) feature aggregations** (closes #84): new `features::panel` module with `PanelAggregator` enum (`Mean`, `Median`, `Std`, `Rank`) and `panel_aggregate(values, kind, exclude_self)` plus per-aggregator convenience helpers. Computes per-timestamp aggregates over a panel of N series, with an optional leakage-safe leave-one-out mode (`exclude_self = true`). Useful as additional regressors for global / batch forecasters in retail-panel or sensor-fleet settings.
- **Multicollinearity diagnostic** (closes #85): new `validation::multicollinearity` module with `variance_inflation_factors(columns)`, `condition_number(columns)`, and a combined `multicollinearity_report(columns, names)` returning a `MulticollinearityReport` (per-column VIF + `Severity::{Ok, Warn, Fail}` flags + condition number with Belsley-Kuh-Welsch threshold). Use before fitting Ridge / ElasticNet / OLS to surface near-singular design matrices. Default thresholds: warn at VIF > 5, fail at VIF > 10, ill-conditioned at cond > 30.

## [0.7.3] - 2026-05-05

### Added

- **Three new conformal prediction methods** (in `postprocess::`, behind the `postprocess` feature):
  - **CQR** (`CqrPredictor`, `CqrResult`) — Conformalized Quantile Regression (Romano, Patterson & Candès, NeurIPS 2019). Wraps any quantile-regression base learner; conformity score `E_i = max(q_lo - y, y - q_hi)` adjusts the bounds symmetrically. Tighter than absolute-residual conformal under heteroscedasticity.
  - **EnbPI** (`EnbPiPredictor`, `EnbPiResult`) — Ensemble Bootstrap Prediction Interval (Xu & Xie, ICML 2021). Leave-one-out residuals from a bagged ensemble (no retraining required). Online residual window via `update()` for distribution drift.
  - **ACI** (`AciPredictor`) — Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021). Stateful streaming wrapper that adjusts α_t each step from coverage error: `α_{t+1} = α_t + γ(α_target - err_t)`. Long-run coverage converges to target under arbitrary distribution drift.
- 22 new tests covering all three predictors, including a stationary-noise CQR coverage test and a drift-response ACI test.

## [0.7.2] - 2026-04-21

### Added

- **Triage pipeline** (`forecastability::triage`) — single-call and batch series classification with model family routing.
  - `run_triage(series, config)` → `TriageResult` with pattern (A–E) + recommended `ModelFamily`
  - `run_batch_triage(all_series, config)` → `BatchTriageResult` with pattern/family counts (rayon-parallelized)
  - `screen_exogenous(target, candidates, max_lag)` → ranked exogenous candidates by transfer entropy
  - `SeriesPattern` enum: WhiteNoise (A), Linear (B), Seasonal (C), Nonlinear (D), Complex (E)
  - `ModelFamily` enum: Skip, LinearStatistical, SeasonalStatistical, NonlinearML, Ensemble
  - `TriageConfig` builder: max_lag, n_surrogates, alpha, seed

### Fixed

- **Fingerprint significance test** (#74): switched from rank-based (max of surrogates) to parametric 3σ (`mean + 3*std`) threshold matching the Python `dependence-forecastability` original. Much more selective, especially with few surrogates.
- **Normalized `information_mass`**: now divided by `max_lag` for cross-setting comparability.
- Added `std` and `threshold_3sigma` fields to `SignificanceBands`.

## [0.7.1] - 2026-04-21

### Changed

- **Forecastability module ~60× faster**: 2D KD-tree for kNN MI (O(n²) → O(n log n)), Kendall tau O(n log n) via merge-sort, fused distance correlation, digamma lookup table, probit cache, streaming surrogates, rayon parallelization of surrogate + lag loops.

### Added

- 19 cross-validation tests against scipy/numpy reference values with tight (±2-3%) tolerances. Covers kNN MI, GCMI, distance correlation, Pearson/Spearman/Kendall, digamma, and the forecastability fingerprint (AR(1), white noise, logistic map).

## [0.7.0] - 2026-04-21

### Added

- **Forecastability analysis module** (`forecastability::`, feature-gated behind `forecastability`) — pre-modeling triage via information-theoretic dependence measures. Port inspired by [dependence-forecastability](https://github.com/AdamKrysztopa/dependence-forecastability) (MIT).
  - **kNN Mutual Information** (Kraskov KSG1): `knn_mutual_information(x, y, k)` — brute-force O(n²k) estimator with digamma, binary-search marginal counting.
  - **AMI curve**: `ami_curve(series, max_lag)` — MI at each horizon lag h=1..max_lag.
  - **pAMI curve**: `pami_curve(series, max_lag, backend)` — partial/conditional AMI via linear residualization (OLS).
  - **Transfer Entropy**: `transfer_entropy_curve(source, target, max_lag)` — directional TE as conditional MI.
  - **GCMI**: `gcmi(x, y)` — Gaussian Copula MI (Ince 2017), rank → probit → closed-form `I = -0.5 log₂(1 - ρ²)`. Plus `gcmi_curve`.
  - **Distance correlation**: `distance_correlation(x, y)` — Szekely/Rizzo (2007), O(n²) doubly-centered distance matrices.
  - **Phase-randomized surrogates**: `phase_surrogates(series, n, seed)` — FFT + random phases + IFFT. `significance_bands()` for any lag-curve metric.
  - **Lag correlations**: `pearson_curve`, `spearman_curve`, `kendall_curve` — |corr(X_t, X_{t+h})| at each lag.
  - **Forecastability fingerprint**: `ForecastabilityFingerprint::compute()` — `information_mass`, `information_horizon`, `information_structure`, `nonlinear_share`, `signal_to_noise`, `directness_ratio`, `informative_horizons`.
  - **Largest Lyapunov exponent**: `largest_lyapunov_exponent()` — Rosenstein (1993), Takens delay embedding + NN divergence slope.
  - **10-scorer registry**: `score(series, Scorer::*)` — Mi, Pearson, Spearman, Kendall, Distance, TransferEntropy, Gcmi, PermutationEntropy, SpectralEntropy, SpectralPredictability.
  - **AR(1) theoretical AMI**: `ar1_theoretical_ami(phi, max_lag)` — validation formula.
  - **Radix-2 complex FFT/IFFT**: `fft_complex::fft()` — used by surrogates, self-contained.
  - 36 new tests covering all primitives, lag curves, fingerprint (AR(1) + white noise), and LLE (sine + logistic map).

## [0.6.0] - 2026-04-09

### Added

- **Rolling-window features for `RegressionForecaster`** — target-derived features that participate correctly in recursive multi-step prediction.
  - New `RecursiveFeature` trait with `compute_fit(values, target_idx, out)` and `compute_predict(recent, out)`; distinct from `StructuralFeature` which is forward-filled with a constant. Recursive features are **recomputed at every horizon step** using the rolling history buffer (training tail + predictions emitted so far) — the same mechanism lag features already use.
  - New `RollingFeature { window, lag, kind }` struct + `RollingStatKind` enum with 9 variants: `Mean`, `Std`, `Var`, `Min`, `Max`, `Median`, `Sum`, `EwmMean { alpha }`, `EwmStd { alpha }`.
  - Builder shortcuts on `RegressionFeatures`: `.with_rolling_mean(w)`, `.with_rolling_std(w)`, `.with_rolling_var(w)`, `.with_rolling_min(w)`, `.with_rolling_max(w)`, `.with_rolling_median(w)`, `.with_rolling_sum(w)`, `.with_ewm_mean(w, α)`, `.with_ewm_std(w, α)`, plus general `.with_rolling(w, kind)` and `.with_rolling_lagged(w, lag, kind)`.
  - **Leakage guard**: `lag == 0` is rejected at construction time (would include the target in its own feature window). Default `lag = 1` is safe; larger lags are accepted.
  - Warmup handling: `lag_offset()` now takes `max(max_effective_lag, max_recursive_warmup)` so the first unusable rows are dropped correctly. `tail_values` grows to match.
  - 14 new tests covering fit correctness (hand-checked for Mean/Std/Var/Min/Max/Median/Sum), recursive predict semantics (anchor verification), leakage guards, end-to-end OLS fit+predict on a rolling-mean-generated pattern, combination with lag features, and cross-validation round-trip.

- **Sequential monitoring of forecast errors** (`monitor::` module) — online changepoint detection on residual streams to flag when a fitted forecaster has drifted.
  - Four CUSUM detectors: `PageCusum` (default, recommended), `PageCusum1`, `Cusum`, `Cusum1`. Two-sided and one-sided variants for both Page and original CUSUM.
  - Three error transformations: `Raw` (mean changes), `Squared` (variance changes), `Both` (default — runs both streams in parallel).
  - `SequentialDetector::fit()` for batch initialisation; `SequentialDetector::update()` for online streaming with constant-size state — bit-equivalent to a single full fit.
  - Baked-in critical value table: full 228-entry grid (4 detectors × 19 γ × 3 α) reproducible from a deterministic Monte Carlo simulator. Off-grid `(γ, α)` falls through to live simulation. Manual override via `CriticalValue::Fixed` and `with_sigma2()` for autocorrelated residuals.
  - `Forecaster` trait integration: `monitor_forecaster()` for cheap in-sample residuals, `monitor_forecaster_cv()` for unbiased rolling-origin CV residuals (calibrated nominal α).
  - JS/WASM bindings: `monitorForecastErrors()` + `updateForecastMonitor()` with full state round-trip via `serde-wasm-bindgen`.
  - Rust port of the R package [`changepoint.forecast`](https://github.com/grundy95/changepoint.forecast) by Thomas Grundy (Lancaster), MIT License. Asymptotic theory from [Fremdt (2014)](https://doi.org/10.1080/02331888.2014.921899).

## [0.5.7] - 2026-04-02

### Fixed

- **AutoARIMA seasonal detection**: add `seas_heuristic` matching statsforecast's `nsdiffs()` — STL decomposition + seasonal strength test (threshold 0.64). Series without detectable weekly pattern now fall back to non-seasonal ARIMA. Fixes M4 Daily accuracy: MAE 177→176 (+3.4% vs statsforecast, down from +4.3%). Excluding 50 outliers, anofox is 13.5% better than statsforecast.

### Added

- Full statsforecast validation: 35 models across 5 data types, 86% within MAD<1.0, 97% within MAD<5.0. See `docs/statsforecast_validation.md`.

## [0.5.6] - 2026-04-02 [YANKED]

### Note

v0.5.6 introduced an adaptive scoring window that regressed overall M4 accuracy (+12.4% gap vs +4.3% in v0.5.5). Use v0.5.7 instead.

## [0.5.5] - 2026-04-01

### Changed

- **AutoARIMA sequential 21x faster** (critical for DuckDB extension which can't use `parallel` feature)
  - Reduce default `max_cap_p`/`max_cap_q` from 2 to 1 (matches R's `auto.arima` default), halving SARIMA candidates
  - Sequential path: use cheap `score_order_static` for all candidates, full fit only for the winner
  - Approximate CSS scoring on last 500 observations for series >500 (matches R's `approximation=TRUE`)
  - Sort candidates simpler-first for faster early-winner detection
  - n=2358 p=7: 3.86s → 183ms sequential (no accuracy regression)

### Fixed

- Dead code cleanup: removed unused fields from DynamicTheta, KalmanFilter, regression

## [0.5.4] - 2026-04-01

### Added

- **Automatic changepoint penalty selection** (`Pelt::auto_detect()`)
  - CROPS (Changepoints for a Range of Penalties, Haynes et al. 2017): evaluates ~30 penalties automatically
  - Largest-gap elbow detection selects optimal penalty; BIC fallback when ambiguous
  - `Pelt::crops()` for manual exploration of the penalty landscape
  - Returns `AutoPeltResult` with selected penalty, best result, and all CROPS segmentations

- **Scalable hierarchical reconciliation** for large hierarchies (100k+ series)
  - `MinTraceVariance`: diagonal W from residual variances, sparse summing matrix. O(N + M²) memory.
  - `MinTraceStruct`: diagonal W from hierarchy structure (no residuals needed). O(N + M²) memory.
  - Both avoid the N×N covariance matrix that causes OOM in `MinTraceShrink` at N > 5k.

### Changed

- **ARIMA MLE 1.05-1.27x faster**: steady-state innovations detection skips redundant ln() calls after convergence; Jones (1980) parameter transform foundation added
- **AutoARIMA 49-158x faster than Python statsforecast** (validated on n=100-500, seasonal/nonseasonal)

## [0.5.3] - 2026-03-30

### Added

- **Global/Batch Forecasting** — shared-parameter models for processing many series simultaneously
  - `GlobalETS`: shared smoothing parameters (α, β, γ, φ) across N series, per-series initial states. 75-96x faster than N individual ETS fits for seasonal models.
  - `GlobalAutoETS`: automatic model selection across N series — each candidate spec fitted once globally, best spec selected per series by NLL. 28-32x faster than N individual AutoETS fits.
  - `GlobalCroston`: shared α across N intermittent demand series. Classic and SBA variants. 3-6x faster.
  - `GlobalTheta`: shared α for Standard Theta Method across N series.
  - `batch::auto_ets()`, `batch::ets()`, `batch::mfles()`: convenience functions for parallel batch processing
  - `STL::decompose_batch()`: batch decomposition with scratch buffer reuse and rayon parallelism

- **Parallel grouped cross-validation**: `grouped_cross_validate` now processes groups concurrently via rayon (was sequential)

### Changed

- **STL decomposition 2.0-2.5x faster**: running-sum moving average (O(n) vs O(n×window)) and precomputed tricube kernel weights in LOESS smoothing
- **MSTL decomposition faster**: pre-creates STL instances outside iteration loop to reuse scratch buffers (benefits from underlying STL speedup)
- **MFLES boosting 90% fewer allocations**: cache X'X Cholesky factor (Fourier design matrix is constant across rounds), pre-allocated residual and temp buffers, `calc_mse_sum()` avoids materializing temp Vecs

## [0.5.2] - 2026-03-30

### Added

- **AutoETS Model Pools** — based on Petropoulos et al. (2023) "Wielding Occam's razor" ([arXiv:2102.13209](https://arxiv.org/abs/2102.13209))
  - `ModelPool` enum: `Complete` (19 models, default), `NoMultiplicativeTrend` (15), `DampedTrendOnly` (12), `MatchErrorSeasonal` (16), `Reduced` (8)
  - `AutoETSConfig::with_model_pool(ModelPool::Reduced)` — 2x faster on M5 dataset (30,490 series) with identical accuracy
  - All models remain available for explicit fitting; pool only controls AutoETS candidate generation

- **AutoARIMA Accuracy Improvements** — model order selection now matches Python statsforecast (3/4 test cases)
  - AICc scoring (corrected AIC) replacing plain AIC for better small-sample penalization
  - Drift parameter for ARIMA(0,1,0) matching R/Python `include.drift` convention
  - Expanded stepwise grid candidates to include p=3 orders
  - Greedy first-improvement stepwise search with 94-model cap (matching Python)
  - Fixed d-selection to use KPSS-suggested value directly (no neighbor search)

- **Innovations MLE for ARIMA** — exact maximum likelihood via ported statsmodels Cython
  - `arma_innovations_algo_fast`: O(n·m²) innovations recursion
  - `arma_acovf`: Brockwell-Davis linear system (eq 3.3.8)
  - `arma_transformed_acovf_fast`, `arma_innovations_filter`, `lfilter`
  - Three-phase optimization: L-BFGS CSS → NM CSS → NM MLE
  - Hannan-Rissanen initialization for near-optimal starting values
  - Parameters match Python to 5 decimal places on validation data

- **Nelder-Mead Stagnation Early Stopping**
  - `NelderMeadConfig::stagnation_window` — terminate when best value hasn't improved for N iterations
  - Applied to ETS seasonal optimization (25-48% faster AutoETS)

- **M5 Benchmark** (`docs/m5_ets_benchmark.md`)
  - Full M5 dataset: 30,490 series, 28-day holdout, weekly seasonality
  - Complete vs Reduced pool comparison: RMSE 1.4332 vs 1.4331 (0.007% diff), 2x speedup

### Changed

- TBATS/AutoTBATS 2.2-2.5x faster via trig precomputation, incremental Fourier, scratch buffers
- AutoETS sorts candidates non-seasonal-first for earlier convergence
- `check_stationarity()` relaxed for p≥3: proper AR polynomial root conditions replace overly strict `sum_abs < 1.5`
- Variance floor (1e-15) in `calculate_fitted` prevents -inf AIC on perfect-fit series

### Fixed

- AutoARIMA `score_order` AIC formula inconsistency between (0,0) and non-(0,0) cases
- AutoARIMA zero-variance bug: `score_order` returned None for constant series after differencing
- ARIMA(0,1,0) missing drift: forecasts were flat instead of continuing trend
- `calculate_fitted` -inf AIC on zero-variance residuals (e.g., linear data after differencing)

## [0.5.1] - 2026-03-28

### Added

- **Cycle Extraction Filters** (`seasonality` module)
  - `cf_filter()` — Christiano-Fitzgerald asymmetric band-pass filter (preserves full series length)
  - `bk_filter()` — Baxter-King symmetric band-pass filter (loses 2k edge observations)
  - `hamilton_filter()` — Hamilton (2018) regression-based trend-cycle decomposition with QR solver
  - `hamilton_quarterly()` / `hamilton_monthly()` / `hamilton_annual()` convenience constructors
  - `CycleDecomposition` and `HamiltonDecomposition` result types

- **Fractional Differencing** (`arima/diff` module)
  - `fractional_difference(series, d, threshold)` — binomial `(1-B)^d` expansion
  - `fractional_weights(d, max_len, threshold)` — weight computation
  - `find_min_fractional_d(series, significance, threshold)` — binary search for minimum d achieving ADF stationarity
  - `RegressionFeatures::fractional_differencing(d)` — builder method
  - Reference: Lopez de Prado, *Advances in Financial Machine Learning* (2018)

- **Multi-Quantile Forecasts**
  - `BootstrapPredictor::predict_quantiles()` — multiple quantile levels from bootstrap simulation
  - `ConformalPredictor::predict_quantiles()` — multiple quantile levels from conformal scores

### Changed

- Updated `anofox-regression` dependency from 0.5.0 to 0.5.3
- Added `codecov.yml` with project threshold 1% and patch target 70%

### Fixed

- `RegressionForecaster`: exogenous regressors preserved during differencing (was silently dropped)
- `KalmanForecaster.smooth()` uses configured SSM (was hardcoding local_level defaults)

## [0.5.0] - 2026-03-26

### Added

- **WASM/JS Parity** — major expansion of JS/WASM bindings to match Rust core:
  - `FeatureGenerator` with cyclical sin/cos, binary indicators, advanced calendar features
  - `generateFutureTimestamps(lastMs, frequency, horizon)` standalone function
  - `TimeSeries.futureTimestamps(horizon)` auto-inferred
  - `JsBootstrapPredictor` with `predictIntervals()` and `predictQuantiles()`
  - `JsConformalPredictor.predictQuantiles()` for multi-quantile conformal forecasts
  - `JsConformalPredictor.fitPerStep()` → `JsPerStepConformalResult`
  - `KalmanForecaster.custom(F, H, Q, R)` for user-defined state-space models
  - `KalmanForecaster.logLikelihood(series)` for model comparison
  - `KalmanForecaster.nState` / `nObs` getters

### Fixed

- `KalmanForecaster.smooth()` was hardcoding `local_level(1.0, 0.1)` instead of using the user-configured state-space model

## [0.4.9] - 2026-03-26

### Added

- `BootstrapPredictor::predict_quantiles()` — multi-quantile forecasts from bootstrap simulation paths (e.g., 10th/25th/50th/75th/90th percentiles)
- `ConformalPredictor::predict_quantiles()` — multi-quantile forecasts from conformal nonconformity scores (symmetric, median = point forecast)

### Fixed

- `RegressionForecaster`: exogenous regressors were silently dropped when differencing was active — now preserved via `TrimOnly` policy (regressors trimmed to match differenced length)

## [0.4.8] - 2026-03-24

### Added

- **Per-Horizon-Step Conformal Prediction** (`postprocess` module)
  - `ConformalPredictor::fit_per_step()` — separate interval widths per forecast step
  - `PerStepConformalResult` with `half_widths()`, `predict()`, `predict_intervals()`
  - Works with Split, CrossVal, and Jackknife+ methods; falls back to pooled quantile when < 2 residuals per step

- **Bootstrap Prediction Intervals** (`postprocess` module)
  - `BootstrapPredictor` — model-agnostic residual resampling with cumulative error path simulation
  - IID and block bootstrap variants
  - Intervals grow with horizon (uncertainty accumulates)

- **Calendar Feature Engineering** (`features` module)
  - `TimeComponent` cyclical sin/cos encoding: Month, Quarter, DayOfWeek, Hour, DayOfYear, etc.
  - `BinaryIndicator`: MonthStart, MonthEnd, QuarterStart, QuarterEnd, YearStart, YearEnd, Weekend
  - `AdvancedFeature`: LeapYear, DaysInMonth

- **Future Timestamp Generation** (`core` module)
  - `generate_future_timestamps()` — calendar-aware (Jan 31 + 1mo = Feb 28/29, not Mar 2)
  - `TimeSeries::future_timestamps(horizon)` — auto-infers frequency from data
  - Preserves original day across monthly/yearly stepping (no month-end drift)

- **Auto-Lag Selection for RegressionForecaster** (`models::regression`)
  - `.auto_lags(max_lag)` — select best lag order by BIC
  - `.auto_lags_with(max_lag, LagSelectionCriterion::Aic)` — select by AIC
  - `LagSelectionCriterion` enum: `Bic`, `Aic`
  - Re-selects on each `fit()` call

- **Differencing for RegressionForecaster** (`models::regression`)
  - `.differencing(d)` — regular differencing, auto-integrates after predict
  - `.seasonal_differencing(D, period)` — chainable for multi-seasonal series
  - `seasonal_integrate()` in `arima/diff.rs` — inverse of `seasonal_difference()`

- **CvFoldGenerator Rework** (`utils::cross_validation`)
  - Backward-anchored fold placement (last fold always covers series end)
  - `n_folds` driven — caps how many folds to take
  - `min_initial_window` as constraint (not driver) — drops early folds with insufficient training
  - `step_size` configurable (default = horizon for non-overlapping test sets)
  - `ConstraintViolation::Error` (default) or `ReduceFolds`

- **ModelSpec Flexibility** (`models::traits`)
  - `ModelSpec.name` changed from `&'static str` to `String` (accepts `impl Into<String>`)
  - `model_type` field for grouping multi-variant models
  - `ModelSpec::with_type()` constructor, `ModelRegistry::by_type()` query

- **Period Validation** — seasonal models reject `period < 2` with `InvalidParameter` error (SeasonalNaive, HoltWinters, SeasonalES, SeasonalWindowAverage, MSTLForecaster, TBATS, DummySeasonality)

### Changed

- **AutoForecast**: removed `SelectionStrategy::InSampleMSE` — always uses cross-validation
- **CvFoldGenerator**: `initial_window` renamed to `min_initial_window`
- Consolidated 4 duplicate `compute_median()` → shared `utils::stats::median()`
- Consolidated 3 duplicate aggregate helpers → shared `utils::stats::aggregate()`
- Removed dead `utils::stats::autocorrelation()` (0 callers)
- Removed batch processing module (`fit_many`, `predict_many`, `fit_predict_many`, `fit_registry`)

### Fixed

- `generate_future_timestamps`: month-end drift (Jan 31→Feb 28→Mar 28) fixed to preserve original day (Jan 31→Feb 28→Mar 31)
- `BootstrapPredictor`: intervals now grow with horizon via cumulative error paths (was flat)
- `auto_lags`: config preserved across re-fit calls (was destroyed after first fit)
- `TimeSeries::future_timestamps`: monthly data now uses `Frequency::Months(1)` instead of `Duration::days(30)`

## [0.4.7] - 2026-03-19

### Added

- **Recency-Aware Trend Components** (`seasonality` module)
  - `Recency` enum — fit on recent data only (`Window(N)`, `Fraction(0.3)`, `Full`, `Auto`) for trend-aware forecasting
  - `Recency::Auto` — PELT-based changepoint detection to automatically determine fitting window
  - `PolynomialTrend` — polynomial trend (degree 1-3) with Vandermonde + Cholesky solve, recency-aware
  - `ExponentialTrend` — log-linear exponential growth/decay trend estimation, recency-aware
  - `LogisticTrend` — logistic S-curve fitting with auto or fixed capacity, recency-aware
  - `TheilSenTrend` — robust Theil-Sen median-of-pairwise-slopes estimator, recency-aware, subsampled for large data
  - `AutoTrend` — automatic selection of best trend component via AICc/BIC/holdout (enum dispatch)
  - `AutoSeasonal` — automatic selection of best seasonal component via AICc/BIC
  - `n_params()` method added to `TrendComponent` and `SeasonalComponent` traits (for IC-based selection)
  - `with_recency()` builder methods on `PiecewiseLinearTrend` and `HodrickPrescottFilter`
  - `trend_components` example demonstrating all components and AutoTrend selection

- **Changepoint-Aware Baseline Models**
  - `SMA::with_changepoint()` — constrains window to post-changepoint data
  - `RandomWalkWithDrift::with_changepoint()` — constrains drift estimation to post-changepoint data

- **Regression Forecaster** (`models::regression` module, requires `postprocess` feature)
  - `RegressionForecaster` — wraps `anofox-regression` backends behind the `Forecaster` trait
  - `RegressionBackend` enum — 11 backends: OLS (default), Ridge, ElasticNet, Quantile, WLS, RLS, Tweedie, Poisson, BLS, NNLS, Dynamic
  - `RegressionFeatures` — configurable feature engineering: trend types, seasonal specs, AR lags, structural features, exogenous regressors
  - Convenience constructors: `linear_trend()`, `ar(lags)`, `trend_ar(lags)`, `ols(features)`, `ridge()`, `elastic_net()`, `quantile()`, `wls_decay()`, `wls()`, `rls()`, `tweedie()`, `poisson()`, `bls()`, `nnls()`, `dynamic()`, `dynamic_smoothed()`
  - `WeightStrategy` enum for WLS — `ExponentialDecay(f64)` or `Custom(Vec<f64>)`
  - Full `Forecaster` trait implementation: `fit`, `predict`, `predict_with_exog`, `fitted_values`, `residuals`, `supports_exog`
  - Recursive multi-step prediction for AR models (feeds predictions back as lag features)
  - Access to fitted statistics (R², coefficients) via `r_squared()`, `fitted_result()`
  - Generalized fitted model storage via `Box<dyn FittedRegressor>` for backend-agnostic prediction
  - `InformationCriterion` re-exported from `anofox-regression` for Dynamic backend configuration
  - Compatible with `ModelRegistry`, pipelines, ensembles, and cross-validation

- **Feature Safety Classification** (`models::regression` module)
  - `FeatureSafety` enum — classifies features by cross-validation leakage risk: `Deterministic`, `DataDependent`, `Structural`, `External`
  - `classify_features()` method on `RegressionFeatures` — returns per-column safety classification
  - Trend types classified as `DataDependent`, Fourier terms as `Deterministic`, structural features as `Structural`

- **Structural Features** (`models::regression` module)
  - `StructuralFeature` trait — general interface for features that are forward-filled during prediction
  - `ChangepointFeature` — encodes detected changepoints as regression features
  - `ChangepointEncoding` enum — `StepFunctions` (k binary columns), `RegimeIndex` (1 ordinal column), `CumulativeCount` (1 additive column)
  - Forward-fill prediction: model continues in last known regime during forecast period
  - Builder methods: `with_structural()`, `with_changepoint_steps()`, `with_changepoints()`

- **Binned Conformal Prediction** (`postprocess` module)
  - `BinnedConformalPredictor` — heteroscedastic prediction intervals that bin calibration residuals by predicted magnitude
  - Quantile-based bin edges with per-bin conformal quantiles — wider intervals where forecast uncertainty is larger
  - Graceful fallback: bins with < 3 residuals merge with neighbors; insufficient data falls back to single global quantile
  - `BinnedConformalResult` — bin edges, per-bin quantiles, global fallback quantile, coverage level

- **Pre-Regression MSTL Decomposition** (`seasonality` + `models` modules)
  - `MSTL::decompose_with_regressors()` — regress out exogenous effects (OLS: y ~ X) before STL decomposition, preventing regressors correlated with trend/seasonality from distorting the decomposition
  - `MSTLResult` now stores `regressor_coefficients` (OLS result) and `regressor_effect` (X*β) for reconstruction
  - `MSTLForecaster` automatically uses pre-regression when `TimeSeries` has calendar regressors
  - Full `Forecaster` exog support: `supports_exog()`, `has_exog()`, `exog_names()`, `predict_with_exog()`
  - `predict()` guards against missing exog when model was fit with regressors

- **Forecasting Metrics** (`utils::metrics` module)
  - `rmsse()` — Root Mean Squared Scaled Error, scaled by in-sample naive forecast MSE (per-series building block of WRMSSE)
  - `wrmsse()` — Weighted RMSSE, the M5 competition primary metric for aggregating across series by importance weights

- **Exogenous Coefficient Extraction** (`models::traits`)
  - `Forecaster::exog_coefficients()` — inspect OLS pre-regression coefficients (intercept, betas, regressor names) on any fitted model with exogenous regressors
  - Implemented across all exog-supporting models: ARIMA, SARIMA, AutoARIMA, ETS, AutoETS, Theta, AutoTheta, OptimizedTheta, DynamicTheta, Naive, MFLES, MSTL, Pipeline

- **Transform Pipeline** (`transform::pipeline` module)
  - `Pipeline` — composable transform chains around any `Forecaster`, itself implements `Forecaster`
  - `PipelineBuilder` — fluent API: `Pipeline::builder().transform(BoxCoxTransform::auto()).transform(DifferenceTransform::new(1)).model(Box::new(Naive::new())).build()`
  - `Transform` trait with concrete implementations: `DifferenceTransform`, `SeasonalDifferenceTransform`, `BoxCoxTransform`, `ScaleTransform`, `LogTransform`

- **Deterministic Feature Generator** (`features::generator` module)
  - `FeatureGenerator` — generate regressors from timestamps: `fourier()`, `day_of_week()`, `month_of_year()`, `quarter()`, `holiday()`
  - `add_to()` — attach generated features to `TimeSeries` as named regressors

- **Composable Seasonality & Trend Components** (`seasonality` module)
  - `SeasonalComponent` / `TrendComponent` traits — dual-purpose: standalone (fit/predict) and feature extraction
  - `DummySeasonality` — one-hot (dummy variable) seasonal encoding for arbitrary seasonal shapes
  - `SeasonalDifference` — standalone seasonal differencing transform with inverse, strength, and variance-reduction features
  - `HodrickPrescottFilter` — smooth trend extraction with cycle decomposition (quarterly/monthly/annual presets)
  - `PiecewiseLinearTrend` — PELT-based changepoint detection with per-segment linear regression
  - Standalone feature functions: `dummy_seasonal_strength`, `dummy_seasonal_amplitude`, `seasonal_diff_strength`, `seasonal_diff_variance_reduction`, `hp_trend_strength`, `hp_cycle_variance_ratio`, `piecewise_n_segments`, `piecewise_trend_features`

- **Comprehensive Examples** (45 examples with `.md` documentation)
  - 10 new examples: `regression` (11 backends), `hierarchy`, `var`, `kalman`, `constraints`, `explainability`, `aid`, `bootstrap`, `temporal_aggregation`, `serialization`
  - `.md` companion documentation for all examples with section descriptions, key types, and run commands
  - `examples/README.md` — categorized index of all examples

### Removed

- **Orchestration module extracted to [`anofox-orchestration`](https://github.com/sipemu/anofox-orchestration)** (private repo)
  - Moved: `DataProfile`, `PipelineBuilder`, `Pipeline`, `PipelineConfig`, `PipelineResult`, `PipelineReport`, `PipelineStore`, `DecisionLog`, `FallbackChain`, `HorizonAnalysis`, `SelectionConfidence`, `ModelConfidenceSet`, `QualityFloor`, `MetricStrategy`, `EnsembleMode`, `PreprocessMode`, `TrendIntegration`, `ChangepointMode`, `DriftConfig`/`DriftReport`, `ExploreBuilder`, `BacktestBuilder`, structured tool functions
  - Moved: `partition_by_structural_break()` helper
  - Moved examples: `orchestration`, `explore`, `regression_exog_changepoints`
  - `anofox-forecast` remains the open-source foundation; `anofox-orchestration` depends on it via git

- **Batch processing module removed** (`models::batch`)
  - Removed `fit_many()`, `predict_many()`, `fit_predict_many()`, `fit_registry()` — batch multi-series operations belong in the orchestration layer
  - `ModelRegistry` and `ModelSpec` remain (used by `compare_models`, `compare_registry`, cross-validation)

### Internal

- Consolidated 4 duplicate `compute_median()` implementations → shared `utils::stats::median()`
- Consolidated 3 duplicate aggregate helpers → shared `utils::stats::aggregate()`
- Removed dead `utils::stats::autocorrelation()` (0 callers)

### Changed

- **ETS/AutoETS Performance Optimizations** (accuracy-preserving, all 2383 tests pass)
  - Eliminate redundant `initialize_state()` calls in non-seasonal Nelder-Mead closures — pass pre-computed heuristic states instead of recomputing O(n) per evaluation
  - Two-stage seasonal optimization: replace 6× expensive joint optimizations (16-18 dim) with 6× cheap smoothing-only explorations (2-4 dim, 500 iter) + 1× joint refinement from winner
  - AutoETS candidate pruning: require 3× period (was 2×) for seasonal models; ANOVA F-test skips seasonal candidates when no seasonal signal detected (F < 1.0)
  - Multi-start early-exit: skip starting points whose initial likelihood exceeds 3× the best found so far
  - `NelderMeadConfig` derives `Copy` (all fields are `Copy` types), eliminating unnecessary `.clone()` calls across ETS, seasonal ES, GARCH, and Theta models

### Fixed

- ETS: replaced unsafe `unwrap()` calls with heuristic fallbacks for edge cases (constant data, zero variance)
- `cross_validate_all` / `fit_all_and_compare`: reduced cognitive complexity with early returns and helper closures
- `TimeSeries::missing_mask()`: fixed panic on empty-dimension series
- Conformal prediction: added SAFETY comments documenting invariants for `expect()` calls
- `RegressionForecaster::predict_with_intervals`: fixed missing `components` argument in `build_future_matrix` call
- `TimeSeries::slice()`: fixed calendar regressors not being sliced — caused regressor/value misalignment during cross-validation with exogenous features
- `cross_validate`: fixed `evaluate_fold` to call `predict_with_exog()` when model has exogenous regressors (previously always called `predict()`, which errors on exog models)

## [0.4.6] - 2026-03-14

### Added

- **Orchestration Module** (`orchestration` module)
  - `DataProfile` — automated data profiling (stationarity, trend, seasonality, quality score, ACF statistics)
  - `PipelineBuilder` — declarative pipeline construction with model selection, cross-validation, fallback chains, and forecast constraints
  - `Pipeline` / `PipelineConfig` — end-to-end pipeline execution and replay from saved configuration
  - `PipelineResult` — forecast with full diagnostics (profile, decision log, confidence metrics, horizon analysis)
  - `DecisionLog` — structured audit trail of pipeline decisions with categories, outcomes, and timing
  - `FallbackChain` — ordered model failover with automatic recovery
  - `HorizonAnalysis` — per-step-ahead error decomposition (RMSE, MAE, bias per horizon step)
  - `ExecutionMetadata` / `ExecutionTimer` — fit/predict timing and convergence tracking
  - `SelectionConfidence` — Diebold-Mariano pairwise forecast accuracy test (via `anofox-statistics`)
  - `ModelConfidenceSet` — Hansen-Lunde-Nason (2011) model confidence set procedure (bootstrap-based)
  - `QualityFloor` — Hansen (2005) Superior Predictive Ability test vs benchmark model
  - Orchestration prelude (`orchestration::prelude::*`) for convenient imports
  - `PreprocessMode` — automatic preprocessing pipeline (Box-Cox for skewed data, outlier replacement for low-quality data)
  - `PreprocessSteps` — explicit control: `boxcox`, `outlier_treatment`, `outlier_window`
  - `MetricStrategy` — data-aware multi-metric model selection
    - `Auto`: intermittent → MAE+SMAPE, non-negative → MAE+WAPE, general → MAE+SMAPE+MDA
    - `Single(Metric)` / `Composite(Vec<(Metric, f64)>)` for custom weighting
    - MDA (higher-is-better) automatically inverted in composite scores
  - `EnsembleMode` — configurable ensemble construction
    - `Auto`: ensemble models from MCS when > 1 model included
    - `Fixed(CombinationMethod)`: always ensemble with specified method
    - `None`: single best model (default)
  - `PipelineReport` — multi-section structured report from `PipelineResult`
    - Sections: Summary, Data Profile, Preprocessing, Model Selection, Ensemble, Forecast, Horizon Analysis, Decision Log, Execution Metadata
    - `Display` impl with column-aligned table formatting
  - `PipelineStore` trait — abstract storage backend decoupled from serde
    - `Value` IR enum (Null, Bool, Int, Float, String, List, Map) for backend-agnostic serialization
    - `Storable` trait for converting orchestration types to/from `Value`
    - `InMemoryStore` — thread-safe in-memory implementation for testing
    - `RecordKind`: Profile, Config, Result, DecisionLog, HorizonAnalysis, Report
  - Structured tool functions for MCP / agent integration (`orchestration::tools`)
    - `profile_data()` — profile a time series with typed I/O
    - `select_models()` — heuristic model recommendation from data profile
    - `run_pipeline()` — end-to-end pipeline execution
    - `explain_result()` — human-readable explanation (Brief/Normal/Detailed)
  - `PipelineBuilder::preprocess()`, `.metric()`, `.ensemble()` builder methods

- **WASM/npm Orchestration Bindings**
  - `JsDataProfile` — full data profiling with 30+ property getters, `toJSON()`, `fromSeries()`, `fromValues()`
  - `JsPipelineBuilder` — fluent builder: `profile()`, `preprocess()`, `metric()`, `ensemble()`, `addModel()`, `execute()`
  - `JsPipelineResult` — forecast, model name, decision log, quality floor, MCS, ensemble weights, metric scores
  - `JsPipelineReport` — `title`, `sectionCount`, `toString()`, `toJSON()` with typed sections
  - `selectModels(profile, availableModels?)` — model recommendation tool function
  - `explainResult(result, verbosity)` — human-readable explanation (brief/normal/detailed)

- **Forecasting Metrics**
  - `bias()` — signed forecast bias (mean error)
  - `periods_in_stock()` — Periods-in-Stock metric for inventory forecasting

- **Automatic Identification of Demand (AID)**
  - `AidAnalyzer` — wraps `anofox-regression`'s AID classifier with `&[f64]`/`TimeSeries` input
  - Summary statistics: demand type (Regular/Intermittent), best-fitting distribution (Poisson, NegBin, Normal, Gamma, LogNormal, etc.), fitted parameters, zero proportion
  - Per-observation anomaly features: `AidFeatures` with `Vec<AidAnomalyLabel>` (Stockout, NewProduct, ObsoleteProduct, HighOutlier, LowOutlier) matching input length
  - `AidResult::summary()` for aggregate statistics, `AidResult::features()` for per-observation labels
  - Builder API: `AidAnalyzer::new().intermittent_threshold(0.3).detect_anomalies(true).analyze(&data)`
  - Gated behind `postprocess` feature (enabled by default)

## [0.4.5] - 2026-03-13

### Added

- **Parallel Processing & AutoForecast**
  - `compare_models()` / `compare_registry()` — parallel model comparison
  - `AutoForecast` candidate model fits run in parallel when `parallel` feature enabled
  - Bootstrap sampling uses `par_iter` when `parallel` enabled

- **Streaming Cross-Validation**
  - `StreamingCVAggregator` — online metric aggregation using Welford's algorithm
  - `cross_validate_early_stop()` — CV with convergence-based early stopping
  - Eliminates need to store all fold results for aggregation

- **Builder Patterns**
  - `Pelt::new(CostFunction::L2).min_size(5).penalty(5.0).detect(&data)` — PELT builder
  - `StlBuilder::new(period).seasonal_window(7).robust(true).decompose(&data)` — STL builder

- **SIMD Correlation & Autocorrelation**
  - `simd::correlation()` — SIMD-accelerated Pearson correlation
  - `simd::autocorrelation()` — SIMD-accelerated autocorrelation at a given lag

- **Binary Serialization** (requires `serde` feature)
  - `to_bincode()` / `from_bincode()` — compact binary serialization via bincode
  - `save_to_bincode()` / `load_from_bincode()` — file persistence

- **Convenience Methods**
  - `Forecaster::fit_predict()` — fit and predict in a single call
  - `Forecaster::fit_predict_with_intervals()` — fit and predict with confidence intervals

- **Specific Error Variants**
  - `ForecastError::ConvergenceFailure` — optimizer/model convergence failures
  - `ForecastError::SingularMatrix` — linear algebra singularity errors
  - `ForecastError::SerializationError` — serialization/deserialization errors

- **WASM/npm Enhancements**
  - `CalendarAnnotations` — holidays, named regressors, and JSON serialization in JS/TS
  - `TimeSeries.setCalendar()` / `hasCalendar()` / `clearCalendar()` — calendar integration
  - Complete TypeScript type definitions (`types.d.ts`) for all 35 exported classes and 21 functions
  - `package.json` with `"types"` field for TypeScript support

- **CI/CD Improvements**
  - `cargo audit` step for security advisory checks
  - `cargo deny` step for license and supply-chain compliance
  - `deny.toml` configuration for allowed licenses

- **Benchmarks**
  - `ensemble_benchmark` — AutoEnsemble fit/predict benchmarks
  - `cv_benchmark` — cross-validation benchmarks with varying folds/horizons/models

- **Documentation**
  - [Model Selection Guide](docs/model_selection_guide.md) — decision flowchart, model families, common patterns

- **Test Coverage**
  - Postprocess tests: +42 IDR tests, +22 Normal tests, +316 lines backtest, +211 lines QRA
  - Streaming CV tests: 8 new tests for aggregator and early stopping
  - SIMD tests: 18 new tests for correlation and autocorrelation

- **Missing Value Imputation Toolbox**
  - `MissingValuePolicy::BackwardFill` — next-observation-carried-backward
  - `MissingValuePolicy::FillMean` — fill with mean of finite values
  - `MissingValuePolicy::FillMedian` — fill with median of finite values
  - `MissingValuePolicy::Interpolate` — linear interpolation via policy enum
  - `TimeSeries::missing_mask()` — boolean mask of NaN/Inf positions (primary dimension)
  - `TimeSeries::missing_count()` — per-dimension count of missing values
  - `TimeSeries::imputed_forward_backward()` — forward-fill then backward-fill (handles leading + trailing NaN)
  - `TimeSeries::imputed_moving_average(window)` — centered moving average imputation with multi-pass for adjacent gaps
  - `TimeSeries::imputed_seasonal(period)` — seasonal median imputation using same-position values across cycles
  - `TimeSeries::with_imputed_regressors(policy)` — apply imputation policy to regressor vectors independently
  - `nan_mean()` / `nan_median()` — NaN-safe statistics helpers in `utils::stats`

- **OLS NaN/Inf Validation**
  - `ols_fit()` now validates `y` for NaN/Inf (returns `ForecastError::MissingValues`)
  - `ols_fit()` now validates each regressor for NaN/Inf (returns `ForecastError::InvalidParameter`)

- **Hierarchical Forecasting** (`hierarchy` module)
  - `HierarchyTree` — define parent→children structure for grouped forecasts
  - `ReconciliationMethod::BottomUp` — aggregate leaf forecasts upward
  - `ReconciliationMethod::TopDown` — disaggregate top-level using historical proportions
  - `ReconciliationMethod::MinTraceOls` — optimal combination via MinT OLS projection
  - 15 tests covering tree construction, all methods, coherence, and error handling

- **Prophet-style Fourier Seasonality** (`seasonality::fourier` module)
  - `FourierSeasonality` — flexible seasonal modeling using Fourier basis functions
  - `fourier_terms()` — generate sin/cos basis vectors for arbitrary period and order
  - Preset constructors: `daily()`, `weekly()`, `yearly()`
  - Fit via normal equations with Cholesky decomposition (no external dependencies)
  - 25 tests covering recovery, orthogonality, periodicity, edge cases

- **Core Type Improvements**
  - `Serialize`/`Deserialize` for `Forecast`, `TimeSeries`, `CalendarAnnotations` (behind `serde` feature)
  - `Display` impls for `Forecast` and `TimeSeries` with preview summaries
  - `PartialEq` for `Forecast` (epsilon-based) and `CalendarAnnotations`

- **WASM Model Parity**
  - Added JS/WASM bindings: `HoltWintersForecaster`, `SESForecaster`, `CrostonForecaster`, `ADIDAForecaster`, `IMAPAForecaster`, `TSBForecaster`, `GARCHForecaster`
  - TypeScript definitions for all new forecaster classes

- **Validation & Stationarity Tests**
  - 14 new residual diagnostic tests (edge cases, NaN, constant, short series)
  - 9 new stationarity test cases (ADF, KPSS, edge cases)

- **Intermittent Model & GARCH Edge Case Tests**
  - Croston: 6 new tests (all zeros, single demand, negative values, very long gaps, single observation)
  - ADIDA: 6 new tests (same pattern)
  - IMAPA: 6 new tests (same pattern)
  - TSB: 5 new tests (same pattern)
  - GARCH: 10 new tests (constant data, extreme volatility, trending data, NaN handling, short series)

- **VAR (Vector Autoregression)** (`models::var` module)
  - `VAR::new(order)` — VAR(p) model for multivariate time series
  - Equation-by-equation OLS estimation
  - Multi-step forecasting across all variables
  - `granger_causality_test(cause, effect)` — F-statistic for Granger causality
  - 18 tests covering coefficient recovery, dimensions, edge cases

- **Kalman Filter Framework** (`models::kalman` module)
  - `KalmanFilter` — forward filtering and Rauch-Tung-Striebel smoothing
  - `StateSpaceModel` — linear Gaussian state-space specification
  - Convenience constructors: `local_level()`, `local_linear_trend()`
  - `filter()`, `smooth()`, `predict()`, `log_likelihood()` methods
  - Internal dense matrix algebra (no external dependencies)
  - 14 tests covering filtering, smoothing, prediction, edge cases

- **Builder Patterns for Models**
  - `GARCH::builder().p(1).q(1).max_iterations(500).build()`
  - `MFLES::builder().seasonal_period(12).num_rounds(5).learning_rate(0.1).build()`
  - `AutoForecast::builder().seasonal_period(12).include_arima(true).build()`

- **Rolling/Expanding Window Forecast**
  - `rolling_forecast()` — walk-forward evaluation with per-window metrics
  - `RollingForecastConfig` — builder for initial train size, horizon, step size, expanding/rolling mode
  - `RollingForecastResult` — per-window predictions, actuals, and aggregated metrics
  - Parallel window evaluation when `parallel` feature enabled

- **Ensemble Prediction Interval Combination**
  - Widest-envelope interval combination: takes min of lower bounds, max of upper bounds
  - `predict_with_intervals()` now produces meaningful combined intervals

- **STL Buffer Caching**
  - `StlScratch` — pre-allocated scratch buffers for zero-allocation repeated decompositions
  - `STL::decompose_with_scratch()` — decompose with reusable buffers
  - `StlBuilder::decompose_reuse()` — amortized allocation across repeated calls

- **TimeSeries Convenience Methods**
  - `seasonal_strength(period)` — seasonal strength via STL decomposition (0 to 1)
  - `trend_strength(period)` — trend strength via STL decomposition (0 to 1)
  - `with_outliers_replaced(config, window)` — replace outliers with local median
  - `to_json()` / `from_json()` — JSON serialization (requires `serde` feature)
  - `Forecast::to_json()` / `Forecast::from_json()` — forecast serialization

- **WASM PostProcessor Bindings**
  - `JsConformalPredictor`, `JsNormalPredictor`, `JsHistoricalSimulator` — probabilistic intervals in JS
  - `JsPostProcessor` — unified API with `conformal()`, `normal()`, `historicalSim()`
  - `JsBacktestConfig` / `JsBacktestResult` — backtesting support
  - `JsPredictionIntervals` — coverage, widths, midpoints, empirical coverage
  - `MFLESForecaster.predictWithIntervals()` — added to WASM bindings
  - TypeScript definitions for all postprocess types

- **Persistence Module Tests**
  - 27 tests (from 6): JSON/bincode round-trips, file I/O, error cases, helper modules

- **Advanced Forecasting Metrics**
  - `wape()` — Weighted Absolute Percentage Error
  - `mda()` — Mean Directional Accuracy (direction-of-change)
  - `theils_u1()` / `theils_u2()` — Theil's U statistics (absolute and relative to naive)
  - `msis()` — Mean Scaled Interval Score for probabilistic forecast evaluation
  - `coverage()` — Empirical coverage rate for prediction intervals
  - `skill_score()` — Relative improvement over a baseline model
  - `ForecastMetrics::compute()` — all 10 metrics (MAE, MSE, RMSE, MAPE, SMAPE, MASE, WAPE, MDA, U1, U2) in one call

- **Model Warm-Starting**
  - `ETS::with_initial_states(spec, period, level, trend, seasonal_values)` — pre-set ETS states
  - `SES::with_alpha(alpha, level)` — pre-fitted SES (predict without fit)
  - `ARIMA::with_coefficients(p, d, q, ar, ma, intercept)` — pre-fitted ARIMA coefficients
  - `Theta::with_theta_value(theta, alpha, level, b)` — pre-fitted Theta state
  - `Forecaster::fitted_params()` — extract `FittedParams` from any fitted model

- **Forecast Constraints**
  - `ForecastConstraint` enum: `NonNegative`, `LowerBound`, `UpperBound`, `Bounds`, `IntegerRound`, `Custom`
  - `ConstrainedForecast::apply()` — apply constraints to point forecasts and intervals
  - `Forecast::non_negative()`, `.clamp(lo, hi)`, `.round_to_integer()` convenience methods

- **Forecast Combination Convenience**
  - `fit_all_and_compare()` — fit all models in registry, rank by holdout MAE/RMSE/MAPE
  - `cross_validate_all()` — cross-validate all registry models with aggregated metrics
  - `ensemble_best_k()` — auto-select top-k models by performance into an ensemble
  - `ModelComparison` / `CVComparison` with `Display` formatted tables

- **STL Convenience Functions** (`seasonality::convenience` module)
  - `deseasonalize()`, `detrend()`, `seasonal_component()`, `trend_component()`, `remainder_component()`
  - `recompose()` — reconstruct series from trend + seasonal + remainder
  - `seasonal_adjust()` — return new TimeSeries with seasonal component removed
  - `STLResult::deseasonalized()`, `.detrended()`, `.recompose()` methods

- **Intermittent Demand Diagnostics**
  - `IntermittentDiagnostics` — Syntetos-Boylan (2005) demand classification framework
  - `DemandClassification`: Smooth, Erratic, Intermittent, Lumpy (ADI/CV² thresholds)
  - ADI (Average Demand Interval), CV² of non-zero demands, zero fraction
  - `recommended_model()` — suggest Croston/TSB/SES based on classification
  - Coverage rate, bias, and Periods-in-Stock (PIS) metrics

- **Model Diagnostics Pipeline**
  - `ModelDiagnostics::from_residuals()` — Ljung-Box, Jarque-Bera, Breusch-Pagan tests
  - `ModelDiagnostics::from_forecaster()` — extract residuals and run all diagnostics
  - ACF/PACF of residuals, residual mean/std, `passes_all` flag

- **Forecast Explainability**
  - `ForecastExplanation` struct: level, trend, seasonal, residual, named components
  - `Explainable` trait implemented for ETS, Theta, MSTLForecaster
  - Components sum to forecast values

- **TimeSeries Temporal Aggregation**
  - `aggregate(period, method)` — Sum, Mean, Median, First, Last, Min, Max
  - `downsample(factor)` — decimation with timestamp preservation
  - `upsample(factor, method)` — Linear, ForwardFill, BackwardFill, Zero interpolation
  - `sliding_window_aggregate(window, step, method)` — configurable sliding windows

- **Hierarchy Reconciliation Methods**
  - `ReconciliationMethod::MiddleOut { middle_level }` — reconcile from a chosen depth
  - `ReconciliationMethod::MinTraceShrink` — MinT with Ledoit-Wolf shrinkage covariance

- **Ensemble Combination Methods**
  - `CombinationMethod::InverseAIC` — Akaike weights from estimated AIC
  - `CombinationMethod::Stacking { folds }` — non-negative constrained linear combiner
  - `CombinationMethod::HorizonAdaptive` — per-horizon weights from rolling-origin evaluation

- **Error Context**
  - `ForecastError::SubModelError` — wraps sub-model failures with model name context
  - `ForecastError::FitRequired` now carries optional model name

- **Forecaster Trait Adapters**
  - `VARForecaster` — adapts multivariate VAR for univariate Forecaster interface
  - `KalmanForecaster` — adapts Kalman filter with local_level/local_linear_trend constructors

- **CV Embargo**
  - `CvFoldGenerator::embargo(n)` — exclude observations after test sets (financial ML)
  - `CVConfig::with_embargo(n)` — embargo for config-based CV

- **WASM/JS Enhancements**
  - `AutoForecastBuilder` — fluent builder for AutoForecast in JS
  - `EnsembleForecaster.setInverseAic()` / `.setStacking()` / `.setHorizonAdaptive()` / `.setMethod(name)`
  - `Forecast.nonNegative()`, `.clamp()`, `.roundToInteger()` — constraint methods in JS
  - `JsModelDiagnostics.fromResiduals()` — diagnostics in JS with all property accessors
  - `VARForecaster`, `KalmanForecaster` — multivariate and state-space models in JS

- **Benchmarks**
  - STL scratch reuse comparison, MSTL multi-period
  - ARIMA/SARIMA fitting at multiple series lengths
  - Periodicity detection (autocorrelation, Welch periodogram)
  - Model comparison (Naive, SES, Theta, ETS, ARIMA)
  - Hot paths (SIMD ops, forecast construction, TimeSeries slicing)

- **Integration Tests**
  - VAR: 13 tests (coefficient recovery, Granger causality, forecast accuracy)
  - Persistence: 16 tests (JSON/bincode round-trips for all model types)
  - Pipeline: 12 tests (ensemble+constraints+postprocessing, STL+recompose, CV+select)

- **Prediction Interval Improvements**
  - RandomWalkWithDrift: proper drift SE + variance scaling formula
  - SMA/WindowAverage: (1 + 1/w) factor for mean estimation uncertainty

### Changed

- Kalman filter uses flat `DenseMatrix` layout with in-place operations and pre-allocated scratch buffers
- Ensemble supports InverseAIC, Stacking, and HorizonAdaptive combination methods

- `parallel` feature now covers AutoForecast, model comparison, bootstrap, cross-validation folds, and rolling forecast windows (previously only AutoARIMA)
- `serde` feature now includes bincode for binary serialization alongside JSON
- Several `ComputationError` uses migrated to specific error variants (`ConvergenceFailure`, `SingularMatrix`)
- Removed dead code in `mfles.rs` and `tbats/model.rs`
- MSTL uses in-place decomposition for reduced allocations
- STL decomposition supports buffer reuse via `StlScratch`
- ETS uses `Cow<[f64]>` to avoid cloning series values when no regressors are present
- Cross-validation uses direct slice references to avoid intermediate allocations per fold
- Ensemble `predict_with_intervals()` produces widest-envelope combined intervals
- Test coverage increased to 2,480+ tests
- `MissingValuePolicy` enum has 4 new variants (breaking for exhaustive `match` — acceptable under 0.x semver)

## [0.4.1] - 2026-01-16

### Added

- **ETS Notation Parser & FPP3 Taxonomy Compliance**
  - `ETSSpec::from_notation("AAA")` - Create ETS models from standard notation
  - `ETSSpec::is_valid()` - Validate model combinations before fitting
  - Reject unstable ETS combinations (MAA, MAdA) per [FPP3 taxonomy](https://otexts.com/fpp3/taxonomy.html)
  - New convenience constructors: `ana()`, `anm()`, `aada()`, `aadm()`, `mnm()`, `madm()`

- **WASM/npm Package Enhancements**
  - `ETSForecaster.fromNotation("AAA", period)` - Standard notation in JavaScript
  - `ETSForecaster.isValidSpec(error, trend, seasonal)` - Validation helper
  - Constructor validation rejects unstable ETS combinations
  - npm package documentation with ETS notation examples

- **Comprehensive Test Coverage**
  - 76 WASM tests covering all 29 forecaster classes
  - 23 JavaScript integration tests (Node.js)
  - Edge case tests: NaN handling, single data point, negative values, large/small values
  - ETS notation parsing tests (valid, invalid, unstable combinations)

- **CI/CD Improvements**
  - JavaScript integration tests in CI workflow
  - npm OIDC trusted publishing (no tokens required)
  - Requires npm >= 11.5.1 for OIDC support

- **Documentation**
  - "Use Cases" section in README with DuckDB extension and npm package links
  - Updated npm README with ETS notation API documentation
  - FPP3 taxonomy reference in API docs

### Changed

- Test coverage increased to 1,400+ tests (unit + integration + WASM + JS)
- Installation instructions updated to v0.4

## [0.4.0] - 2026-01-12

### Added

- **Probabilistic Forecasting Module** (`postprocess` feature, enabled by default)
  - `PostProcessor` - Unified API for probabilistic forecast calibration
  - `PredictionIntervals` - Multi-level interval representation with coverage guarantees
  - **Conformal Prediction** - Distribution-free prediction intervals
  - **Conformalized Quantile Regression** - Calibrated quantile forecasts
  - **Quantile Regression Averaging (QRA)** - Ensemble-based probabilistic forecasts
  - **Historical Simulation** - Bootstrap-based uncertainty estimation
  - **Normal Approximation** - Parametric prediction intervals
  - **Isotonic Distributional Regression (IDR)** - Non-parametric calibration
  - **Backtesting** - Horizon-aware backtesting with automatic calibration
    - `BacktestConfig` with expanding/rolling windows
    - Per-horizon calibration for improved accuracy
    - Coverage and calibration error metrics

- **Cross-Validation Enhancements**
  - `CvFoldGenerator` - Standalone fold generation for custom workflows
  - `gap` parameter - Prevents data leakage from lagged features
  - `purge` parameter - Removes observations to prevent lookahead bias (financial applications)
  - `FillStrategy` trait - Handle unknown future features during CV
    - Implementations: `LastValueFill`, `MeanFill`, `MedianFill`, `ZeroFill`, `ConstantFill`, `ModeFill`
  - `train_test_split()` - Simple ratio or index-based splitting
  - `train_test_split_at()` - Split at specific index
  - `grouped_cross_validate()` - Multi-series CV with consistent fold boundaries
  - `GroupedCVResults` - Per-group and aggregated metrics
  - `Fold` struct - Explicit train/test index representation

- **Architecture Documentation**
  - ADR explaining CV split design decisions
  - Documents why CV split lives in DuckDB extension vs this crate
  - Component distribution rationale (fold generation, orchestration, etc.)

- **New Examples**
  - `postprocess/quickstart.rs` - Getting started with probabilistic forecasts
  - `postprocess/conformal.rs` - Conformal prediction intervals
  - `postprocess/conformalize.rs` - Conformalized quantile regression
  - `postprocess/qra_ensemble.rs` - QRA ensemble methods
  - `postprocess/quantile_methods.rs` - Various quantile approaches
  - `postprocess/unified_api.rs` - PostProcessor unified API
  - `postprocess/backtest.rs` - Backtesting workflows

### Changed

- Cross-validation now returns `folds` field in `CVResults` for transparency
- Test coverage increased to 1,316+ tests

### Dependencies

- Added `faer` (optional) for linear algebra in postprocessing
- Added `anofox-regression` v0.5.0 (optional) for quantile regression

## [0.3.2] - 2026-01-07

### Fixed

- **Stable Rust Compatibility**
  - Replaced unstable `is_multiple_of()` method with `% 2 == 0`
  - Fixes WASM builds on stable Rust 1.86.0+

## [0.3.1] - 2026-01-07

### Added

- **WASM Target Support**
  - Compilation for `wasm32-unknown-unknown` target
  - `js` feature flag for browser environments (enables getrandom/js)
  - Compile-time guard preventing `parallel` feature on WASM targets

## [0.3.0] - 2026-01-03

### Added

- **Optional Parallel AutoARIMA**
  - Feature-gated Rayon parallelization for model evaluation
  - Enable with `--features parallel` for 4-8x speedup
  - Default: sequential execution (DuckDB compatible)
- **Bootstrap Confidence Intervals**
  - `BootstrapConfig` for configuring bootstrap parameters
  - `bootstrap_intervals()` for empirical confidence intervals
  - `bootstrap_forecast()` convenience function
  - Residual bootstrap and block bootstrap methods
- **True Stepwise Search for AutoARIMA**
  - Neighbor-based hill climbing algorithm
  - Reduces model evaluations by 60-70%
  - Enable with `AutoARIMAConfig::with_true_stepwise(true)`
- **Property-Based Testing**
  - 20+ proptest cases for model invariants
  - Tests forecast length, finite values, interval ordering
  - Tests fitted values + residuals reconstruction
- **Interval Calibration Testing**
  - Rolling origin cross-validation for coverage rate testing
  - Winkler score for interval quality assessment
  - Coverage rate tests for analytical and bootstrap intervals

### Removed

- **Time Series Clustering** - Removed clustering module (not needed)

### Changed

- Improved test coverage with 1,136+ tests total
- Updated dependencies

## [0.2.0] - 2025-12-17

### Added

- **Periodicity Detection Module**
  - `ACFPeriodicityDetector` - time-domain detection using ACF peaks
  - `FFTPeriodicityDetector` - frequency-domain detection using periodogram
  - `Autoperiod` - hybrid FFT+ACF detector (Vlachos et al. 2005)
  - `CFDAutoperiod` - noise-resistant detector with clustering (Puech et al. 2020)
  - `SAZED` - parameter-free ensemble method (Toller et al. 2019)
  - Convenience functions: `detect_period()`, `detect_period_ensemble()`, `detect_period_range()`
  - `PeriodicityDetector` trait for unified API
- **FFT Utilities**
  - `fft_real()` - FFT for real-valued signals
  - `periodogram()` - power spectral density computation
  - `periodogram_peaks()` - significant peak detection
  - `welch_periodogram()` - Welch's method for reduced variance
- **SIMD-Accelerated Operations**
  - Vector sum, mean, variance, standard deviation
  - Dot product and sum of squares
  - Squared Euclidean and Manhattan distances
  - Element-wise operations (add, subtract, multiply, divide, scale)
  - Uses Trueno for AVX2/SSE2/NEON acceleration
- **Validation Tools**
  - CLI tool for periodicity detection (`examples/analysis/detect_period.rs`)
  - Python cross-validation script against pyriodicity
  - Criterion benchmarks for periodicity detection

### Changed

- Updated documentation with periodicity detection examples
- Added `rustfft` dependency for FFT operations

## [0.1.0] - 2025-12-11

### Added

- Initial release of anofox-forecast
- **Core Data Structures**
  - `TimeSeries` for univariate and multivariate time series data
  - `Forecast` for prediction results with confidence intervals
  - `CalendarAnnotations` for holidays and regressors
- **Forecasting Models (35+)**
  - ARIMA and AutoARIMA with automatic order selection
  - Exponential Smoothing: SES, Holt's Linear, Holt-Winters, ETS, AutoETS
  - Baseline methods: Naive, Seasonal Naive, Random Walk with Drift, SMA
  - Theta method
  - Intermittent demand: Croston, ADIDA, TSB
  - Ensemble methods with multiple combination strategies
- **Feature Extraction (76+ features)**
  - Basic statistics (mean, variance, quantiles, etc.)
  - Distribution features (skewness, kurtosis, etc.)
  - Autocorrelation and partial autocorrelation
  - Entropy features (approximate, sample, permutation, binned)
  - Complexity features (C3, CID, Lempel-Ziv)
  - Trend analysis and stationarity tests
- **Seasonality & Decomposition**
  - STL (Seasonal-Trend decomposition using LOESS)
  - MSTL (Multiple Seasonal-Trend decomposition)
- **Changepoint Detection**
  - PELT algorithm with L1, L2, Normal, and Poisson cost functions
- **Anomaly Detection**
  - Statistical methods (IQR, z-score)
  - Automatic threshold selection
- **Time Series Clustering**
  - Dynamic Time Warping (DTW) distance
  - K-Means clustering with multiple distance metrics
- **Data Transformations**
  - Scaling: standardization, min-max, robust scaling
  - Box-Cox transformation with automatic lambda selection
  - Window functions: rolling and expanding statistics, EWM
- **Model Evaluation**
  - Accuracy metrics (MAE, MSE, RMSE, MAPE, etc.)
  - Time series cross-validation
  - Residual testing and stationarity tests
