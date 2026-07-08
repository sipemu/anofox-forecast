# Accuracy improvement audit — notes

Systematic evaluation of the 8 accuracy improvement ideas proposed after the STL-auto revert. For each: current state in code, feasibility check, expected impact, regression risk, decision.

*Last updated: fev-27 baseline is `.skaters()` at geomean MASE 6.085, WQL 0.6695 (post Fix A+B, before items 2-9).*

---

## Attempted #1 — STACKING REGRESSED, REVERTED

**Implemented** as `.with_stacking()`: snapshots per-leaf 1-step-ahead
prediction means during fit, solves ridge OLS (λ=1e-4) via normal
equations + Gaussian elimination, projects to non-negative simplex,
uses those weights in `forecast_dist` for the mean blend.

**Result on fev-27 with `.auto().with_stacking()`**: **REGRESSION**.

| model | before | after stacking | Δ |
|---|---:|---:|---:|
| `.auto()` MASE | 5.924 | **6.519** | **+10 %** ❌ |
| `.auto()` WQL | 0.159 | 0.168 | +5.7 % ❌ |

**Diagnosis**:
- Ridge too small: `1e-4 · effective_n` is negligible compared to
  typical XᵀX magnitudes. Effectively unregularized OLS on highly
  collinear leaf predictions (multiple EMAs at nearby α).
- In-sample overfit: OLS on training predictions can't distinguish
  "leaf that genuinely predicts" from "leaf that overfits training".
  Softmax at least uses log-likelihood which is bounded, whereas OLS
  on unbounded MSE can lock onto noise.
- Missing holdout CV: proper stacking uses out-of-sample predictions
  (K-fold or time-series CV). My in-sample impl is a known
  anti-pattern.

**Fix candidates** (untried):
- Rolling-origin holdout: for each observation `y_t`, use leaves that
  had NOT seen `y[t-1..t]` when they emitted their prediction. Since
  our streaming leaves see each obs exactly once at `observe(y)`, and
  the prediction at step t is made BEFORE observe(y_t), our
  predictions ARE technically one-step-ahead — but the ensembled
  prediction still overfits because all leaves have seen the same
  training window.
- Aggressive ridge: try λ = 0.1, 1.0, 10.0.
- Blend stacking + softmax: `w_final = α · w_stacking + (1-α) · w_softmax`.
- NNLS with sum-to-one via Lagrangian rather than simplex projection.

**Code kept**: `.with_stacking()` builder + fields remain — opt-in only,
not auto-enabled. Callers can experiment with `.with_stacking()` on
their own data. See `predictions_history`, `stacking_weights`,
`solve_stacking()`, `project_to_simplex()` in `forecaster.rs`.

**Decision**: MOVE ON. Stacking as implemented doesn't work. Would
need deeper thought — potentially rolling-origin CV or aggressive ridge
tuning — before it's a real improvement.

## Original evaluation of #1 (before implementation)

## #1 — Ensemble stacking with OLS/ridge on training predictions

**Idea**: after fit, learn a linear blend of leaf forecasts by ridge against training values. Replace/augment the softmax blend at forecast time.

**Current state**:
- `crate::utils::ols::ols_fit(y, HashMap<String, Vec<f64>>)` already exists and is imported by `forecaster.rs` (`use crate::utils::ols::{ols_fit, OLSResult}`).
- The fit loop already computes `per_leaf: Vec<Gaussian>` per step. Only the current-step Gaussians survive; earlier steps' predictions are dropped.
- No ridge (just plain OLS) — collinearity from `EMA(0.05) ≈ EMA(0.1)` would need small ridge λ.

**Feasibility**: HIGH.
1. Add a `predictions_history: Vec<Vec<f64>>` field (N_leaves × N_steps of means) or a `Vec<Vec<f64>>` of shape [leaf_idx][step].
2. During the fit loop, after `predict_one()`, push `per_leaf[i].mean` into the history.
3. After the fit loop, run OLS: `y_train[burn_in..] ~ Σ w_i · predictions_history[i][burn_in..]`.
4. At `forecast_dist(h)`, use OLS weights (`w_i`) instead of softmax weights for the mean blend. Keep softmax for the σ mixture.

Storage: N_leaves × N_train × 8 = 30 × 1400 × 8 = 336 KB on M5. Fine.

**Expected impact**: 5–15 % MASE on M-competition monthly/quarterly. Directly attacks the "softmax's 1-step objective ≠ H-step accuracy" issue.

**Risk**: MODERATE.
- Ridge overfitting on short-history data (N < 100). Mitigate: use holdout-fold CV instead of in-sample OLS on short data, or fall back to softmax when N < some threshold.
- Weights could be negative (OLS doesn't constrain non-negativity) → non-convex mixture. Mitigate: use non-negative least squares (NNLS) which we already have as a regression backend.
- Won't help distributional metrics (WQL, CRPS) — those depend on σ mixture, not point mean.

**Decision**: STRONG CANDIDATE. Use NNLS with ridge, gated on `N ≥ 100`. Below threshold, fall back to softmax. Estimated effort: 1-2 days.

---

## #2 — Multi-horizon softmax scoring

**Idea**: score each leaf on cumulative h-step LL retrospectively during fit (not just 1-step).

**Current state**:
- Fit loop already has a periodic snapshot mechanism (line 2392-2413) for per-h calibration. It runs `l.predict(per_h_horizon)` on every leaf every `snapshot_stride` steps.
- Snapshots are stored in `per_h_snapshots: Vec<(step, Vec<(f64, f64)>)>` — but they only capture the MIXTURE mean/std, not per-leaf predictions.

**Feasibility**: MEDIUM.
1. Modify snapshot storage to also keep `per_leaf_predictions: Vec<Vec<Vec<f64>>>` (snapshot × leaf × horizon).
2. After the fit loop, retrospectively score: at snapshot time `t`, we now know `y[t+h]` for `h ∈ 1..=per_h_horizon`.
3. Compute `retrospective_lp[i] = Σ_snapshots Σ_h logpdf(pred_i_at(t, h), y[t+h])`.
4. Blend the 1-step `cum_log_liks[i]` with the retrospective score.

**Expected impact**: 3-8 % MASE on long-horizon datasets. But the biggest flat-line issue (seasonal-diff drift) is already fixed by Fix B. Marginal after Fix B.

**Risk**: LOW.
- Snapshot storage: `snapshots × N_leaves × H × 8 = ~30 × 30 × 48 × 8 = 350 KB` on a 30-snapshot m4_hourly fit. Fine.
- Doesn't replace existing softmax; augments the signal.
- Only affects blend weights; leaves untouched.

**Decision**: MODERATE CANDIDATE. Effort ~1 day. Should try only after #1 (stacking) if that doesn't give enough gain.

---

## #3 — Robust batch warm-start on short data

**Idea**: for N < 5 × period, initialize leaf state from batch stats (level = median, σ = MAD).

**Current state**:
- EMA/Drift/AR leaves all start `level: None`. On first `observe(y)`, they set `level = y[0]`. Vulnerable to outlier at position 0.
- Welford-like bootstrap: `α = max(configured_α, 1/n)` in the first steps. So the effective learning rate is high early. Already partial-batch-init behavior.
- `TerminalScaleMixture` also uses `α = max(scale_alpha, 1/n)` bootstrap.

**Feasibility split into two variants**:

**3a) Robust terminal-σ init only** — LOW EFFORT:
- Before the fit loop, compute `mad = 1.4826 * median(|y - median(y)|)` on the training values.
- Initialize `terminal.v = mad²` and set `terminal.n_obs = 30` (so the EWMA doesn't over-write it immediately).
- ~30 min work.

**3b) Per-leaf batch stats** — HIGH EFFORT:
- Add `fn from_batch_stats(alpha, level, sigma) -> Self` on each hot leaf (~8 leaves).
- Pass `median(y), mad(y)` at construction time.
- ~3-4 days work.

**Expected impact**: 3-7 % MASE on N < 50 panels (M1/M3/M4 yearly). Marginal elsewhere.

**Risk**: 3a is LOW (only changes σ init). 3b is MODERATE (could lock in bad state on regime-change data).

**Decision**: 3a is a GOOD FIRST STEP. Test it, measure, decide about 3b based on result.

---

## #4 — Trend regularization with horizon decay

**Idea**: multiply extrapolated trend by `exp(-h · λ)`.

**Current state**:
- `DriftLeaf::predict`: `mean = level + h · drift` — LINEAR extrapolation, unbounded.
- `HoltLeaf::predict`: uses `damped_sum(h) · trend` where `damped_sum` uses `φ ∈ (0, 1)` — ALREADY dampened per #180's α-27 fix. Default `φ=0.9`.
- `AutoTheta` (our reference in fev-27) uses damped trend to great effect on m4_yearly / m3_yearly.

**Feasibility**: TWO PATHS:

**4a) Damp DriftLeaf** — 30 min work. Add `damping: f64` field (default 1.0 = no damping). Change `predict` to `level + drift * (1 - damping^h) / (1 - damping)` when damping < 1.

**4b) Add HoltLeaf to `.skaters()` pool** — 5 min work. HoltLeaf is currently only enabled when `.auto()` detects trend. Add unconditional Holt(0.3, 0.1, 0.9) and Holt(0.2, 0.05, 0.85) to `.skaters()`.

**Expected impact**:
- 4a helps only when DriftLeaf wins the softmax on trending data (probably rare — Holt usually wins on trend).
- 4b: modest. `.skaters()` already has damped Holt via `use_auto=true` path but `.skaters()` doesn't set `use_auto`. Adding 2 Holt variants directly to `.skaters()` pool might close 3-5% on trending panels.

**Risk**: LOW for both. Just adds candidates.

**Decision**: 4b is a QUICK WIN worth trying. Effort: 5 minutes. Independent of stacking.

---

## #5 — Autoregressive residual model in the terminal

**Idea**: replace terminal's IID assumption with AR(1) on residuals.

**Current state**:
- `TerminalScaleMixture::observe(r)` treats each residual as IID: EWMA of `r²` gives σ, weights update on standardized `z = r/σ`.
- `TerminalCrpsMixture::observe(r)` — same shape, CRPS-gradient weight update.
- Neither tracks `E[r_t · r_{t-1}]` for AR(1) φ estimation.
- At forecast time, `predict_shifted` emits mixture centered at `softmax_mean`, spread from `√v`. Fev-27 follow-up (#3) added `√h` scaling for multi-horizon. With true AR(1) φ, the h-step variance would be:
   `Var[r_{t+h}] = σ² · (1 - φ^(2h)) / (1 - φ²)` — slower growth than √h for `φ > 0`.

**Feasibility**: MEDIUM.
1. Add `last_r: Option<f64>` and `phi: f64` fields to `TerminalScaleMixture`.
2. In `observe`, update: `φ_new = 0.98 · φ + 0.02 · sign(r · last_r) · min(1, |r · last_r| / max(v, 1e-9))`. (EWMA of the residual autocorrelation.)
3. Cap `|φ| ≤ 0.95` for stationarity.
4. At `predict_shifted(mean, h)`, scale σ by `√((1 - φ^(2h)) / (1 - φ²))` instead of `√h`.

**Expected impact**:
- 5-15 % WQL improvement on datasets where residuals are correlated (mostly economic / macro — fred_md, m3_monthly).
- No MASE impact (only changes σ, not mean).

**Risk**: MODERATE.
- Wrong φ makes the variance too tight or too loose. Cap prevents divergence but not calibration errors.
- On truly IID data (M5 differences), added complexity without benefit.

**Decision**: FEASIBLE. Focused on WQL, not MASE. Only worth doing if WQL is the priority (currently the fev-27 gap on WQL is bigger than MASE for `.skaters()`). Effort: 1 day. Test on FRED bakeoff first (continuous, likely AR residuals).

---

## #6 — Bayesian shrinkage across correlated leaves

**Idea**: pull log-weights within a family (all EMAs, all ARs) toward the family mean.

**Current state**:
- `.skaters()` pool contains ~30 candidates with heavy family overlap:
  - 3-5 EMAs at α ∈ {0.05, 0.10, 0.25, 0.3, 0.5, 0.95}
  - 3-4 Drift at α ∈ {0.05, 0.10, 0.15, 0.25}
  - Multiple Ar1/Ar2 variants
  - 3 Theta at α ∈ {0.05, 0.10, 0.30}
  - Seasonal-diff at 6 (period, α) combinations
- Softmax scores each independently. Correlated leaves waste degrees of freedom.

**Feasibility**: MEDIUM.
1. Tag each leaf with a `family: &'static str` (e.g., `"ema"`, `"drift"`, `"theta"`).
2. After computing `cum_log_liks`, group by family and compute family mean `μ_f`.
3. Shrinkage: `log_w_shrunk[i] = (1 - λ) · cum_log_liks[i] + λ · μ_f[family_i]`.
4. Softmax on shrunk weights.

**Expected impact**: 2-5 % MASE on short-history panels where softmax is noisy. Marginal on long-history.

**Risk**: MODERATE.
- Wrong λ smooths out real signal. Would need to be conservative (λ = 0.1?).
- Doesn't address the underlying issue: MULTIPLE ALTERNATIVES aren't the same as MORE DATA.

**Decision**: LOW PRIORITY. Better to fix the objective (via #1 stacking) than to tweak the mechanism.

---

## #7 — `.auto()` with `.skaters()`-style η schedule + log-clamp

**Idea**: 2-line change. `.auto()` currently uses `η=1.0, clamp=-∞`; try `η=0.5, clamp=-20`.

**Current state**:
- `pub fn auto(mut self) -> Self` (forecaster.rs:1531): sets `use_auto = true`, enables terminal scale-mixture, Theta variants, standardize+EMA, seasonal-diff+EMA compositions. Does NOT touch `learning_rate` or `log_clamp`.
- `.skaters()` sets `learning_rate = 0.5, log_clamp = -20.0`.
- The fev-27 shows `.auto()` at MASE 6.436 and `.skaters()` at 6.085. But per-dataset it's mixed — `.auto()` beats `.skaters()` on m3_monthly (0.791 vs 0.798), m4_daily (1.165 vs 1.109), etc.

**Feasibility**: TRIVIAL. 2 lines in `.auto()`:
```rust
self.learning_rate = 0.5;
self.log_clamp = -20.0;
```

**Expected impact**: Uncertain. Could go either way per-dataset. The XGBoost-shrunk update reduces variance at the cost of slower convergence. On short-history panels where `.auto()` currently over-concentrates, might help; on long-history where it's fine, might hurt.

**Risk**: LOW-MODERATE. A/B test.

**Decision**: MEASURE FIRST. Since the change is 2 lines, worth running a targeted fev-27 subset (m3_monthly, m3_yearly, m4_daily, m1_yearly) to check direction. Effort: 30 min.

---

## #8 — MSTL / multi-seasonality

**Idea**: use existing `MSTL` for multi-period series (electricity: daily + weekly).

**Current state**:
- `crate::seasonality::MSTL::new(seasonal_periods: Vec<usize>)` exists and has `.decompose(series) -> Option<MSTLResult>`.
- Result has multiple `seasonal` components (one per period), plus trend + remainder.
- `MstlDecompLeaf` doesn't exist. `StlDecompLeaf` (which we ported) uses single-period `STL`, not MSTL.

**Where multi-seasonality actually helps in fev-27**:
- **m4_hourly** (H=48, period=24): has both daily (24) AND weekly (168) seasonality. Currently we only see period=24.
- **electricity_hourly** (deferred dataset): daily + weekly.
- **traffic** (deferred): daily + weekly.
- Everything else in fev-27 has single dominant period.

**Feasibility**: MEDIUM.
- Port `MstlDecompLeaf(&[period]))` — same shape as `StlDecompLeaf` but with vector of periods.
- Would inherit STL's regression problem on short-history data. **Same tourism_monthly-style risk as #9 STL revert.**
- Only useful on datasets with multi-seasonality AND enough training data (N > 5 × max_period).

**Expected impact**: 5-10 % MASE on m4_hourly specifically, marginal elsewhere. Not enough to move the fev-27 geomean.

**Risk**: HIGH. Same as STL — extrapolation compounds on short history.

**Decision**: NOT RECOMMENDED given the STL failure. If we ever want to close m4_hourly further, the multi-scale wrapper (#6 from prior audit) or a proper seasonal-Kalman is safer than MSTL extrapolation.

---

## Consolidated ranking (post-audit)

Given the STL failure, my updated ranking is:

| item | expected impact | risk | effort | verdict |
|---|---|---|---|---|
| #7 (η + clamp on `.auto()`) | unknown, could win 5% or lose 5% | low | 30 min | **MEASURE FIRST** (2-line change) |
| #4b (Holt in `.skaters()` pool) | 3-5 % on trending panels | low | 5 min | **QUICK WIN** |
| #3a (robust terminal σ init from MAD) | 3-7 % on short-history | low | 30 min | **QUICK WIN** |
| #1 (ensemble stacking) | 5-15 % MASE potentially | moderate | 1-2 days | **BEST FOR M-COMPETITION MONTHLY/QUARTERLY** |
| #5 (AR terminal) | 5-15 % WQL | moderate | 1 day | **WORTH IF WQL IS PRIORITY** |
| #2 (multi-h scoring) | 3-8 % marginal after Fix B | low | 1 day | Do after #1 |
| #6 (family shrinkage) | 2-5 % | moderate | 1 day | Skip in favor of #1 |
| #8 (MSTL) | 5-10 % on 1 dataset | HIGH (STL failure risk) | 2-3 days | **SKIP** |

## Recommended sequence

1. **Quick wins (35 min total)**: try #7 + #4b + #3a as a batch. Measure. If any regress, back that one out. Cheap first step.
2. **Structural win (1-2 days)**: implement #1 (ensemble stacking with NNLS + ridge). This has the highest expected impact on M-competition monthly/quarterly which is our biggest current gap.
3. **Optional (1 day each)**: #5 for WQL, #2 for further multi-horizon polish. Only if #1's gain is smaller than expected.

## What I would NOT do

- STL / MSTL variants (documented failure)
- Family shrinkage (masks the real issue; better to fix objective)
- Batch per-leaf init (high effort, moderate risk, marginal impact)
- Multi-scale universal enable (documented — only helps N > 10 × horizon)

