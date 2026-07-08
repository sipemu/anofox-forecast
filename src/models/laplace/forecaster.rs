//! `LaplaceForecaster` — online distributional shell over EMA / drift /
//! AR(1) / damped-Holt, plus optional seasonal-EMA.
//!
//! Alpha surface (behind the `distributional` feature). Inspired by
//! [`microprediction/skaters`](https://github.com/microprediction/skaters):
//! streaming leaves, likelihood-weighted mixture, per-horizon
//! [`GaussianMixture`] output. Only the shell
//! and a small leaf set is implemented — no CRPS terminal, no
//! OU / fractional-differencing / Yeo-Johnson leaves.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::inspect::{Explanation, Inspectable, LaplaceExplanation};
use crate::models::traits::{validate_series_complete, Forecaster};

use super::dist::GaussianMixture;
use super::leaves::{SlowStandardizeWrapper, TerminalScaleMixture};
use crate::transform::yeo_johnson::yeo_johnson_lambda;
use crate::utils::ols::{ols_fit, OLSResult};
use std::collections::HashMap;

use super::ensemble::{blend_horizon, softmax};

/// Series characteristics used by the auto-selector.
#[derive(Clone, Copy)]
struct AutoChars {
    seasonality_strength: f64,
    acf1: f64,
    /// R² of a linear fit `y ~ t`. High values (> ~0.5) indicate a
    /// dominant trend — the auto-selector uses this to avoid enabling
    /// AR(2) on trending series (its MoM estimator pushes `φ₁ + φ₂ → 1`
    /// on strong trends, producing recursive h-step blow-ups even with
    /// the leaf's stationarity projection).
    trend_strength: f64,
    /// Fraction of observations at or near zero. Used to route
    /// demand-side (Croston, seasonal-Croston) leaves.
    zero_fraction: f64,
    /// Sample mean. Positive-mean series can be routed to multiplicative
    /// / lognormal / gamma leaves.
    mean_y: f64,
    /// True if all observations are ≥ 0 (needed for multiplicative
    /// seasonal, lognormal, gamma leaves).
    all_positive: bool,
}

/// Detect the most likely seasonal period from the training window.
/// Scans a canonical set of candidate periods {7, 12, 24, 30, 52, 4}
/// and picks the one with the highest ACF at that lag. Returns `None`
/// if none of the candidates has ACF above a threshold — the caller
/// then falls back to the user-configured `auto_seasonal_period`.
pub(crate) fn detect_seasonal_period(train: &[f64]) -> Option<usize> {
    let n = train.len();
    if n < 30 {
        return None;
    }
    let mean_y: f64 = train.iter().sum::<f64>() / n as f64;
    let var: f64 = train.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-9 {
        return None;
    }
    let candidates: [usize; 6] = [12, 7, 24, 52, 4, 30];
    let mut best_period = 0usize;
    let mut best_acf = 0.35_f64; // threshold — below this, no period is picked
    for &p in &candidates {
        if p >= n / 2 {
            continue;
        }
        let mut cov = 0.0f64;
        for i in p..n {
            cov += (train[i] - mean_y) * (train[i - p] - mean_y);
        }
        let acf = (cov / ((n - p) as f64 * var)).clamp(-1.0, 1.0).abs();
        if acf > best_acf {
            best_acf = acf;
            best_period = p;
        }
    }
    if best_period > 0 {
        Some(best_period)
    } else {
        None
    }
}

/// Compute (seasonality_strength_R², |ACF(1)|) on the training window.
/// Same formulas as `examples/skaters_m5_benchmark.rs` so the auto-selector
/// respects the same slicing evidence.
fn auto_characteristics(train: &[f64], period: usize) -> AutoChars {
    let n = train.len();
    if n < 2 {
        return AutoChars {
            seasonality_strength: 0.0,
            acf1: 0.0,
            trend_strength: 0.0,
            zero_fraction: 0.0,
            mean_y: 0.0,
            all_positive: true,
        };
    }
    let mean_y: f64 = train.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = train.iter().map(|y| (y - mean_y).powi(2)).sum();
    let zero_fraction = train.iter().filter(|&&y| y.abs() < 1e-9).count() as f64 / n as f64;
    let all_positive = train.iter().all(|&y| y >= 0.0);

    // Trend strength: R² of the linear fit y ~ t.
    let t_mean = (n - 1) as f64 / 2.0;
    let (mut sum_ty, mut sum_tt) = (0.0, 0.0);
    for (t, y) in train.iter().enumerate() {
        let dt = t as f64 - t_mean;
        sum_ty += dt * (y - mean_y);
        sum_tt += dt * dt;
    }
    let slope = if sum_tt > 0.0 { sum_ty / sum_tt } else { 0.0 };
    let intercept = mean_y - slope * t_mean;
    let ss_res_trend: f64 = train
        .iter()
        .enumerate()
        .map(|(t, y)| (y - (intercept + slope * t as f64)).powi(2))
        .sum();
    let trend_strength = if ss_tot > 0.0 {
        (1.0 - ss_res_trend / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Phase-mean seasonal fit R².
    let period = period.max(1);
    let mut phase_sum = vec![0.0f64; period];
    let mut phase_count = vec![0usize; period];
    for (i, &y) in train.iter().enumerate() {
        phase_sum[i % period] += y;
        phase_count[i % period] += 1;
    }
    let phase_mean: Vec<f64> = phase_sum
        .iter()
        .zip(phase_count.iter())
        .map(|(s, &c)| if c > 0 { s / c as f64 } else { mean_y })
        .collect();
    let ss_res_season: f64 = train
        .iter()
        .enumerate()
        .map(|(i, y)| (y - phase_mean[i % period]).powi(2))
        .sum();
    let seasonality_strength = if ss_tot > 0.0 {
        (1.0 - ss_res_season / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // |AR(1)| lag-1 autocorrelation.
    let mut num = 0.0f64;
    for i in 1..n {
        num += (train[i - 1] - mean_y) * (train[i] - mean_y);
    }
    let acf1 = if ss_tot > 0.0 {
        (num / ss_tot).clamp(-1.0, 1.0).abs()
    } else {
        0.0
    };

    AutoChars {
        seasonality_strength,
        acf1,
        trend_strength,
        zero_fraction,
        mean_y,
        all_positive,
    }
}

/// Yeo-Johnson forward transform (scalar).
#[inline]
fn yj_forward(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if lambda.abs() < 1e-12 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        -(-x + 1.0).ln()
    } else {
        -(((-x + 1.0).powf(2.0 - lambda)) - 1.0) / (2.0 - lambda)
    }
}

/// Yeo-Johnson inverse (scalar). Returns `(x, |dx/dy|)` for delta-method
/// std propagation. Saturates to the domain boundary and Jacobian = 0
/// when the requested inverse is outside the definition (e.g. `λ · y + 1
/// ≤ 0` on the positive branch).
#[inline]
fn yj_inverse_with_jac(y: f64, lambda: f64) -> (f64, f64) {
    if y >= 0.0 {
        if lambda.abs() < 1e-12 {
            let ey = y.exp();
            (ey - 1.0, ey)
        } else {
            let base = lambda * y + 1.0;
            if base <= 0.0 {
                (0.0, 0.0)
            } else {
                let inv_lambda = 1.0 / lambda;
                let x = base.powf(inv_lambda) - 1.0;
                let dxdy = base.powf(inv_lambda - 1.0);
                (x, dxdy)
            }
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        let emy = (-y).exp();
        (1.0 - emy, emy)
    } else {
        let base = 1.0 - (2.0 - lambda) * y;
        if base <= 0.0 {
            (1.0, 0.0)
        } else {
            let inv_c = 1.0 / (2.0 - lambda);
            let x = 1.0 - base.powf(inv_c);
            let dxdy = base.powf(inv_c - 1.0);
            (x, dxdy)
        }
    }
}
use super::leaf::Leaf;
use super::leaves::{
    Ar1Leaf, Ar2Leaf, BetaLeaf, DiscreteUniformLeaf, DriftLeaf, EmaLeaf, FractionalDiffLeaf,
    GammaLeaf, HoltLeaf, IntermittentLeaf, LogNormalLeaf, MultiplicativeSeasonalLeaf,
    NegativeBinomialLeaf, OuLeaf, PoissonLeaf, RectifiedNormalLeaf, SeasonalEmaLeaf,
    SeasonalIntermittentLeaf, SkewNormalLeaf, StudentTLeaf, TweedieLeaf, YjWrappedLeaf,
    ZeroInflatedNegativeBinomialLeaf, ZeroInflatedPoissonLeaf,
};
use super::DistributionalForecaster;

/// Distributional forecaster returning a `GaussianMixture` per horizon.
///
/// Wraps three streaming leaves (EMA, drift, AR(1)) and mixes them by
/// cumulative one-step log-likelihood. Optionally adds:
///
/// * a damped-Holt (level + trend + damping) leaf via [`Self::with_holt`];
/// * an AR(2) leaf via [`Self::with_ar2`] — catches longer-memory
///   autocorrelation that AR(1) misses;
/// * a seasonal-EMA leaf via [`Self::with_seasonal`] — pass the period
///   explicitly (no auto-detection).
///
/// All three are opt-in. Empirical M5-retail benchmarking showed the
/// mature Holt formulation actively *hurting* the mixture by default
/// (Holt's noisy trend estimate steals softmax weight from other leaves
/// on series with weak or no trend). Seasonal and AR(2) are cheap wins
/// on the panels where they apply, but the shell keeps them opt-in for
/// symmetry with Holt and to preserve the alpha-2 3-leaf default.
pub struct LaplaceForecaster {
    ema_alpha: f64,
    drift_alpha: f64,
    ar_alpha_mean: f64,
    holt: Option<(f64, f64, f64)>, // (alpha, beta, phi)
    ar2: Option<f64>,              // mean-EMA alpha
    seasonal_period: Option<usize>,
    seasonal_periods_multi: Vec<usize>,
    seasonal_alpha: f64,
    calibrate: bool,
    /// If true, calibration additionally fits per-horizon scale factors
    /// via periodic in-sample snapshots (α-14). Applied after the shared
    /// `calibration_scale`. Empty vector when unused.
    calibrate_per_h: bool,
    /// Number of horizons over which to fit per-h calibration. Defaults to
    /// 28 (matches the M5 competition horizon). Callers requesting forecasts
    /// beyond this horizon get the last saved `λ_h`.
    per_h_horizon: usize,
    /// Per-h scale factors fit in `fit()` when `calibrate_per_h` is on.
    calibration_scale_per_h: Vec<f64>,
    /// User-supplied Yeo-Johnson λ. Overrides `yj_auto` when both are set.
    yj_lambda: Option<f64>,
    /// If true, fit the Yeo-Johnson λ via MLE at the start of `fit()` and
    /// store it in `fitted_yj_lambda`.
    yj_auto: bool,
    /// If true, [`Self::init_leaves`] replaces the 3-leaf default with an
    /// expanded 7-leaf population — EMA at 3 rates, drift at 2, AR(1)
    /// mean-EMA at 2. The likelihood weighting picks the effective rate
    /// per series, imitating skaters' "Bayesian ensemble over a large
    /// candidate population" without adding new leaf families.
    use_populations: bool,
    /// If true, `init_leaves` swaps in a wider 15-leaf population — the α-7
    /// grid plus additional fast/slow pairs. Larger softmax pool at ~3×
    /// compute; helps only on panels with strongly heterogeneous dynamics.
    use_populations_wide: bool,
    /// Yeo-Johnson coordinate grid. If non-empty, `init_leaves` wraps every
    /// base leaf with `YjWrappedLeaf(inner, λ)` for each λ in the grid,
    /// turning the mixture into a `(leaf, λ)` softmax matrix. Skaters'
    /// original YJ recipe. Mutually exclusive with the single-λ paths
    /// (`with_yeo_johnson` / `with_yeo_johnson_mle`).
    yj_grid: Vec<f64>,
    /// If true, `fit()` inspects the training series' characteristics
    /// (`trend_strength`, `seasonality_strength`, `acf1`) and configures
    /// the opt-in toggles from the α-8 residual-slicing evidence: always
    /// add OU; add AR(2) if `acf1 > 0.4`; add seasonal(7) if
    /// `seasonality_strength > 0.15`; add fractional-diff if `acf1 > 0.5`.
    /// Does not enable Holt / populations / Yeo-Johnson (evidence-negative
    /// on M5). The user-configured toggles are respected — `auto()` only
    /// adds, never removes.
    use_auto: bool,
    /// Enable AID-driven leaf selection (in addition to `use_auto` rules).
    /// Only meaningful when the `postprocess` feature is on.
    use_aid: bool,
    /// Seasonal period used by `auto()`. Defaults to 7 (weekly). Set via
    /// [`Self::auto_with_seasonal_period`] for non-daily panels.
    auto_seasonal_period: usize,
    /// `(d, α_mean, α_diff)` for the fractional-differencing leaf. Adds
    /// a long-memory drift-like leaf.
    frac_diff: Option<(f64, f64, f64)>,
    /// `α_mean` for the OU mean-reversion leaf. Adds an explicit
    /// mean-reverting leaf parameterised by `θ = 1 − φ`.
    ou: Option<f64>,
    /// `α` for the Croston-flavored intermittent-demand leaf. Adds a
    /// demand-per-period leaf that handles zero-inflated series much
    /// better than level-EMAs (which get dragged toward 0 by the
    /// zero periods).
    intermittent: Option<f64>,
    /// `(period, α)` for the seasonal-Croston leaf. Adds a
    /// per-phase demand-EMA on top of the shared interval-EMA so
    /// intermittent series with weekly / periodic non-zero clusters
    /// (SKU weekend spikes) get the phase shape right.
    seasonal_intermittent: Option<(usize, f64)>,
    /// `α` for the Poisson-family count leaf.
    poisson: Option<f64>,
    /// `α` for the Negative-Binomial count leaf.
    neg_binomial: Option<f64>,
    /// `α` for the Log-Normal positive-multiplicative leaf.
    lognormal: Option<f64>,
    /// `α` for the Gamma positive-skewed continuous leaf.
    gamma: Option<f64>,
    /// `α` for the Rectified-Normal (hurdle) leaf.
    rectified_normal: Option<f64>,
    /// `α` for the Zero-Inflated Poisson leaf.
    zip: Option<f64>,
    /// `α` for the Zero-Inflated Negative-Binomial leaf.
    zinb: Option<f64>,
    /// `α` for the Student-t leaf.
    student_t: Option<f64>,
    /// `α` for the Beta leaf (bounded [0,1] data).
    beta: Option<f64>,
    /// `(α, p)` for the Tweedie leaf. `p ∈ (1, 2)`.
    tweedie: Option<(f64, f64)>,
    /// `α` for the Skew-Normal leaf.
    skew_normal: Option<f64>,
    /// Toggle for the Discrete-Uniform leaf (no hyperparameter).
    discrete_uniform: bool,
    /// When true, forecast means are clipped to `max(0, μ)` — the cheap
    /// "no-negative demand forecast" fix. Distribution std is left
    /// alone (so the 90% interval can still dip below zero — proper
    /// truncated-Gaussian output is deferred).
    non_negative: bool,
    /// `(period, α)` for the multiplicative seasonal-EMA leaf. Complements
    /// the additive `seasonal_period`; retail seasonality is often
    /// proportional (peak week = 3× baseline).
    seasonal_mult: Option<(usize, f64)>,
    /// Names of exogenous regressors to preregress `y` on via OLS before
    /// feeding residuals to the leaves. Empty = no preregression. See
    /// [`Self::with_exog_preregression`].
    exog_names: Vec<String>,
    /// Cached OLS result after `fit()` — used by `predict_with_exog` to
    /// add `β · X_future` back to the mixture mean.
    exog_ols: Option<OLSResult>,
    /// α-23 opt-in: synthesize an `is_stockout` binary column from AID's
    /// per-observation labels and add it to the exog preregression
    /// design matrix. Default off.
    use_stockout_indicator: bool,
    /// α-23 opt-in: trim the training window to start after the last
    /// AID-flagged `NewProduct` observation. Default off.
    trim_new_product_prefix: bool,

    leaves: Vec<Box<dyn Leaf + Send>>,
    cum_log_liks: Vec<f64>,
    n_obs: usize,

    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
    training_values: Vec<f64>,
    /// 1-step mixture std at each training step (transformed space if YJ
    /// is enabled, else original space). Used by [`Self::with_calibration`].
    predictive_stds: Vec<f64>,
    /// 1-step residuals `y_trans - mixture_mean_trans` in the space the
    /// leaves operate in. Kept alongside `predictive_stds` so the
    /// calibration quantile-match uses matched-space `|z|`.
    predictive_residuals_trans: Vec<f64>,
    /// Terminal scale factor: `1.0` when uncalibrated. Applied to every
    /// `GaussianMixture` component's std at forecast time — in transformed
    /// space (before Yeo-Johnson inverse-transform).
    calibration_scale: f64,
    /// The Yeo-Johnson λ actually used for this fit. `None` if YJ was
    /// disabled. Populated even when the user supplied a fixed λ, so
    /// downstream callers can inspect the transform.
    fitted_yj_lambda: Option<f64>,
    /// Observed range of training values in transformed space. Used to
    /// clamp forecast-time transformed-space means before applying the
    /// YJ inverse — the inverse's Jacobian explodes exponentially in the
    /// log branch when a leaf's h-step forecast extrapolates far beyond
    /// the training window. Empty when YJ is disabled.
    yj_trans_range: Option<(f64, f64)>,
    /// PR #1 of #180: opt-in terminal scale-mixture leaf that reshapes
    /// the softmax mixture's density from an averaged-Gaussians blend
    /// into a fixed-scale mixture of zero-mean Gaussians centered at the
    /// softmax mean. Ports skaters' `scale_mixture_leaf` — "model first,
    /// conform last". Enabled automatically by `.auto()`.
    terminal: Option<TerminalScaleMixture>,
    /// PR #2 of #180: fast-slow decomposition family. When enabled and
    /// `N > 200`, appends 12 candidates (6 fast trackers × 2 slow scales)
    /// to the softmax pool. Each candidate wraps a fast mean tracker
    /// with a slow-EWMA residual-variance spread. Enabled automatically
    /// by `.auto()`.
    fast_slow: bool,
}

impl LaplaceForecaster {
    /// Default 3-leaf shell: EMA α=0.2, drift α=0.1, AR(1) mean α=0.1;
    /// no Holt, no seasonal leaf.
    pub fn new() -> Self {
        Self::with_alphas(0.2, 0.1, 0.1)
    }

    pub fn with_alphas(ema_alpha: f64, drift_alpha: f64, ar_alpha_mean: f64) -> Self {
        Self {
            ema_alpha,
            drift_alpha,
            ar_alpha_mean,
            holt: None,
            ar2: None,
            seasonal_period: None,
            seasonal_periods_multi: Vec::new(),
            seasonal_alpha: 0.15,
            calibrate: false,
            calibrate_per_h: false,
            per_h_horizon: 28,
            calibration_scale_per_h: Vec::new(),
            yj_lambda: None,
            yj_auto: false,
            use_populations: false,
            use_populations_wide: false,
            yj_grid: Vec::new(),
            use_auto: false,
            use_aid: false,
            auto_seasonal_period: 7,
            frac_diff: None,
            ou: None,
            intermittent: None,
            seasonal_intermittent: None,
            poisson: None,
            neg_binomial: None,
            lognormal: None,
            gamma: None,
            rectified_normal: None,
            zip: None,
            zinb: None,
            student_t: None,
            beta: None,
            tweedie: None,
            skew_normal: None,
            discrete_uniform: false,
            non_negative: false,
            seasonal_mult: None,
            exog_names: Vec::new(),
            exog_ols: None,
            use_stockout_indicator: false,
            trim_new_product_prefix: false,
            leaves: Vec::new(),
            cum_log_liks: Vec::new(),
            n_obs: 0,
            fitted_values: Vec::new(),
            residuals: Vec::new(),
            training_values: Vec::new(),
            predictive_stds: Vec::new(),
            predictive_residuals_trans: Vec::new(),
            calibration_scale: 1.0,
            fitted_yj_lambda: None,
            yj_trans_range: None,
            terminal: None,
            fast_slow: false,
        }
    }

    /// Enable the fast-slow decomposition candidate family (PR #2 of #180).
    ///
    /// Appends 12 candidates to the softmax pool: 6 fast mean trackers
    /// × 2 slow-EWMA residual-variance scales. Ports skaters'
    /// "thinking fast and slow" pattern — the mean reacts to every
    /// observation, the residual scale drifts an order of magnitude
    /// more slowly, so short bursts of noise don't blow up the density.
    ///
    /// Gated on `N > 200` at `fit()` time so the slow EWMA has enough
    /// residuals to converge.
    ///
    /// Enabled automatically by `.auto()`.
    pub fn with_fast_slow(mut self) -> Self {
        self.fast_slow = true;
        self
    }

    /// Enable the terminal scale-mixture leaf (PR #1 of #180).
    ///
    /// Reshapes `forecast_dist` output from an averaged-Gaussians blend
    /// into a **5-component fixed-scale Gaussian mixture** centered at
    /// the softmax mean. Component scales `(0.7, 1.0, 1.6, 3.0, 6.0)` are
    /// fixed relative to a running residual σ (EWMA at rate `scale_alpha`,
    /// default 0.03); component weights are learned online by
    /// likelihood-EM (recency rate `gamma`, default 0.02).
    ///
    /// This is the "model first, conform last" pattern from
    /// [`microprediction/skaters`](https://github.com/microprediction/skaters):
    /// the softmax ensemble decides the *mean* forecast, and this leaf
    /// reshapes the *distribution* once at the top so heavy tails
    /// survive averaging.
    ///
    /// Enabled automatically by `.auto()`.
    pub fn with_terminal_scale_mixture(mut self) -> Self {
        self.terminal = Some(TerminalScaleMixture::new());
        self
    }

    /// Same as [`Self::with_terminal_scale_mixture`] but lets you tune
    /// the two rate parameters. Defaults are 0.03 and 0.02 — matches
    /// skaters' `laplace(..., scale_alpha=0.03)` default with
    /// EM `gamma=0.02`.
    pub fn with_terminal_scale_mixture_params(mut self, scale_alpha: f64, gamma: f64) -> Self {
        self.terminal = Some(TerminalScaleMixture::with_params(scale_alpha, gamma));
        self
    }

    /// Apply a Yeo-Johnson power transform with fixed λ before feeding
    /// observations to the leaves. Predictions are delta-method
    /// inverse-transformed back to original space at forecast time.
    /// Variance-stabilizes retail-style panels where the residual scale
    /// is proportional to level.
    pub fn with_yeo_johnson(mut self, lambda: f64) -> Self {
        self.yj_lambda = Some(lambda);
        self.yj_auto = false;
        self
    }

    /// Fit the Yeo-Johnson λ via MLE at the start of `fit()`. Uses the
    /// crate's [`crate::transform::yeo_johnson::yeo_johnson_lambda`]
    /// estimator (grid search over `[-2, 2]` at Δ=0.01, refined at
    /// Δ=0.001).
    pub fn with_yeo_johnson_mle(mut self) -> Self {
        self.yj_auto = true;
        self.yj_lambda = None;
        self
    }

    /// Add a fractional-differencing leaf with fractional order `d ∈
    /// (0.05, 0.95)`. Captures long-memory persistence that AR(1) / AR(2)
    /// miss. `alpha_mean` tracks the level; `alpha_diff` tracks the
    /// running fractional-diff step.
    pub fn with_fractional_diff(mut self, d: f64, alpha_mean: f64, alpha_diff: f64) -> Self {
        self.frac_diff = Some((d, alpha_mean, alpha_diff));
        self
    }

    /// Fractional-differencing leaf with defaults `d=0.4`, `α_mean=0.1`,
    /// `α_diff=0.1`.
    pub fn with_fractional_diff_defaults(self) -> Self {
        self.with_fractional_diff(0.4, 0.1, 0.1)
    }

    /// Add an Ornstein-Uhlenbeck mean-reversion leaf with the given
    /// mean-EMA rate. Behaves better than a mean-shifted AR(1) on
    /// bounded / mean-reverting series at longer horizons.
    pub fn with_ou(mut self, alpha_mean: f64) -> Self {
        self.ou = Some(alpha_mean);
        self
    }

    /// OU leaf with the default mean-EMA rate 0.1.
    pub fn with_ou_defaults(self) -> Self {
        self.with_ou(0.1)
    }

    /// Add a Croston-flavored intermittent-demand leaf that tracks demand
    /// size and inter-demand interval as separate EMAs. Handles
    /// zero-inflated series (SKU sales with many zero days) much better
    /// than level-EMAs. `α` clamped to `(0.001, 0.999)`.
    pub fn with_intermittent(mut self, alpha: f64) -> Self {
        self.intermittent = Some(alpha);
        self
    }

    /// Intermittent leaf with the default rate `α = 0.1` (Croston's classic
    /// value).
    pub fn with_intermittent_defaults(self) -> Self {
        self.with_intermittent(0.1)
    }

    /// Add a **seasonal-Croston** leaf that tracks per-phase demand-EMAs
    /// on top of a shared interval-EMA. Retail SKU data typically has
    /// non-zero clusters aligned to a period (weekend spikes on daily
    /// data). Classic Croston predicts a flat constant and misses the
    /// phase shape; this leaf captures it. `period < 2` is a no-op.
    pub fn with_seasonal_intermittent(mut self, period: usize, alpha: f64) -> Self {
        if period >= 2 {
            self.seasonal_intermittent = Some((period, alpha));
        }
        self
    }

    /// Seasonal-Croston with the default rate `α = 0.1`.
    pub fn with_seasonal_intermittent_defaults(self, period: usize) -> Self {
        self.with_seasonal_intermittent(period, 0.1)
    }

    /// Add a Poisson leaf — moment-matched Gaussian output for small
    /// count data with `variance ≈ mean`. See [`super::leaves::PoissonLeaf`].
    pub fn with_poisson(mut self, alpha: f64) -> Self {
        self.poisson = Some(alpha);
        self
    }

    /// Poisson leaf with `α = 0.1`.
    pub fn with_poisson_defaults(self) -> Self {
        self.with_poisson(0.1)
    }

    /// Add a Negative-Binomial leaf — moment-matched Gaussian output for
    /// overdispersed count data (retail-demand norm). Nests Poisson when
    /// observed variance ≤ mean.
    pub fn with_negative_binomial(mut self, alpha: f64) -> Self {
        self.neg_binomial = Some(alpha);
        self
    }

    /// Negative-Binomial leaf with `α = 0.05` (slow — retail dispersion
    /// estimates need more history than mean estimates).
    pub fn with_negative_binomial_defaults(self) -> Self {
        self.with_negative_binomial(0.05)
    }

    /// Add a Log-Normal leaf — moment-matched Gaussian output for positive
    /// multiplicative processes. Works on `ln(y + 1)` internally.
    pub fn with_lognormal(mut self, alpha: f64) -> Self {
        self.lognormal = Some(alpha);
        self
    }

    /// Log-Normal leaf with `α = 0.05`.
    pub fn with_lognormal_defaults(self) -> Self {
        self.with_lognormal(0.05)
    }

    /// Add a Gamma leaf — moment-matched Gaussian output for
    /// positive-skewed continuous data.
    pub fn with_gamma(mut self, alpha: f64) -> Self {
        self.gamma = Some(alpha);
        self
    }

    /// Gamma leaf with `α = 0.05`.
    pub fn with_gamma_defaults(self) -> Self {
        self.with_gamma(0.05)
    }

    /// Add a Rectified-Normal (hurdle) leaf — intermittent continuous
    /// demand modeled as `p_zero · 0 + (1 - p_zero) · N(μ, σ²)`.
    pub fn with_rectified_normal(mut self, alpha: f64) -> Self {
        self.rectified_normal = Some(alpha);
        self
    }

    /// Rectified-Normal leaf with `α = 0.1`.
    pub fn with_rectified_normal_defaults(self) -> Self {
        self.with_rectified_normal(0.1)
    }

    /// Add a Zero-Inflated Poisson (ZIP) leaf — hurdle model on Poisson
    /// for high-zero-fraction count series where the observed zero
    /// share exceeds Poisson's own zero probability.
    pub fn with_zip(mut self, alpha: f64) -> Self {
        self.zip = Some(alpha);
        self
    }

    /// ZIP leaf with `α = 0.1`.
    pub fn with_zip_defaults(self) -> Self {
        self.with_zip(0.1)
    }

    /// Add a Zero-Inflated Negative-Binomial (ZINB) leaf — hurdle on NB
    /// for overdispersed excess-zero counts (retail-SKU norm).
    pub fn with_zinb(mut self, alpha: f64) -> Self {
        self.zinb = Some(alpha);
        self
    }

    /// ZINB leaf with `α = 0.05` (slow — dispersion needs history).
    pub fn with_zinb_defaults(self) -> Self {
        self.with_zinb(0.05)
    }

    /// Add a Student-t leaf — heavy-tailed continuous, softmax weighting
    /// then sees plausible density around outliers. `ν` (degrees of
    /// freedom) is estimated via kurtosis when N ≥ 50.
    pub fn with_student_t(mut self, alpha: f64) -> Self {
        self.student_t = Some(alpha);
        self
    }

    /// Student-t leaf with `α = 0.05`.
    pub fn with_student_t_defaults(self) -> Self {
        self.with_student_t(0.05)
    }

    /// Add a Beta leaf for bounded `[0, 1]` data (rates, proportions,
    /// service levels, conversion rates). Observations outside are
    /// clamped.
    pub fn with_beta(mut self, alpha: f64) -> Self {
        self.beta = Some(alpha);
        self
    }

    /// Beta leaf with `α = 0.05`.
    pub fn with_beta_defaults(self) -> Self {
        self.with_beta(0.05)
    }

    /// Add a Tweedie leaf — compound Poisson-gamma for aggregate retail
    /// (SKU × store × week) with point mass at zero + positive continuous
    /// branch + overdispersion. `p ∈ (1, 2)` interpolates between
    /// Poisson (p=1) and Gamma (p=2). Values outside are clamped.
    pub fn with_tweedie(mut self, alpha: f64, p: f64) -> Self {
        self.tweedie = Some((alpha, p));
        self
    }

    /// Tweedie leaf with the canonical retail-aggregate `α = 0.05, p = 1.5`.
    pub fn with_tweedie_defaults(self) -> Self {
        self.with_tweedie(0.05, 1.5)
    }

    /// Add a Skew-Normal leaf — asymmetric continuous data where YJ/log
    /// doesn't fully symmetrize. Skewness estimated via sample M3 when
    /// `N >= 30`; otherwise treated as Gaussian.
    pub fn with_skew_normal(mut self, alpha: f64) -> Self {
        self.skew_normal = Some(alpha);
        self
    }

    /// Skew-Normal leaf with `α = 0.05`.
    pub fn with_skew_normal_defaults(self) -> Self {
        self.with_skew_normal(0.05)
    }

    /// Add a Discrete-Uniform leaf for bounded small-count series
    /// `{0, 1, ..., K}`. `K` inferred as `max(observed)`. No
    /// hyperparameter.
    pub fn with_discrete_uniform(mut self) -> Self {
        self.discrete_uniform = true;
        self
    }

    /// Clip forecast component means to `max(0, μ)` at prediction time.
    /// The cheap "no-negative demand forecast" fix. Distribution std is
    /// left alone (the 90% interval can still dip below 0); proper
    /// truncated-Gaussian output is deferred.
    pub fn non_negative(mut self) -> Self {
        self.non_negative = true;
        self
    }

    /// Add a **multiplicative** seasonal-EMA leaf with the caller-supplied
    /// period. Tracks per-phase multipliers on a shared level (retail
    /// seasonality is often proportional — peak week = 3× baseline, not
    /// baseline + 5). Composes with the additive
    /// [`Self::with_seasonal`] — mixture picks whichever fits the data
    /// better per series. `period < 2` is a no-op.
    pub fn with_seasonal_multiplicative(mut self, period: usize, alpha: f64) -> Self {
        if period >= 2 {
            self.seasonal_mult = Some((period, alpha));
        }
        self
    }

    /// Multiplicative seasonal-EMA with the default rate `α = 0.15`.
    pub fn with_seasonal_multiplicative_defaults(self, period: usize) -> Self {
        self.with_seasonal_multiplicative(period, 0.15)
    }

    /// Preregress `y` on the named regressors via OLS at `fit()` time,
    /// then feed the residuals `y - Xβ` to the leaves. The OLS intercept
    /// and `β · X_future` are added back to the mixture mean when the
    /// caller uses [`Self::predict_with_exog`].
    ///
    /// Regressor names must exist in `TimeSeries::all_regressors()`
    /// (`TimeSeries::with_calendar(...)` on construction). Unknown names
    /// cause `fit()` to error.
    ///
    /// Standard [`Self::predict`] returns the residual-space mixture
    /// only. To get the level forecast, use [`Self::predict_with_exog`]
    /// with the future regressor values. Requires the `postprocess`
    /// feature for the OLS solver.
    pub fn with_exog_preregression(mut self, names: &[&str]) -> Self {
        self.exog_names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// α-23 opt-in: at `fit()` time, run the AID classifier on the training
    /// values and synthesize a binary `__aid_stockout` column marking
    /// AID-flagged stockout observations. That column is added to the
    /// exog preregression design matrix — the OLS coefficient captures
    /// the mean demand shift during stockout periods. **Default off.**
    ///
    /// Requires that `.with_exog_preregression(...)` is also called
    /// (the synthesized column joins the exog set). Requires the
    /// `postprocess` feature (for AID).
    pub fn with_stockout_indicator(mut self) -> Self {
        self.use_stockout_indicator = true;
        self
    }

    /// α-23 opt-in: at `fit()` time, run the AID classifier and trim the
    /// training window to start after the last observation flagged as
    /// `NewProduct`. Reasoning: the new-product lifecycle phase is a
    /// different regime (ramp-up, no equilibrium) that pollutes the
    /// leaves' state. **Default off.**
    ///
    /// If AID doesn't flag any `NewProduct` observations (or the flag
    /// is at the very end), no trimming happens.
    pub fn trim_new_product_prefix(mut self) -> Self {
        self.trim_new_product_prefix = true;
        self
    }

    /// Level-space point forecast for callers that used
    /// [`Self::with_exog_preregression`]. Requires the future values of
    /// every named regressor (and, if
    /// [`Self::with_stockout_indicator`] was set, the future
    /// `__aid_stockout` column). Returns
    /// `mixture_mean_residual + β · X_future` per horizon.
    ///
    /// When called without any exog preregression having been configured,
    /// this is equivalent to [`Self::predict`].
    pub fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }
        let mixtures = self.forecast_dist(horizon)?;
        let points: Vec<f64> = mixtures.iter().map(|m| m.mean()).collect();
        match &self.exog_ols {
            None => Ok(Forecast::from_values(points)),
            Some(ols) => {
                for name in &ols.regressor_names {
                    let col = future_regressors.get(name).ok_or_else(|| {
                        ForecastError::InvalidParameter(format!(
                            "predict_with_exog: missing future regressor `{name}`"
                        ))
                    })?;
                    if col.len() != horizon {
                        return Err(ForecastError::InvalidParameter(format!(
                            "predict_with_exog: future `{name}` length {} != horizon {}",
                            col.len(),
                            horizon
                        )));
                    }
                }
                let level_shift = ols.predict(future_regressors)?;
                let level_points: Vec<f64> = points
                    .iter()
                    .zip(level_shift.iter())
                    .map(|(p, s)| p + s)
                    .collect();
                Ok(Forecast::from_values(level_points))
            }
        }
    }

    /// Replace the 3-leaf default set (one EMA / drift / AR(1) each) with
    /// an expanded 7-leaf population that hyperparameter-sweeps the same
    /// families:
    ///
    /// * `EMA` at α ∈ {0.05, 0.2, 0.5} (slow / medium / fast level tracking)
    /// * `Drift` at α ∈ {0.05, 0.15}
    /// * `AR(1)` mean-EMA at α ∈ {0.05, 0.15}
    ///
    /// The softmax-over-cumulative-log-lik weighting picks the effective
    /// rate per series. Composes freely with `with_holt` / `with_ar2` /
    /// `with_seasonal` — those still add their own opt-in leaves on top.
    /// Adds compute proportional to the leaf count (roughly 2.3×).
    pub fn with_populations(mut self) -> Self {
        self.use_populations = true;
        self
    }

    /// Wider hyperparameter population (15 leaves): EMA at 5 rates, Drift
    /// at 3, AR(1) at 3, plus explicit "fast/slow two-systems" EMA pairs
    /// at extreme rates (α=0.02 slow / α=0.60 fast). Same principle as
    /// [`Self::with_populations`], larger softmax pool at ~3× compute.
    pub fn with_populations_wide(mut self) -> Self {
        self.use_populations_wide = true;
        self
    }

    /// Yeo-Johnson coordinate grid — wraps every base leaf with each λ
    /// in `lambdas`, turning the mixture into a `(leaf, λ)` softmax
    /// matrix. Skaters' original YJ recipe (α-6's single-λ path was a
    /// simplification). Compute scales linearly with grid size; typical
    /// grids are `{0.0, 0.5, 1.0, 1.5}` (4×). Mutually exclusive with
    /// the single-λ paths — passing an empty grid is a no-op.
    pub fn with_yeo_johnson_grid(mut self, lambdas: &[f64]) -> Self {
        self.yj_grid = lambdas.iter().copied().filter(|l| l.is_finite()).collect();
        if !self.yj_grid.is_empty() {
            self.yj_lambda = None;
            self.yj_auto = false;
        }
        self
    }

    /// Enable the per-series meta-selector. At `fit()` time, inspect the
    /// training series' characteristics and add opt-in leaves based on
    /// the α-8 residual-slicing evidence:
    ///
    /// * OU is always added (best single-leaf logpdf across all configs);
    /// * AR(2) is added when `|acf1| > 0.4` (its best segment);
    /// * seasonal-EMA at the auto period is added when the phase-mean R² > 0.15;
    /// * fractional-diff is added when `|acf1| > 0.5`.
    ///
    /// Holt / populations / Yeo-Johnson are NOT added (evidence-negative
    /// on M5). Composes with the explicit `with_*` builders — auto only
    /// adds leaves, never removes.
    pub fn auto(mut self) -> Self {
        self.use_auto = true;
        // PR #1 of #180: terminal scale-mixture leaf — reshape the
        // predictive density once at the top. Cheap, meaningful LL win.
        if self.terminal.is_none() {
            self.terminal = Some(TerminalScaleMixture::new());
        }
        // PR #2 of #180: fast-slow family stays opt-in via
        // `.with_fast_slow()`. On M5 discrete counts the terminal
        // scale-mixture already captures residual spread; the fast-slow
        // trackers dilute the softmax without a meaningful LL win
        // (+0.008 nats on M5 100-series bakeoff, +0.013 CRPS regression,
        // 2× fit time). On continuous FRED-style data they help more —
        // hence the builder is available but off by default.
        self
    }

    /// AID-driven auto-selector: run the `anofox-regression` AID demand
    /// classifier on the training values at `fit()` time and enable the
    /// distribution-family leaf that matches the fitted family. Replaces
    /// the hand-tuned rules in [`Self::auto`] with a statistically-derived
    /// choice.
    ///
    /// Family → leaf mapping:
    ///
    /// * `Poisson`, `Geometric` → [`super::leaves::PoissonLeaf`]
    /// * `NegativeBinomial` → [`super::leaves::NegativeBinomialLeaf`]
    /// * `LogNormal` → [`super::leaves::LogNormalLeaf`]
    /// * `Gamma` → [`super::leaves::GammaLeaf`]
    /// * `RectifiedNormal` → [`super::leaves::RectifiedNormalLeaf`]
    /// * `Normal` → falls through to [`Self::auto`]'s rule set
    ///
    /// Any AID-detected count / positive family also enables
    /// [`Self::non_negative`] on the output.
    ///
    /// Composes with explicit `with_*` builders. Requires the
    /// `postprocess` feature (default).
    #[cfg(feature = "postprocess")]
    pub fn auto_aid(mut self) -> Self {
        self.use_auto = true;
        self.use_aid = true;
        if self.terminal.is_none() {
            self.terminal = Some(TerminalScaleMixture::new());
        }
        self
    }

    /// Override the seasonal period used by [`Self::auto`] (default 7,
    /// weekly). Set to 12 for monthly, 24 for hourly-with-daily, etc.
    pub fn auto_with_seasonal_period(mut self, period: usize) -> Self {
        self.auto_seasonal_period = period.max(2);
        self
    }

    /// The Yeo-Johnson λ actually used for this fit — `None` if YJ was
    /// disabled or the model hasn't been fit yet.
    pub fn yeo_johnson_lambda(&self) -> Option<f64> {
        self.fitted_yj_lambda
    }

    /// Enable the terminal calibration step. After leaf-training, a single
    /// scale factor `λ = std(residuals) / mean(1-step mixture std)` is
    /// computed and applied to every mixture at forecast time. This is a
    /// method-of-moments version of the "model first, conform last"
    /// scheme in [`microprediction/skaters`](https://github.com/microprediction/skaters):
    /// the likelihood weights fit the shape, the terminal scale fixes the
    /// spread. The result is honest ~90% coverage at 90% target, at the
    /// cost of one extra pass over the training vector at fit time.
    ///
    /// Applies uniformly across horizons — the underlying leaves already
    /// scale std by `√h`, so a scalar terminal is horizon-invariant under
    /// the current shell.
    pub fn with_calibration(mut self) -> Self {
        self.calibrate = true;
        self
    }

    /// Enable per-horizon calibration on top of the shared quantile-match.
    /// During `fit()`, save the mixture at periodic snapshots; after
    /// training, fit a per-h scale factor `λ_h` via quantile matching on
    /// `|residual_h / σ_h|` for each horizon `h ∈ 1..=horizon_max`.
    /// Applied multiplicatively with the shared scalar at forecast time.
    /// Requires `with_calibration()` to also be set.
    pub fn with_per_horizon_calibration(mut self, horizon_max: usize) -> Self {
        self.calibrate_per_h = true;
        self.per_h_horizon = horizon_max.max(1);
        self.calibrate = true; // per-h needs the shared machinery too
        self
    }

    /// Add a damped-Holt (level + trend + damping) leaf. Sensible defaults
    /// via [`Self::with_holt_defaults`]. `phi = 1.0` gives pure Holt;
    /// `phi ∈ (0.5, 1.0)` damps the trend. All params clamped by the leaf.
    pub fn with_holt(mut self, alpha: f64, beta: f64, phi: f64) -> Self {
        self.holt = Some((alpha, beta, phi));
        self
    }

    /// Add a damped-Holt leaf with defaults α=0.3 β=0.1 φ=0.98.
    pub fn with_holt_defaults(self) -> Self {
        self.with_holt(0.3, 0.1, 0.98)
    }

    /// Add an AR(2) leaf that solves the 2×2 normal equations online.
    /// `alpha_mean` is the EMA rate for the tracking mean (defaults via
    /// [`Self::with_ar2_defaults`]).
    pub fn with_ar2(mut self, alpha_mean: f64) -> Self {
        self.ar2 = Some(alpha_mean);
        self
    }

    /// Add an AR(2) leaf with the default mean-EMA rate 0.1.
    pub fn with_ar2_defaults(self) -> Self {
        self.with_ar2(0.1)
    }

    /// Add a seasonal-EMA leaf with the caller-supplied period. A period
    /// of 0 or 1 is treated as "no seasonal leaf" — no runtime error.
    pub fn with_seasonal(mut self, period: usize) -> Self {
        if period >= 2 {
            self.seasonal_period = Some(period);
        }
        self
    }

    /// Add multiple seasonal-EMA leaves, one per period in `periods`.
    /// Composes with [`Self::with_seasonal`] (the single-period leaf) — both
    /// families can be set simultaneously. Periods `< 2` are silently
    /// dropped. Useful for panels with multiple periodicities (e.g. daily
    /// data with weekly + annual seasonality → `&[7, 365]`).
    pub fn with_seasonal_multi(mut self, periods: &[usize]) -> Self {
        self.seasonal_periods_multi = periods.iter().copied().filter(|p| *p >= 2).collect();
        self
    }

    /// Override the smoothing rate for the seasonal-EMA leaf. Only meaningful
    /// after `with_seasonal(period)` has been called. Clamped by the leaf.
    pub fn seasonal_alpha(mut self, alpha: f64) -> Self {
        self.seasonal_alpha = alpha;
        self
    }

    /// Build a fresh copy of the base leaf set (respecting user toggles).
    /// Used both for the single-shell path and per-λ in the YJ coord grid.
    fn build_base_leaves(&self) -> Vec<Box<dyn Leaf + Send>> {
        let mut leaves: Vec<Box<dyn Leaf + Send>> = if self.use_populations_wide {
            vec![
                Box::new(EmaLeaf::new(0.02)),
                Box::new(EmaLeaf::new(0.10)),
                Box::new(EmaLeaf::new(0.25)),
                Box::new(EmaLeaf::new(0.45)),
                Box::new(EmaLeaf::new(0.60)),
                Box::new(DriftLeaf::new(0.03)),
                Box::new(DriftLeaf::new(0.10)),
                Box::new(DriftLeaf::new(0.25)),
                Box::new(Ar1Leaf::new(0.03)),
                Box::new(Ar1Leaf::new(0.10)),
                Box::new(Ar1Leaf::new(0.25)),
            ]
        } else if self.use_populations {
            vec![
                Box::new(EmaLeaf::new(0.05)),
                Box::new(EmaLeaf::new(0.20)),
                Box::new(EmaLeaf::new(0.50)),
                Box::new(DriftLeaf::new(0.05)),
                Box::new(DriftLeaf::new(0.15)),
                Box::new(Ar1Leaf::new(0.05)),
                Box::new(Ar1Leaf::new(0.15)),
            ]
        } else {
            vec![
                Box::new(EmaLeaf::new(self.ema_alpha)),
                Box::new(DriftLeaf::new(self.drift_alpha)),
                Box::new(Ar1Leaf::new(self.ar_alpha_mean)),
            ]
        };
        if let Some((a, b, phi)) = self.holt {
            leaves.push(Box::new(HoltLeaf::new(a, b, phi)));
        }
        if let Some(a) = self.ar2 {
            leaves.push(Box::new(Ar2Leaf::new(a)));
        }
        if let Some((d, am, ad)) = self.frac_diff {
            leaves.push(Box::new(FractionalDiffLeaf::new(d, am, ad)));
        }
        if let Some(a) = self.ou {
            leaves.push(Box::new(OuLeaf::new(a)));
        }
        if let Some(a) = self.intermittent {
            leaves.push(Box::new(IntermittentLeaf::new(a)));
        }
        if let Some((p, a)) = self.seasonal_intermittent {
            leaves.push(Box::new(SeasonalIntermittentLeaf::new(p, a)));
        }
        if let Some(a) = self.poisson {
            leaves.push(Box::new(PoissonLeaf::new(a)));
        }
        if let Some(a) = self.neg_binomial {
            leaves.push(Box::new(NegativeBinomialLeaf::new(a)));
        }
        if let Some(a) = self.lognormal {
            leaves.push(Box::new(LogNormalLeaf::new(a)));
        }
        if let Some(a) = self.gamma {
            leaves.push(Box::new(GammaLeaf::new(a)));
        }
        if let Some(a) = self.rectified_normal {
            leaves.push(Box::new(RectifiedNormalLeaf::new(a)));
        }
        if let Some(a) = self.zip {
            leaves.push(Box::new(ZeroInflatedPoissonLeaf::new(a)));
        }
        if let Some(a) = self.zinb {
            leaves.push(Box::new(ZeroInflatedNegativeBinomialLeaf::new(a)));
        }
        if let Some(a) = self.student_t {
            leaves.push(Box::new(StudentTLeaf::new(a)));
        }
        if let Some(a) = self.beta {
            leaves.push(Box::new(BetaLeaf::new(a)));
        }
        if let Some((a, p)) = self.tweedie {
            leaves.push(Box::new(TweedieLeaf::new(a, p)));
        }
        if let Some(a) = self.skew_normal {
            leaves.push(Box::new(SkewNormalLeaf::new(a)));
        }
        if self.discrete_uniform {
            leaves.push(Box::new(DiscreteUniformLeaf::new()));
        }
        if let Some(p) = self.seasonal_period {
            leaves.push(Box::new(SeasonalEmaLeaf::new(p, self.seasonal_alpha)));
        }
        for &p in &self.seasonal_periods_multi {
            leaves.push(Box::new(SeasonalEmaLeaf::new(p, self.seasonal_alpha)));
        }
        if let Some((p, a)) = self.seasonal_mult {
            leaves.push(Box::new(MultiplicativeSeasonalLeaf::new(p, a)));
        }
        leaves
    }

    fn init_leaves(&mut self, n_train: usize) {
        let leaves = self.build_base_leaves();
        let mut leaves = if !self.yj_grid.is_empty() {
            // Coordinate grid: build one fresh base set per λ, wrap each
            // element with YjWrappedLeaf(inner, λ). Every (leaf, λ)
            // becomes its own softmax candidate.
            let mut wrapped: Vec<Box<dyn Leaf + Send>> =
                Vec::with_capacity(leaves.len() * self.yj_grid.len());
            for lam in self.yj_grid.clone() {
                let per_lambda = self.build_base_leaves();
                for l in per_lambda {
                    wrapped.push(Box::new(YjWrappedLeaf::new(l, lam)));
                }
            }
            wrapped
        } else {
            leaves
        };
        // PR #2 of #180: fast-slow family. Only append when we have
        // enough training obs for the slow EWMA to converge (>200).
        // 6 fast trackers × 2 slow scales = 12 candidates. Each
        // candidate is (fast tracker) + slow-EWMA residual variance.
        if self.fast_slow && n_train > 200 {
            for slow_alpha in [0.02, 0.05] {
                let trackers: [Box<dyn Leaf + Send>; 6] = [
                    Box::new(EmaLeaf::new(0.3)),
                    Box::new(EmaLeaf::new(0.5)),
                    Box::new(HoltLeaf::new(0.4, 0.2, 0.98)),
                    Box::new(Ar1Leaf::new(0.3)),
                    Box::new(DriftLeaf::new(0.05)),
                    // 6th tracker: EMA α=0.95 as a random-walk-like tracker
                    // (skaters uses `difference()` here; the level EMA at
                    // near-1 α is effectively naive-with-noise, matching
                    // "the mean is roughly the last obs").
                    Box::new(EmaLeaf::new(0.95)),
                ];
                for t in trackers {
                    leaves.push(Box::new(SlowStandardizeWrapper::new(t, slow_alpha)));
                }
            }
        }
        // Clear the old redundant setter path — keep `leaves` mut for
        // the trailing existing logic (self.cum_log_liks / self.leaves).
        let _ = &mut leaves;
        self.cum_log_liks = vec![0.0; leaves.len()];
        self.leaves = leaves;
    }

    fn weights(&self) -> Vec<f64> {
        softmax(&self.cum_log_liks)
    }

    fn per_leaf_horizons(&self, horizon: usize) -> Vec<Vec<super::dist::Gaussian>> {
        self.leaves.iter().map(|l| l.predict(horizon)).collect()
    }
}

impl Default for LaplaceForecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for LaplaceForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let raw = series.primary_values();
        if raw.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "LaplaceForecaster requires at least one observation".into(),
            ));
        }

        // Reset exog state so a re-fit doesn't reuse the previous OLS.
        self.exog_ols = None;

        // α-23: Run AID once at the top when any AID-driven pre-step
        // (trim NewProduct, stockout indicator) is requested. Cached
        // labels are consumed by the two branches below. Behind the
        // `postprocess` feature.
        #[cfg(feature = "postprocess")]
        let aid_labels: Option<Vec<crate::validation::aid::AidAnomalyLabel>> = {
            if self.trim_new_product_prefix || self.use_stockout_indicator {
                use crate::validation::aid::AidAnalyzer;
                let result = AidAnalyzer::new().analyze(raw);
                Some(result.features().labels)
            } else {
                None
            }
        };

        // α-23 opt-in: trim leading NewProduct observations. `train_start`
        // is the offset into `raw` where the leaf-observed training
        // sub-window begins. Default 0.
        let mut train_start = 0usize;
        #[cfg(feature = "postprocess")]
        if self.trim_new_product_prefix {
            if let Some(labels) = &aid_labels {
                let last_np = labels
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, l)| matches!(l, crate::validation::aid::AidAnomalyLabel::NewProduct))
                    .map(|(i, _)| i);
                if let Some(idx) = last_np {
                    // Never trim to fewer than 12 obs — the leaves need
                    // *some* data to warm up.
                    let candidate = idx + 1;
                    if candidate + 12 <= raw.len() {
                        train_start = candidate;
                    }
                }
            }
        }
        let raw_train: &[f64] = &raw[train_start..];

        // α-23: OLS preregression on named exog regressors + (optionally)
        // an AID-derived is_stockout column. Residuals `y - Xβ` are what
        // the leaves observe; the OLS is cached for `predict_with_exog`.
        let leaf_values: Vec<f64> = if !self.exog_names.is_empty() {
            let mut regressors: HashMap<String, Vec<f64>> = HashMap::new();
            for name in &self.exog_names {
                let col = series.regressor(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "LaplaceForecaster: exog regressor `{name}` not in TimeSeries"
                    ))
                })?;
                if col.len() != raw.len() {
                    return Err(ForecastError::InvalidParameter(format!(
                        "LaplaceForecaster: exog regressor `{name}` length {} != series {}",
                        col.len(),
                        raw.len()
                    )));
                }
                regressors.insert(name.clone(), col[train_start..].to_vec());
            }
            #[cfg(feature = "postprocess")]
            if self.use_stockout_indicator {
                if let Some(labels) = &aid_labels {
                    let col: Vec<f64> = labels[train_start..]
                        .iter()
                        .map(|l| {
                            if matches!(l, crate::validation::aid::AidAnomalyLabel::Stockout) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    regressors.insert("__aid_stockout".to_string(), col);
                }
            }
            let ols = ols_fit(raw_train, &regressors)?;
            let fitted = ols.predict(&regressors)?;
            let residuals: Vec<f64> = raw_train
                .iter()
                .zip(fitted.iter())
                .map(|(y, f)| y - f)
                .collect();
            self.exog_ols = Some(ols);
            residuals
        } else {
            raw_train.to_vec()
        };

        // Existing downstream code reads a `values` slice; alias to the
        // (potentially trimmed & residual) `leaf_values` we just built.
        let values: &[f64] = &leaf_values;

        // AID-driven family selection (α-21). Runs before the classical
        // `use_auto` rules so those only fill in gaps AID didn't cover.
        // The AID call is behind the `postprocess` feature; when off, this
        // block compiles out and `use_aid` stays `false`.
        #[cfg(feature = "postprocess")]
        if self.use_aid {
            use crate::validation::aid::AidAnalyzer;
            use anofox_regression::solvers::DemandDistribution;
            let aid_result = AidAnalyzer::new().analyze(values);
            let summary = aid_result.summary();
            let mut count_or_positive = false;
            // α-24: When AID picks Poisson/NB AND the observed zero
            // fraction exceeds what that distribution would predict,
            // route to the zero-inflated variant instead. Threshold:
            // observed zero fraction > 0.5 → ZIP/ZINB.
            let excess_zeros = summary.zero_proportion > 0.5;
            match summary.distribution {
                DemandDistribution::Poisson | DemandDistribution::Geometric => {
                    if excess_zeros {
                        if self.zip.is_none() {
                            self.zip = Some(0.1);
                        }
                    } else if self.poisson.is_none() {
                        self.poisson = Some(0.1);
                    }
                    count_or_positive = true;
                }
                DemandDistribution::NegativeBinomial => {
                    if excess_zeros {
                        if self.zinb.is_none() {
                            self.zinb = Some(0.05);
                        }
                    } else if self.neg_binomial.is_none() {
                        self.neg_binomial = Some(0.05);
                    }
                    count_or_positive = true;
                }
                DemandDistribution::LogNormal => {
                    if self.lognormal.is_none() {
                        self.lognormal = Some(0.05);
                    }
                    count_or_positive = true;
                }
                DemandDistribution::Gamma => {
                    if self.gamma.is_none() {
                        self.gamma = Some(0.05);
                    }
                    count_or_positive = true;
                }
                DemandDistribution::RectifiedNormal => {
                    if self.rectified_normal.is_none() {
                        self.rectified_normal = Some(0.1);
                    }
                    count_or_positive = true;
                }
                DemandDistribution::Normal => {}
            }
            if count_or_positive {
                self.non_negative = true;
            }
        }

        // Auto-selector: inspect series characteristics before initialising
        // leaves and set the opt-in toggles from residual-slicing evidence.
        // User-configured toggles are respected — auto only adds.
        if self.use_auto {
            // α-27 fix #2: auto-detect the seasonal period when the user
            // hasn't set one explicitly. Falls back to `auto_seasonal_period`
            // (default 7) when no candidate has ACF > 0.35.
            let detected_period = detect_seasonal_period(values);
            let effective_period = detected_period.unwrap_or(self.auto_seasonal_period);

            let chars = auto_characteristics(values, effective_period);
            if self.ou.is_none() {
                self.ou = Some(0.1);
            }
            // Trending-guard: on trending series, `acf1` is inflated because
            // consecutive samples share the trend. Enabling AR(2) then pushes
            // its MoM estimator toward the unit-root boundary, and the
            // recursive h-step forecast diverges (M4-daily benchmark caught
            // a catastrophic mean-MAE blow-up). Skip AR(2) when trend
            // dominates.
            if chars.acf1 > 0.4 && chars.trend_strength < 0.5 && self.ar2.is_none() {
                self.ar2 = Some(0.1);
            }
            if chars.seasonality_strength > 0.15 && self.seasonal_period.is_none() {
                self.seasonal_period = Some(effective_period);
            }
            // α-27 fix #1: enable the multiplicative seasonal leaf when
            // seasonality is present AND series is strictly positive
            // (tourism, retail-aggregate — where the peak-trough pattern
            // is proportional to the level, not additive).
            if chars.seasonality_strength > 0.3
                && chars.all_positive
                && chars.mean_y > 0.0
                && self.seasonal_mult.is_none()
            {
                self.seasonal_mult = Some((effective_period, 0.15));
            }
            if chars.acf1 > 0.5 && chars.trend_strength < 0.5 && self.frac_diff.is_none() {
                self.frac_diff = Some((0.4, 0.1, 0.1));
            }
            // α-20 additions:
            // - Mid-trend series get Holt (was evidence-negative on full-M5
            //   only because it was applied to trend-free series; on the
            //   trend_strength ∈ [0.3, 0.7] slice it wins).
            // α-27 fix #3: use damped Holt (φ=0.9) instead of near-undamped
            // (φ=0.98). fev tourism/m4_yearly show classical damped-trend
            // wins big on long horizons — damping bends extrapolation.
            if chars.trend_strength >= 0.3 && chars.trend_strength <= 0.7 && self.holt.is_none() {
                self.holt = Some((0.3, 0.1, 0.9));
            }
            // α-27 fix #3b: strong-trend (>0.7) series also benefit from
            // damped Holt with more aggressive damping. Otherwise our
            // Drift leaf's linear extrapolation blows the tail on long
            // horizons (fev m4_yearly, m4_quarterly).
            if chars.trend_strength > 0.7 && self.holt.is_none() {
                self.holt = Some((0.2, 0.05, 0.85));
            }
            // - Zero-inflated seasonal series get the seasonal-Croston leaf.
            //   Retail SKU data with weekend spikes is the biggest lose
            //   segment on full M5; classic Croston misses the phase shape.
            if chars.zero_fraction > 0.3
                && chars.seasonality_strength > 0.10
                && self.seasonal_intermittent.is_none()
            {
                self.seasonal_intermittent = Some((effective_period, 0.1));
            }
            // - Purely intermittent (no phase signal) still gets classic
            //   Croston.
            if chars.zero_fraction > 0.4 && self.intermittent.is_none() {
                self.intermittent = Some(0.1);
            }
            // - Any auto-detected intermittency implies non-negative output.
            if chars.zero_fraction > 0.3 {
                self.non_negative = true;
            }
        }

        self.init_leaves(values.len());
        self.training_values = values.to_vec();
        self.fitted_values = Vec::with_capacity(values.len());
        self.residuals = Vec::with_capacity(values.len());
        self.predictive_stds = if self.calibrate {
            Vec::with_capacity(values.len())
        } else {
            Vec::new()
        };
        self.predictive_residuals_trans = if self.calibrate {
            Vec::with_capacity(values.len())
        } else {
            Vec::new()
        };
        self.calibration_scale = 1.0;
        self.n_obs = 0;

        // Resolve Yeo-Johnson λ. User-supplied wins over MLE; MLE happens
        // exactly once at fit start over the full training window. When
        // the coordinate grid is set, per-leaf `YjWrappedLeaf` handles
        // the transform — the shell-level path is disabled.
        self.fitted_yj_lambda = if !self.yj_grid.is_empty() {
            None
        } else if let Some(l) = self.yj_lambda {
            Some(l)
        } else if self.yj_auto {
            Some(yeo_johnson_lambda(values))
        } else {
            None
        };
        let yj = self.fitted_yj_lambda;
        // Cache the observed range of training values in transformed
        // space — used to clamp forecast-time means before the inverse.
        // Clamp to exact training range in transformed space. Any padding
        // lets the inverse extrapolate; on the log-branch (λ near 0) even
        // small extrapolation produces astronomical values.
        self.yj_trans_range = yj.map(|l| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &v in values {
                let t = yj_forward(v, l);
                if t.is_finite() {
                    lo = lo.min(t);
                    hi = hi.max(t);
                }
            }
            (lo, hi)
        });

        // Snapshot collection for per-horizon calibration: at periodic
        // intervals during fit, save the H-horizon mixture-mean/std so that
        // after fit we can quantile-match per h against known-future values.
        let snapshot_stride = (values.len() / 30).clamp(1, 200);
        let per_h_horizon = self.per_h_horizon;
        let mut per_h_snapshots: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();

        for (step, &y_orig) in values.iter().enumerate() {
            let y = match yj {
                Some(l) => yj_forward(y_orig, l),
                None => y_orig,
            };

            // Periodic snapshot: take before observing y at this step. Only
            // useful when the snapshot's H-step horizon fits inside training.
            if self.calibrate_per_h
                && step >= 30
                && step % snapshot_stride == 0
                && step + per_h_horizon <= values.len()
            {
                let weights_now = self.weights();
                let per_leaf_h: Vec<Vec<super::dist::Gaussian>> = self
                    .leaves
                    .iter()
                    .map(|l| l.predict(per_h_horizon))
                    .collect();
                let mixtures: Vec<(f64, f64)> = (0..per_h_horizon)
                    .map(|h| {
                        let m = blend_horizon(&weights_now, &per_leaf_h, h);
                        if m.is_empty() {
                            (0.0, 1.0)
                        } else {
                            (m.mean(), m.std())
                        }
                    })
                    .collect();
                per_h_snapshots.push((step, mixtures));
            }

            // 1-step predictions from each leaf, before observing y.
            let per_leaf: Vec<super::dist::Gaussian> =
                self.leaves.iter().map(|l| l.predict(1)[0]).collect();
            let weights = self.weights();

            let mixture =
                GaussianMixture::new(weights.iter().zip(per_leaf.iter()).map(|(w, g)| (*w, *g)));
            // Fitted / residuals: expose in ORIGINAL space so downstream
            // consumers (Explanation, tests, callers computing MAE) see
            // the same scale as the training values.
            let fitted_orig = if mixture.is_empty() {
                y_orig
            } else {
                let m_trans = mixture.mean();
                match yj {
                    Some(l) => yj_inverse_with_jac(m_trans, l).0,
                    None => m_trans,
                }
            };
            self.fitted_values.push(fitted_orig);
            self.residuals.push(y_orig - fitted_orig);
            if self.calibrate {
                // Calibration operates on transformed-space residuals (the
                // leaves' Gaussian assumption lives there); stash both the
                // transformed-space 1-step σ and the transformed-space
                // residual so quantile-match sees matched-space `|z|`.
                let (mu_trans, sigma_trans) = if mixture.is_empty() {
                    (y, 1.0)
                } else {
                    (mixture.mean(), mixture.std())
                };
                self.predictive_stds.push(sigma_trans);
                self.predictive_residuals_trans.push(y - mu_trans);
            }

            // Score each leaf on this y, then absorb.
            for (i, leaf) in self.leaves.iter_mut().enumerate() {
                let g = per_leaf[i];
                let lp = g.logpdf(y);
                if lp.is_finite() {
                    self.cum_log_liks[i] += lp;
                }
                leaf.observe(y);
            }
            // Terminal scale-mixture: absorb the residual (transformed
            // space) between the softmax mixture mean and y. This leaf
            // tracks the residual's own distribution independently of
            // the individual leaves' Gaussian assumptions.
            if let Some(t) = self.terminal.as_mut() {
                let mixture_mean = if mixture.is_empty() {
                    y
                } else {
                    mixture.mean()
                };
                t.observe(y - mixture_mean);
            }
            self.n_obs += 1;
        }

        // Terminal calibration — quantile matching on |z| = |residual / σ|.
        // A well-calibrated Gaussian mixture has P90(|z|) = 1.645; rescale
        // so that fires exactly. Directly targets the interval coverage
        // metric (unlike a MoM variance match, which is fooled by bounded
        // or heavy-tailed panels where variance already matches but the
        // tail shape doesn't).
        if self.calibrate
            && !self.predictive_residuals_trans.is_empty()
            && !self.predictive_stds.is_empty()
        {
            const TARGET_LEVEL: f64 = 0.90;
            const GAUSSIAN_Z_AT_90: f64 = 1.644_853_626_951_472_7; // Φ⁻¹(0.95)
            let mut zabs: Vec<f64> = self
                .predictive_residuals_trans
                .iter()
                .zip(self.predictive_stds.iter())
                .filter_map(|(r, s)| {
                    if *s > 1e-9 && s.is_finite() {
                        Some((r / s).abs())
                    } else {
                        None
                    }
                })
                .collect();
            if !zabs.is_empty() {
                zabs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((zabs.len() as f64 * TARGET_LEVEL).ceil() as usize)
                    .saturating_sub(1)
                    .min(zabs.len() - 1);
                let p90 = zabs[idx].max(1e-9);
                self.calibration_scale = p90 / GAUSSIAN_Z_AT_90;
            }
        }

        // Per-horizon calibration: for each h, quantile-match |z_h| =
        // |(y_{t+h} - predicted_mean_h) / predicted_std_h| against a
        // Gaussian's P90 = 1.645. When there aren't enough snapshots at
        // some h, fall back to `self.calibration_scale`.
        if self.calibrate_per_h && !per_h_snapshots.is_empty() {
            const TARGET_LEVEL: f64 = 0.90;
            const GAUSSIAN_Z_AT_90: f64 = 1.644_853_626_951_472_7;
            let mut per_h = Vec::with_capacity(per_h_horizon);
            for h in 1..=per_h_horizon {
                let mut zabs: Vec<f64> = per_h_snapshots
                    .iter()
                    .filter_map(|(step, mixtures)| {
                        let (mu_trans, sigma_trans) = mixtures[h - 1];
                        if !(sigma_trans > 1e-9 && sigma_trans.is_finite()) {
                            return None;
                        }
                        let target_idx = *step + h;
                        if target_idx >= values.len() {
                            return None;
                        }
                        let y_trans = match yj {
                            Some(l) => yj_forward(values[target_idx], l),
                            None => values[target_idx],
                        };
                        Some(((y_trans - mu_trans) / sigma_trans).abs())
                    })
                    .collect();
                if zabs.len() < 5 {
                    // Too few points for a stable per-h estimate; reuse
                    // the shared scalar so we don't over-fit noise.
                    per_h.push(self.calibration_scale);
                    continue;
                }
                zabs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((zabs.len() as f64 * TARGET_LEVEL).ceil() as usize)
                    .saturating_sub(1)
                    .min(zabs.len() - 1);
                let p90 = zabs[idx].max(1e-9);
                per_h.push(p90 / GAUSSIAN_Z_AT_90);
            }
            self.calibration_scale_per_h = per_h;
        }
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }
        let mixtures = self.forecast_dist(horizon)?;
        let points: Vec<f64> = mixtures.iter().map(|m| m.mean()).collect();
        Ok(Forecast::from_values(points))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if !(0.0..1.0).contains(&level) {
            return Err(ForecastError::InvalidParameter(format!(
                "confidence level must be in [0, 1), got {level}"
            )));
        }
        let mixtures = self.forecast_dist(horizon)?;
        let alpha = 1.0 - level;
        let lo_p = alpha / 2.0;
        let hi_p = 1.0 - alpha / 2.0;
        let points: Vec<f64> = mixtures.iter().map(|m| m.mean()).collect();
        let lower: Vec<f64> = mixtures.iter().map(|m| m.quantile(lo_p)).collect();
        let upper: Vec<f64> = mixtures.iter().map(|m| m.quantile(hi_p)).collect();
        Ok(Forecast::from_values_with_intervals(points, lower, upper))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        if self.fitted_values.is_empty() {
            None
        } else {
            Some(&self.fitted_values)
        }
    }

    fn residuals(&self) -> Option<&[f64]> {
        if self.residuals.is_empty() {
            None
        } else {
            Some(&self.residuals)
        }
    }

    fn training_values(&self) -> Result<&[f64]> {
        if self.training_values.is_empty() {
            Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            })
        } else {
            Ok(&self.training_values)
        }
    }

    fn name(&self) -> &str {
        "LaplaceForecaster"
    }

    fn explanation(&self) -> Result<Explanation> {
        <Self as Inspectable>::explanation(self)
    }
}

impl Inspectable for LaplaceForecaster {
    fn explanation(&self) -> Result<Explanation> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        let horizon = 8;
        let mixtures = self.forecast_dist(horizon)?;
        let weights = self.weights();
        let names = self.leaves.iter().map(|l| l.name().to_string()).collect();
        Ok(Explanation::Laplace(LaplaceExplanation {
            horizon_dists: mixtures,
            leaf_weights: weights,
            leaf_names: names,
            fitted_values: self.fitted_values.clone(),
            residuals: self.residuals.clone(),
        }))
    }
}

impl DistributionalForecaster for LaplaceForecaster {
    fn forecast_dist(&self, horizon: usize) -> Result<Vec<GaussianMixture>> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }
        let weights = self.weights();
        let per_leaf = self.per_leaf_horizons(horizon);
        let scale = self.calibration_scale;
        let per_h = &self.calibration_scale_per_h;
        let yj = self.fitted_yj_lambda;
        let trans_range = self.yj_trans_range;
        let non_negative = self.non_negative;
        Ok((0..horizon)
            .map(|h| {
                let m = blend_horizon(&weights, &per_leaf, h);
                // Terminal scale-mixture: replace the softmax blend's
                // shape with a fixed-scale mixture centered at its mean.
                // Mean-preserving; only reshapes the density.
                let m = if let Some(t) = self.terminal.as_ref() {
                    if t.n_obs() > 5 && !m.is_empty() {
                        t.predict_shifted(m.mean())
                    } else {
                        m
                    }
                } else {
                    m
                };
                let scale_h = per_h.get(h).copied().unwrap_or(scale);
                let components = m.components.into_iter().map(|(w, g)| {
                    let sigma_scaled = g.std * scale_h;
                    let (mut mean_out, sigma_out) = match yj {
                        Some(l) => {
                            let mean_trans = match trans_range {
                                Some((lo, hi)) => g.mean.clamp(lo, hi),
                                None => g.mean,
                            };
                            let (m_orig, jac) = yj_inverse_with_jac(mean_trans, l);
                            (m_orig, (sigma_scaled * jac.abs()).max(1e-9))
                        }
                        None => (g.mean, sigma_scaled),
                    };
                    if non_negative && mean_out < 0.0 {
                        mean_out = 0.0;
                    }
                    (w, super::dist::Gaussian::new(mean_out, sigma_out))
                });
                GaussianMixture::new(components)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use chrono::{Duration, TimeZone, Utc};

    fn ts_ar1(n: usize, phi: f64) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut vals = Vec::with_capacity(n);
        let mut y = 0.0;
        for i in 0..n {
            let eps = ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
            y = phi * y + eps;
            vals.push(y);
        }
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn fit_and_forecast_dist_returns_mixture_per_horizon() {
        let ts = ts_ar1(200, 0.6);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        let dists = f.forecast_dist(5).unwrap();
        assert_eq!(dists.len(), 5);
        for d in &dists {
            assert_eq!(d.components.len(), 3);
            let ws: f64 = d.components.iter().map(|(w, _)| w).sum();
            assert!((ws - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn predict_matches_mixture_means() {
        let ts = ts_ar1(150, 0.5);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        let dists = f.forecast_dist(3).unwrap();
        let fc = f.predict(3).unwrap();
        let means: Vec<f64> = dists.iter().map(|m| m.mean()).collect();
        assert_eq!(fc.primary(), means.as_slice());
    }

    #[test]
    fn predict_before_fit_errors() {
        let f = LaplaceForecaster::new();
        assert!(matches!(
            f.predict(1),
            Err(ForecastError::FitRequired { .. })
        ));
        assert!(matches!(
            f.forecast_dist(1),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[test]
    fn intervals_are_ordered() {
        let ts = ts_ar1(120, 0.4);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        let fc = f.predict_with_intervals(3, 0.90).unwrap();
        let lower = fc.lower_series(0).unwrap();
        let upper = fc.upper_series(0).unwrap();
        let point = fc.primary();
        for i in 0..3 {
            assert!(lower[i] <= point[i] && point[i] <= upper[i]);
        }
    }

    #[test]
    fn explanation_after_fit_matches_leaf_names() {
        let ts = ts_ar1(80, 0.5);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names, vec!["ema", "drift", "ar1"]);
                assert_eq!(e.leaf_weights.len(), 3);
                assert!(!e.fitted_values.is_empty());
                assert_eq!(e.fitted_values.len(), e.residuals.len());
                assert_eq!(e.horizon_dists.len(), 8);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    fn ts_seasonal(n: usize, period: usize) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let vals: Vec<f64> = (0..n)
            .map(|i| {
                10.0 * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin()
                    + 50.0
            })
            .collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn with_seasonal_adds_seasonal_leaf_and_helps_periodic_series() {
        let ts = ts_seasonal(240, 12);
        let mut plain = LaplaceForecaster::new();
        let mut seasonal = LaplaceForecaster::new().with_seasonal(12);
        plain.fit(&ts).unwrap();
        seasonal.fit(&ts).unwrap();

        match Inspectable::explanation(&seasonal).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names, vec!["ema", "drift", "ar1", "seasonal_ema"]);
                assert_eq!(e.leaf_weights.len(), 4);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }

        // On a pure periodic series the seasonal fitted residual should be
        // smaller than the plain fitted residual (mean absolute residual).
        let plain_mae: f64 = plain
            .residuals()
            .unwrap()
            .iter()
            .map(|r| r.abs())
            .sum::<f64>()
            / plain.residuals().unwrap().len() as f64;
        let seasonal_mae: f64 = seasonal
            .residuals()
            .unwrap()
            .iter()
            .map(|r| r.abs())
            .sum::<f64>()
            / seasonal.residuals().unwrap().len() as f64;
        assert!(
            seasonal_mae < plain_mae,
            "seasonal MAR ({}) should beat plain MAR ({}) on a pure periodic series",
            seasonal_mae,
            plain_mae
        );
    }

    fn ts_positive_multiplicative(n: usize) -> TimeSeries {
        // A positive series whose noise scales with level — the setting
        // Yeo-Johnson is designed to help.
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let level = 50.0 + 0.1 * i as f64;
            let noise = ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
            vals.push(level * (1.0 + 0.3 * noise));
        }
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn with_yeo_johnson_mle_finds_a_lambda_and_returns_original_scale() {
        let ts = ts_positive_multiplicative(300);
        let mut f = LaplaceForecaster::new().with_yeo_johnson_mle();
        f.fit(&ts).unwrap();
        let lambda = f.yeo_johnson_lambda().expect("YJ MLE should populate λ");
        assert!(
            lambda.is_finite() && (-2.0..=2.0).contains(&lambda),
            "λ out of expected range: {}",
            lambda
        );
        // Forecasts should come back in original scale (roughly around the
        // series' level, not the transformed sub-unit region).
        let dists = f.forecast_dist(3).unwrap();
        for d in &dists {
            let m = d.mean();
            assert!(
                m.is_finite() && m > 5.0,
                "point forecast {} out of original scale",
                m
            );
        }
    }

    #[test]
    fn with_yeo_johnson_fixed_lambda_is_recorded() {
        let ts = ts_ar1(200, 0.5);
        let mut f = LaplaceForecaster::new().with_yeo_johnson(0.5);
        f.fit(&ts).unwrap();
        assert_eq!(f.yeo_johnson_lambda(), Some(0.5));
    }

    #[test]
    fn with_fractional_diff_and_ou_add_leaves_in_expected_order() {
        let ts = ts_ar1(120, 0.4);
        let mut f = LaplaceForecaster::new()
            .with_fractional_diff_defaults()
            .with_ou_defaults();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names, vec!["ema", "drift", "ar1", "frac_diff", "ou"]);
                assert_eq!(e.leaf_weights.len(), 5);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn auto_on_strongly_seasonal_series_adds_seasonal_leaf() {
        let ts = ts_seasonal(240, 12);
        let mut f = LaplaceForecaster::new()
            .auto()
            .auto_with_seasonal_period(12);
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                // Always OU; strong seasonal → seasonal_ema; likely ar2 (sinusoidal has ACF > 0.4).
                assert!(
                    e.leaf_names.iter().any(|n| n == "ou"),
                    "OU should always be added: {:?}",
                    e.leaf_names
                );
                assert!(
                    e.leaf_names.iter().any(|n| n == "seasonal_ema"),
                    "seasonal_ema should be added: {:?}",
                    e.leaf_names
                );
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn auto_on_pure_ar1_adds_ar2_and_ou_but_not_seasonal() {
        let ts = ts_ar1(240, 0.7);
        let mut f = LaplaceForecaster::new().auto();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert!(
                    e.leaf_names.iter().any(|n| n == "ar2"),
                    "AR(2) should be added on high-ACF: {:?}",
                    e.leaf_names
                );
                assert!(
                    e.leaf_names.iter().any(|n| n == "ou"),
                    "OU should always be added: {:?}",
                    e.leaf_names
                );
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn auto_respects_explicit_user_toggles() {
        let ts = ts_ar1(200, 0.5);
        let mut f = LaplaceForecaster::new().auto().with_holt_defaults();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                // User asked for Holt — auto never removes.
                assert!(e.leaf_names.iter().any(|n| n == "holt_damped"));
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_populations_expands_leaf_count() {
        let ts = ts_ar1(120, 0.4);
        let mut f = LaplaceForecaster::new().with_populations();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names.len(), 7, "population set: {:?}", e.leaf_names);
                // Rate labels are the same as the singleton versions —
                // three EMAs, two Drifts, two AR(1)s.
                let counts = |name: &str| -> usize {
                    e.leaf_names.iter().filter(|n| n.as_str() == name).count()
                };
                assert_eq!(counts("ema"), 3);
                assert_eq!(counts("drift"), 2);
                assert_eq!(counts("ar1"), 2);
                assert!((e.leaf_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_populations_composes_with_seasonal_and_ar2() {
        let ts = ts_ar1(120, 0.4);
        let mut f = LaplaceForecaster::new()
            .with_populations()
            .with_seasonal(7)
            .with_ar2_defaults();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                // 7 population + 1 AR(2) + 1 seasonal = 9.
                assert_eq!(e.leaf_names.len(), 9);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_calibration_narrows_mixture_std_toward_residual_std() {
        // Very smooth series → predictive mixture std overestimates the true
        // residual std, so calibration scale should be < 1 and narrow the
        // returned mixture.
        let ts = ts_ar1(400, 0.1);
        let mut plain = LaplaceForecaster::new();
        let mut calibrated = LaplaceForecaster::new().with_calibration();
        plain.fit(&ts).unwrap();
        calibrated.fit(&ts).unwrap();

        let plain_dist = plain.forecast_dist(1).unwrap();
        let cal_dist = calibrated.forecast_dist(1).unwrap();
        assert!(
            cal_dist[0].std() < plain_dist[0].std() * 1.05,
            "calibrated std {} should be at or below plain std {}",
            cal_dist[0].std(),
            plain_dist[0].std()
        );
        // Calibration should have adjusted at all (test tolerance is
        // deliberately lax — smoother series produce smaller adjustments).
        assert!(
            (calibrated.calibration_scale - 1.0).abs() > 0.005,
            "expected non-trivial calibration scale, got {}",
            calibrated.calibration_scale
        );
    }

    #[test]
    fn with_ar2_adds_ar2_leaf() {
        let ts = ts_ar1(80, 0.5);
        let mut f = LaplaceForecaster::new().with_ar2_defaults();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names, vec!["ema", "drift", "ar1", "ar2"]);
                assert_eq!(e.leaf_weights.len(), 4);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_holt_adds_holt_leaf() {
        let ts = ts_ar1(80, 0.5);
        let mut f = LaplaceForecaster::new().with_holt_defaults();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names, vec!["ema", "drift", "ar1", "holt_damped"]);
                assert_eq!(e.leaf_weights.len(), 4);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_holt_and_seasonal_stack_in_expected_order() {
        let ts = ts_seasonal(240, 12);
        let mut f = LaplaceForecaster::new()
            .with_holt_defaults()
            .with_seasonal(12);
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => assert_eq!(
                e.leaf_names,
                vec!["ema", "drift", "ar1", "holt_damped", "seasonal_ema"]
            ),
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_seasonal_multi_adds_one_leaf_per_period() {
        let ts = ts_ar1(200, 0.4);
        let mut f = LaplaceForecaster::new().with_seasonal_multi(&[7, 30, 365]);
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names.len(), 6); // 3 base + 3 seasonal
                assert_eq!(
                    e.leaf_names
                        .iter()
                        .filter(|n| n.as_str() == "seasonal_ema")
                        .count(),
                    3
                );
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_seasonal_multi_drops_invalid_periods() {
        let ts = ts_ar1(100, 0.4);
        let mut f = LaplaceForecaster::new().with_seasonal_multi(&[0, 1, 7]);
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names.len(), 4); // 3 base + 1 valid seasonal
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn with_seasonal_period_lt_2_is_a_no_op() {
        let ts = ts_ar1(100, 0.4);
        let mut f = LaplaceForecaster::new().with_seasonal(1);
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => assert_eq!(e.leaf_names.len(), 3),
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn explanation_before_fit_errors() {
        let f = LaplaceForecaster::new();
        assert!(matches!(
            Inspectable::explanation(&f),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[cfg(feature = "postprocess")]
    #[test]
    fn auto_aid_predicts_finite_on_intermittent_data() {
        // Sparse count series (60% zeros, mean ≈ 0.6) — AID should
        // classify as intermittent count and the fit should succeed.
        let n = 200;
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let vals: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 2.0 } else { 0.0 }).collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        let ts = TimeSeries::univariate(stamps, vals).unwrap();
        let mut f = LaplaceForecaster::new().auto_aid();
        f.fit(&ts).unwrap();
        let fc = f.predict(10).unwrap();
        for v in fc.primary() {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }

    #[cfg(feature = "postprocess")]
    #[test]
    fn auto_aid_predicts_finite_on_normal_data() {
        let n = 200;
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let vals: Vec<f64> = (0..n)
            .map(|i| 50.0 + ((i as f64 * 0.1).sin() * 5.0))
            .collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        let ts = TimeSeries::univariate(stamps, vals).unwrap();
        let mut f = LaplaceForecaster::new().auto_aid();
        f.fit(&ts).unwrap();
        let fc = f.predict(10).unwrap();
        for v in fc.primary() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn exog_preregression_removes_linear_component() {
        // y = 3.0 + 2.0 * promo + noise. Preregress on promo → residuals
        // should be near-zero-mean and small; predict_with_exog should
        // add ~2 back when future promo=1.
        let n = 200;
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let promo: Vec<f64> = (0..n).map(|i| if i % 7 == 0 { 1.0 } else { 0.0 }).collect();
        let vals: Vec<f64> = promo
            .iter()
            .enumerate()
            .map(|(i, p)| 3.0 + 2.0 * p + ((i as f64 * 0.13).sin() * 0.1))
            .collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        let cal = crate::core::time_series::CalendarAnnotations::default()
            .with_regressor("promo".into(), promo.clone());
        let mut ts = TimeSeries::univariate(stamps, vals).unwrap();
        ts.set_calendar(cal);
        let mut f = LaplaceForecaster::new().with_exog_preregression(&["promo"]);
        f.fit(&ts).unwrap();

        // Future promo=1 for 5 steps.
        let mut fut = std::collections::HashMap::new();
        fut.insert("promo".to_string(), vec![1.0; 5]);
        let fc = f.predict_with_exog(5, &fut).unwrap();
        // Level forecast should include the promo lift (~2 above baseline).
        for v in fc.primary() {
            assert!(*v > 4.0, "expected level >4 with promo lift, got {v}");
            assert!(*v < 6.5, "level should be bounded above ~5+noise, got {v}");
        }
    }

    #[cfg(feature = "postprocess")]
    #[test]
    fn trim_new_product_prefix_smoke() {
        // Series with an obvious 10-obs early-life ramp, then stable.
        let n = 150;
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut vals = vec![0.0; 10]; // NewProduct-like zeros
        vals.extend((0..(n - 10)).map(|i| 5.0 + ((i as f64 * 0.1).sin() * 0.5)));
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        let ts = TimeSeries::univariate(stamps, vals).unwrap();
        let mut f = LaplaceForecaster::new().trim_new_product_prefix();
        f.fit(&ts).unwrap();
        let fc = f.predict(5).unwrap();
        for v in fc.primary() {
            assert!(v.is_finite());
        }
    }
}
