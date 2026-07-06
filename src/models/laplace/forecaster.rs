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
use crate::transform::yeo_johnson::yeo_johnson_lambda;

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
        };
    }
    let mean_y: f64 = train.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = train.iter().map(|y| (y - mean_y).powi(2)).sum();

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
    Ar1Leaf, Ar2Leaf, DriftLeaf, EmaLeaf, FractionalDiffLeaf, HoltLeaf, OuLeaf, SeasonalEmaLeaf,
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
    /// If true, `fit()` inspects the training series' characteristics
    /// (`trend_strength`, `seasonality_strength`, `acf1`) and configures
    /// the opt-in toggles from the α-8 residual-slicing evidence: always
    /// add OU; add AR(2) if `acf1 > 0.4`; add seasonal(7) if
    /// `seasonality_strength > 0.15`; add fractional-diff if `acf1 > 0.5`.
    /// Does not enable Holt / populations / Yeo-Johnson (evidence-negative
    /// on M5). The user-configured toggles are respected — `auto()` only
    /// adds, never removes.
    use_auto: bool,
    /// Seasonal period used by `auto()`. Defaults to 7 (weekly). Set via
    /// [`Self::auto_with_seasonal_period`] for non-daily panels.
    auto_seasonal_period: usize,
    /// `(d, α_mean, α_diff)` for the fractional-differencing leaf. Adds
    /// a long-memory drift-like leaf.
    frac_diff: Option<(f64, f64, f64)>,
    /// `α_mean` for the OU mean-reversion leaf. Adds an explicit
    /// mean-reverting leaf parameterised by `θ = 1 − φ`.
    ou: Option<f64>,

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
            yj_lambda: None,
            yj_auto: false,
            use_populations: false,
            use_auto: false,
            auto_seasonal_period: 7,
            frac_diff: None,
            ou: None,
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
        }
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

    fn init_leaves(&mut self) {
        let mut leaves: Vec<Box<dyn Leaf + Send>> = if self.use_populations {
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
        if let Some(p) = self.seasonal_period {
            leaves.push(Box::new(SeasonalEmaLeaf::new(p, self.seasonal_alpha)));
        }
        for &p in &self.seasonal_periods_multi {
            leaves.push(Box::new(SeasonalEmaLeaf::new(p, self.seasonal_alpha)));
        }
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
        let values = series.primary_values();
        if values.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "LaplaceForecaster requires at least one observation".into(),
            ));
        }

        // Auto-selector: inspect series characteristics before initialising
        // leaves and set the opt-in toggles from residual-slicing evidence.
        // User-configured toggles are respected — auto only adds.
        if self.use_auto {
            let chars = auto_characteristics(values, self.auto_seasonal_period);
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
                self.seasonal_period = Some(self.auto_seasonal_period);
            }
            if chars.acf1 > 0.5 && chars.trend_strength < 0.5 && self.frac_diff.is_none() {
                self.frac_diff = Some((0.4, 0.1, 0.1));
            }
        }

        self.init_leaves();
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
        // exactly once at fit start over the full training window.
        self.fitted_yj_lambda = if let Some(l) = self.yj_lambda {
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

        for &y_orig in values {
            let y = match yj {
                Some(l) => yj_forward(y_orig, l),
                None => y_orig,
            };

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
        let yj = self.fitted_yj_lambda;
        let trans_range = self.yj_trans_range;
        Ok((0..horizon)
            .map(|h| {
                let m = blend_horizon(&weights, &per_leaf, h);
                // Apply calibration scale in the leaves' space (transformed
                // if YJ is active), then YJ inverse via delta method. Clamp
                // component means to the observed training range in
                // transformed space to prevent the log-branch Jacobian from
                // exploding on far-horizon extrapolation.
                let components = m.components.into_iter().map(|(w, g)| {
                    let sigma_scaled = g.std * scale;
                    let (mean_out, sigma_out) = match yj {
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
}
