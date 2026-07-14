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

use super::dist::{Gaussian, GaussianMixture};
use super::leaves::{TerminalCrpsMixture, TerminalScaleMixture};

/// PR #7 of #180: recency-weighted frequency table for the sticky
/// lattice projection. Ports skaters' `sticky` wrapper.
#[derive(Debug, Clone)]
struct StickyState {
    /// Recency-weighted count of each exact-value observation.
    counts: Vec<(f64, f64)>,
    /// EMA rate for the frequency table.
    propensity_alpha: f64,
    /// Spike width as fraction of predictive σ. Smaller = harder atom.
    spike_frac: f64,
    /// A value becomes an atom once `count > thresh_mult * propensity_alpha`.
    thresh_mult: f64,
    /// Max simultaneous atoms.
    max_atoms: usize,
    /// Prune entries whose recency weight drops below this.
    prune_eps: f64,
}

impl StickyState {
    fn new() -> Self {
        Self {
            counts: Vec::new(),
            propensity_alpha: 0.05,
            spike_frac: 0.005,
            thresh_mult: 1.8,
            max_atoms: 6,
            prune_eps: 1e-6,
        }
    }

    /// Skaters-style observe: decay all counts, add propensity to y.
    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        let decay = 1.0 - self.propensity_alpha;
        let mut existing = None;
        for (v, w) in self.counts.iter_mut() {
            *w *= decay;
            if (*v - y).abs() < 1e-12 {
                existing = Some(*w);
            }
        }
        self.counts.retain(|(_, w)| *w >= self.prune_eps);
        if existing.is_some() {
            for (v, w) in self.counts.iter_mut() {
                if (*v - y).abs() < 1e-12 {
                    *w += self.propensity_alpha;
                    return;
                }
            }
        }
        self.counts.push((y, self.propensity_alpha));
    }

    /// Return the current lattice atoms (revisited values above threshold),
    /// top `max_atoms` by weight.
    fn atoms(&self) -> Vec<(f64, f64)> {
        let thr = self.thresh_mult * self.propensity_alpha;
        let mut sorted: Vec<(f64, f64)> = self
            .counts
            .iter()
            .copied()
            .filter(|(_, w)| *w > thr)
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(self.max_atoms);
        sorted
    }

    /// Apply sticky-lattice projection to a Gaussian mixture. Returns
    /// a new mean-preserving mixture with atom spikes plus the
    /// original continuous mass, recentered so `E[out] == m.mean()`.
    ///
    /// `h` is the forecast horizon (1-based). Fix A of the fev-27
    /// follow-up: atom mass decays exponentially with `h`:
    /// `p_atoms(h) = p_atoms · (1 - decay_per_step)^(h-1)`
    /// with `decay_per_step = 0.05` (half-life ~14 steps). This models
    /// the fact that revisited-value evidence gets stale as the
    /// forecast moves further ahead — timeless atoms were the root of
    /// the fev-27 continuous-panel WQL blowup (up to 1800× worse than
    /// classical on `m1_yearly`).
    fn project(&self, m: &GaussianMixture, h: usize) -> GaussianMixture {
        let atoms = self.atoms();
        if atoms.is_empty() || m.is_empty() {
            return m.clone();
        }
        let sw: f64 = atoms.iter().map(|(_, w)| w).sum();
        if sw <= 0.0 {
            return m.clone();
        }
        // Fix A of fev-27 follow-up: horizon-decayed atom mass.
        const DECAY_PER_STEP: f64 = 0.05;
        let horizon_factor = (1.0 - DECAY_PER_STEP).powi(h.saturating_sub(1) as i32);
        // Cap total atom mass at 0.999 to keep some continuous coverage.
        let p_atoms = (sw * horizon_factor).min(0.999);
        let p_cont = 1.0 - p_atoms;
        let atom_mean = atoms.iter().map(|(v, w)| v * w).sum::<f64>() / sw;
        // Spike width from average predictive std.
        let avg_std: f64 = m
            .components
            .iter()
            .map(|(w, g)| w * g.std)
            .sum::<f64>()
            .max(1e-9);
        let spike_std = (self.spike_frac * avg_std).max(1e-9);
        let mu = m.mean();
        let mut comps: Vec<(f64, Gaussian)> = Vec::with_capacity(atoms.len() + m.components.len());
        if p_cont <= 1e-9 {
            for (v, w) in &atoms {
                comps.push((p_atoms * (w / sw), Gaussian::new(*v, spike_std)));
            }
            return GaussianMixture::new(comps);
        }
        // Mean-preserving recenter of the continuous component:
        //   E[out] = P_atoms · atom_mean + P_cont · (mu + δ) = mu
        //   δ = P_atoms · (mu - atom_mean) / P_cont
        let delta = p_atoms * (mu - atom_mean) / p_cont;
        for (v, w) in &atoms {
            comps.push((p_atoms * (w / sw), Gaussian::new(*v, spike_std)));
        }
        for (w, g) in &m.components {
            comps.push((p_cont * w, Gaussian::new(g.mean + delta, g.std)));
        }
        GaussianMixture::new(comps)
    }
}
use crate::transform::yeo_johnson::yeo_johnson_lambda;
use crate::utils::ols::{ols_fit, OLSResult};
use std::collections::HashMap;

use super::ensemble::{blend_horizon, softmax, softmax_into};

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

/// Fev-27 follow-up (#5): learning-rate warmup schedule.
///
/// For the first 30 observations, use `η=1.0` (fast — the softmax
/// needs to move away from uniform quickly during warmup). Then
/// linearly decay to the configured `learning_rate` over the next
/// 70 observations. Beyond n=100, hold at `learning_rate`.
///
/// Prevents the short-history yearly regression that #180's Fix B
/// introduced: with η=0.5 the whole way, on N=30 yearly panels the
/// softmax doesn't reach a peaked distribution before we need to
/// predict.
#[inline]
fn eta_schedule(base_eta: f64, n_obs: usize) -> f64 {
    const WARMUP: usize = 30;
    const DECAY_END: usize = 100;
    if n_obs < WARMUP {
        1.0
    } else if n_obs < DECAY_END {
        let t = (n_obs - WARMUP) as f64 / (DECAY_END - WARMUP) as f64;
        1.0 + t * (base_eta - 1.0)
    } else {
        base_eta
    }
}

/// Short-data softmax dampener (yearly Trick 3).
///
/// When the total training length is very short (M-competition yearly:
/// N=24-33), the fit loop's η stays at 1.0 throughout warmup. Softmax
/// accumulates full log-likelihood per step, which can lock the
/// ensemble onto whichever leaf was best in the first few rounds —
/// with only 30 obs behind the ranking of ~15 leaves, this "winner"
/// is essentially noise.
///
/// Returns a multiplier `≤ 1.0` applied to the schedule's η. For
/// `total_n ≥ 60` returns 1.0 (no dampening). Below that, scales
/// linearly down to `0.4` at N=0 — mild flattening that leaves room
/// for slower but more reliable ensemble averaging.
#[inline]
#[allow(dead_code)] // Kept for future Trick-3 iterations, see docs/ACCURACY_AUDIT.md.
fn short_data_multiplier(total_n: usize) -> f64 {
    if total_n >= 60 {
        1.0
    } else {
        (total_n as f64 / 60.0).max(0.4)
    }
}

/// Solve the ensemble stacking problem (accuracy-audit #1).
///
/// Given per-leaf 1-step prediction history and the target training
/// values, solve:
///   `min || y_train[burn..] − X · w ||² + λ · ||w||²`
///   s.t. `w >= 0, Σ w_i = 1`
///
/// Uses ridge-regularized OLS via normal equations, then projects to
/// the non-negative simplex.
///
/// The ridge term prevents blowup on collinear leaves (e.g.
/// `EMA(0.05)` ≈ `EMA(0.1)` on smooth series). Simplex projection
/// keeps the blend interpretable and non-degenerate.
fn solve_stacking(
    predictions: &[Vec<f64>], // [leaf_idx][step]
    values: &[f64],
    burn: usize,
) -> Vec<f64> {
    let n_leaves = predictions.len();
    if n_leaves == 0 {
        return Vec::new();
    }
    let n_steps = predictions[0].len().min(values.len());
    if n_steps <= burn + n_leaves {
        // Not enough data — fall back to uniform weights.
        return vec![1.0 / n_leaves as f64; n_leaves];
    }
    let effective_n = n_steps - burn;
    // Ridge parameter — small compared to typical MSE.
    let ridge_lambda = 1e-4;
    // Build X^T X (n_leaves × n_leaves) and X^T y (n_leaves).
    let mut xtx = vec![vec![0.0f64; n_leaves]; n_leaves];
    let mut xty = vec![0.0f64; n_leaves];
    for step in burn..n_steps {
        let y = values[step];
        if !y.is_finite() {
            continue;
        }
        for i in 0..n_leaves {
            let xi = predictions[i][step];
            if !xi.is_finite() {
                continue;
            }
            xty[i] += xi * y;
            for j in 0..n_leaves {
                let xj = predictions[j][step];
                if xj.is_finite() {
                    xtx[i][j] += xi * xj;
                }
            }
        }
    }
    // Add ridge to the diagonal.
    for i in 0..n_leaves {
        xtx[i][i] += ridge_lambda * effective_n as f64;
    }
    // Solve via Gaussian elimination (n_leaves is small, ~30).
    let mut aug: Vec<Vec<f64>> = xtx
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(xty[i]);
            r
        })
        .collect();
    // Forward elimination.
    for i in 0..n_leaves {
        // Partial pivoting.
        let mut max_row = i;
        for k in i + 1..n_leaves {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);
        let pivot = aug[i][i];
        if pivot.abs() < 1e-12 {
            // Singular — fall back to uniform.
            return vec![1.0 / n_leaves as f64; n_leaves];
        }
        for k in i + 1..n_leaves {
            let factor = aug[k][i] / pivot;
            for j in i..=n_leaves {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }
    // Back substitution.
    let mut w = vec![0.0f64; n_leaves];
    for i in (0..n_leaves).rev() {
        let mut sum = aug[i][n_leaves];
        for j in i + 1..n_leaves {
            sum -= aug[i][j] * w[j];
        }
        w[i] = sum / aug[i][i];
    }
    // Simplex projection (Duchi 2008).
    project_to_simplex(&mut w);
    w
}

/// Project a vector onto the probability simplex (non-negative, sum-to-one).
/// Uses the Duchi et al. (2008) algorithm — same as
/// `ensemble::model::nnls_simplex`.
fn project_to_simplex(w: &mut [f64]) {
    let n = w.len();
    if n == 0 {
        return;
    }
    let mut sorted: Vec<f64> = w.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0;
    let mut rho = 0;
    for (j, &val) in sorted.iter().enumerate() {
        cumsum += val;
        if val - (cumsum - 1.0) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }
    let theta = (sorted[..=rho].iter().sum::<f64>() - 1.0) / (rho as f64 + 1.0);
    for w_i in w.iter_mut() {
        *w_i = (*w_i - theta).max(0.0);
    }
    // Renormalize (should already sum to 1, but be safe).
    let sum: f64 = w.iter().sum();
    if sum > 0.0 {
        for w_i in w.iter_mut() {
            *w_i /= sum;
        }
    }
}

/// Heuristic: does the training window look **trending** enough that
/// batch-initializing Drift + Holt with OLS β is a net win? (Yearly Trick 1.)
///
/// Returns `true` when `|β| > 0.5 · residual_σ / N`, i.e. the OLS
/// slope is at least half the "noise slope" you'd get from N random-
/// walk steps of size σ. Trend-strength threshold empirically tuned to
/// avoid initializing zero-drift on flat-noise series (where init
/// would hurt).
#[allow(dead_code)] // Retained for future Trick-1 iterations, see docs/ACCURACY_AUDIT.md.
fn looks_trending(values: &[f64]) -> bool {
    let n = values.len();
    if n < 5 {
        return false;
    }
    let n_f = n as f64;
    let mean_t = (n_f - 1.0) / 2.0;
    let mean_y: f64 = values.iter().sum::<f64>() / n_f;
    let mut num = 0.0;
    let mut den = 0.0;
    let mut ss = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let dt = i as f64 - mean_t;
        let dy = y - mean_y;
        num += dt * dy;
        den += dt * dt;
        ss += dy * dy;
    }
    if den < 1e-12 || ss < 1e-12 {
        return false;
    }
    let beta = num / den;
    let sigma = (ss / n_f).sqrt();
    // Trend contribution over the window: β · (N-1). Compare to the
    // total "noise budget" σ · √N.
    let trend_over_window = (beta * (n_f - 1.0)).abs();
    let noise_budget = sigma * n_f.sqrt();
    trend_over_window > 0.5 * noise_budget
}

/// Compute the sample lag-`p` autocorrelation of `values`. Retained
/// for future issue #198 iterations — the naïve version of the
/// boost caused a +43 % MASE regression on fev-27 because ACF-driven
/// seasonal forcing overrode data-driven leaf ranking on datasets
/// where seasonal_ema isn't the best fit. Returns 0.0 on degenerate
/// inputs (fewer than `2*p+1` obs, zero variance).
#[allow(dead_code)]
fn lag_acf(values: &[f64], p: usize) -> f64 {
    if p == 0 || values.len() < 2 * p + 1 {
        return 0.0;
    }
    let n = values.len();
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let var: f64 = values.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-12 {
        return 0.0;
    }
    let mut cov = 0.0;
    for i in p..n {
        cov += (values[i] - mean) * (values[i - p] - mean);
    }
    let normalizer = (n - p) as f64 * var;
    (cov / normalizer).clamp(-1.0, 1.0)
}

/// Median-absolute-deviation robust σ estimator (accuracy-audit #3a).
///
/// Returns `1.4826 · median(|y_i − median(y)|)` — the MAD scaled to
/// match a Gaussian σ. Used to warm-start the terminal scale-mixture
/// so short-history panels don't spend 30 observations recalibrating.
fn compute_mad(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return 0.0;
    }
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    };
    let mut abs_dev: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if abs_dev.len() % 2 == 0 {
        0.5 * (abs_dev[mid - 1] + abs_dev[mid])
    } else {
        abs_dev[mid]
    };
    1.4826 * mad
}

/// Heuristic: does the training window look **discrete-count-like**?
/// Returns `true` if it does (few distinct near-integer values relative
/// to sample size). Used to auto-gate sticky-lattice — atoms are
/// meaningful only when the data actually revisits exact values.
///
/// Test: count distinct values in the first `N.min(1000)` observations
/// after rounding to the nearest integer. If the ratio
/// `distinct / total < 0.15` AND most values are within 0.05 of an
/// integer, the data is discrete-count-like.
fn looks_discrete_count(train: &[f64]) -> bool {
    let n = train.len().min(1000);
    if n < 20 {
        return false;
    }
    let sample = &train[..n];
    // Are most values integer-like?
    let near_int = sample
        .iter()
        .filter(|y| y.is_finite() && (y.round() - **y).abs() < 0.05)
        .count();
    if (near_int as f64 / n as f64) < 0.8 {
        return false;
    }
    // How many distinct integers?
    let mut ints: Vec<i64> = sample
        .iter()
        .filter(|y| y.is_finite())
        .map(|y| y.round() as i64)
        .collect();
    ints.sort_unstable();
    ints.dedup();
    let distinct = ints.len();
    (distinct as f64 / n as f64) < 0.15
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
    GammaLeaf, GarchWrappedLeaf, HoltLeaf, IntermittentLeaf, LogNormalLeaf,
    MultiplicativeSeasonalLeaf, NegativeBinomialLeaf, OuLeaf, PoissonLeaf, PowerTransformWrapper,
    RectifiedNormalLeaf, SeasonalDifferenceWrapper, SeasonalEmaLeaf, SeasonalIntermittentLeaf,
    SkewNormalLeaf, SlowStandardizeWrapper, StandardizeWrapper, StlDecompLeaf, StudentTLeaf,
    ThetaLeaf, TweedieLeaf, YjWrappedLeaf, ZeroInflatedNegativeBinomialLeaf,
    ZeroInflatedPoissonLeaf,
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
// Serde derives WIP — 27/33 leaves + terminals have derives, but the
// six Box<dyn Leaf + Send> wrapper leaves + LeafEnum::Wrapped variant
// need a manual impl (or refactor to Box<LeafEnum>). Removed from
// LaplaceForecaster + LeafEnum + wrappers until that refactor lands
// so cargo build --features serde succeeds. See Option A plan.
pub struct LaplaceForecaster {
    ema_alpha: f64,
    drift_alpha: f64,
    ar_alpha_mean: f64,
    holt: Option<(f64, f64, f64)>, // (alpha, beta, phi)
    ar2: Option<f64>,              // mean-EMA alpha
    seasonal_period: Option<usize>,
    /// Opt-in flag for [`Self::with_seasonal_batch_init`]: pre-fills
    /// the seasonal-EMA / multiplicative-seasonal phase levels from
    /// the last training cycle so the leaf competes fairly on the
    /// first observation. Off by default — measured regression on
    /// tourism_monthly-shape data when applied unconditionally.
    /// See `docs/ACCURACY_AUDIT.md`.
    seasonal_batch_init: bool,
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
    /// PR #3 of #180: Theta-method leaves at these α values (skaters'
    /// pool is `{0.05, 0.1, 0.3}`). Empty = no theta leaves.
    theta_alphas: Vec<f64>,
    /// PR #3 of #180: opt-in Yeo-Johnson coordinate composition —
    /// wraps every current base leaf with each λ in this list. Skaters
    /// ships `{0.0, 0.5}` composed only with `{diff, ema}` — this
    /// broader "wrap everything" version is our approximation.
    /// Distinct from [`Self::yj_grid`] (which replaces the base list).
    /// Empty = disabled.
    yj_coord_lambdas: Vec<f64>,
    /// PR #4 of #180: standardize + EMA depth-2 compositions. Each `α`
    /// in this list adds a `StandardizeWrapper(EmaLeaf(α), 0.05)`
    /// candidate. Skaters' pool has `α ∈ {0.05, 0.1}`. Empty = disabled.
    standardize_ema_alphas: Vec<f64>,
    /// v0.15.4: fast_slow pool — "thinking fast and slow" combos. Each
    /// entry is `slow_alpha` for `SlowStandardizeWrapper`. For each,
    /// wraps 5 fast trackers (EMA 0.3, EMA 0.5, Holt(0.4, 0.2, 1.0),
    /// AR1(0.1), Drift(0.05)) → 5·|slow_alphas| leaves. Ports skaters'
    /// `fast_slow` block from `api.py::_build_candidates`.
    /// Enabled by `.skaters()` with `[0.02, 0.05]` (10 leaves).
    fast_slow_slow_alphas: Vec<f64>,
    /// PR #4 of #180: seasonal-diff + EMA depth-2 compositions. Each
    /// `(period, α)` adds a `SeasonalDifferenceWrapper(EmaLeaf(α), period)`.
    /// Skaters' pool has `period ∈ {7, 12, 24}` × `α ∈ {0.05, 0.1}` = 6.
    seasonal_diff_ema: Vec<(usize, f64)>,
    /// PR #4 of #180: diff + EMA depth-2 compositions (period=1 special
    /// case). Skaters' pool has 3 candidates: `α ∈ {0.05, 0.1, 0.3}`.
    /// Each adds a `SeasonalDifferenceWrapper(EmaLeaf(α), 1)`.
    diff_ema_alphas: Vec<f64>,
    /// PR #4 of #180: multi-speed drift grid. Skaters' pool has 4
    /// speed/shrinkage combos; here we just carry `α` speeds. Each α
    /// adds a `DriftLeaf(α)` candidate.
    drift_alphas: Vec<f64>,
    /// PR #5 of #180: Bayesian-ensemble learning rate. Log-weight update
    /// per observation is `log_w[i] += η · lp` — smaller η keeps the
    /// ensemble adaptive to regime change (XGBoost-style shrinkage).
    /// Skaters ships `η = 0.5`; our historical default was `η = 1.0`
    /// (exact cumulative log-likelihood). Applied uniformly in
    /// `fit()`'s per-leaf scoring loop.
    learning_rate: f64,
    /// PR #5 of #180: floor for per-leaf log-likelihood before it hits
    /// the cumulative-weight update. Bounds catastrophic single-obs
    /// losses so a candidate can recover from one bad prediction.
    /// Skaters ships `-20.0`; `f64::NEG_INFINITY` disables (our historical
    /// default).
    log_clamp: f64,
    /// PR #6 of #180: fractional-diff variants for the fixed pool.
    /// Each `(d, α_mean, α_diff)` adds a `FractionalDiffLeaf`.
    /// Skaters' pool has `d ∈ {0.2, 0.4}` composed with EMA.
    frac_diff_variants: Vec<(f64, f64, f64)>,
    /// PR #6 of #180: GARCH + EMA composition candidates. Each entry
    /// adds `GarchWrappedLeaf(EmaLeaf(α), 0.01, 0.1, 0.85)` — skaters'
    /// default GARCH(1,1) hyperparameters composed with an inner EMA.
    garch_ema_alphas: Vec<f64>,
    /// PR #6 of #180: PowerTransform + EMA composition candidates.
    /// Each `(p, α)` adds `PowerTransformWrapper(EmaLeaf(α), p)`.
    /// Skaters ships `p = 0.5` composed with EMA α = 0.1.
    power_ema: Vec<(f64, f64)>,
    /// PR #6 of #180: Yeo-Johnson coordinate compositions with an EMA
    /// inner. Each `(λ, α)` adds `YjWrappedLeaf(EmaLeaf(α), λ)`.
    /// Skaters ships `λ ∈ {0.0, 0.5}` composed with EMA α = 0.1.
    yj_ema: Vec<(f64, f64)>,
    /// PR #6 of #180: Yeo-Johnson coordinate compositions with a
    /// differencing inner. Each `(λ, ema_α)` adds
    /// `YjWrappedLeaf(SeasonalDifferenceWrapper(EmaLeaf(ema_α), 1), λ)`.
    /// Skaters ships `λ ∈ {0.0, 0.5}` composed with diff+EMA α = 0.1.
    yj_diff_ema: Vec<(f64, f64)>,
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

    leaves: Vec<super::leaf_enum::LeafEnum>,
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
    /// PR #7 of #180: alternate terminal — CRPS-gradient variant.
    /// Ports skaters' `crps_leaf`. When set, takes precedence over
    /// [`Self::terminal`]. Enabled automatically by `.skaters()`.
    terminal_crps: Option<TerminalCrpsMixture>,
    /// PR #7 of #180: sticky lattice projection. Ports skaters'
    /// `sticky` wrapper — near-Dirac atoms at revisited values so a
    /// continuous mixture doesn't pay density mass on exact-integer
    /// counts (the modal outcome on M5).
    sticky: Option<StickyState>,
    /// Fev-27 follow-up: auto-gate sticky based on data characteristics
    /// at fit time. When true, sticky stays on only if the training
    /// values look discrete-count-like (few distinct values → atoms
    /// are meaningful). On continuous data, sticky is disabled at fit
    /// time regardless of the initial `sticky` setting. `.skaters()`
    /// sets this true; explicit `.with_sticky()` or `.no_sticky()`
    /// leaves it false so the caller's choice is honored.
    sticky_auto_gate: bool,
    /// Fev-27 follow-up (#9): STL-decomposition leaf period.
    /// When `Some(p)` with `p >= 2`, adds an `StlDecompLeaf(p)` to the
    /// pool. Auto-enabled by `.auto()` / `.skaters()` when seasonality
    /// detection returns a period.
    stl_period: Option<usize>,
    /// Perf: reusable scratch buffer for per-leaf `Gaussian` predictions
    /// so the fit-loop's `predict_one` results aren't heap-allocated per
    /// observation. Sized once to `self.leaves.len()` at fit start.
    scratch_per_leaf: Vec<Gaussian>,
    /// Perf: parallel scratch buffer of `ln(std)` for each entry of
    /// [`Self::scratch_per_leaf`]. Precomputed once per obs so the
    /// scoring loop's inlined `logpdf` has zero transcendentals.
    #[allow(dead_code)] // Retained field; consumed by an inlined logpdf path.
    scratch_ln_std: Vec<f64>,
    /// Perf: reusable scratch buffer for softmax weights so
    /// `self.weights()` doesn't allocate on the fit hot path.
    scratch_weights: Vec<f64>,
    /// Accuracy-audit #2 (multi-horizon scoring): when enabled, during
    /// fit, periodically snapshot each leaf's h-step predictions and
    /// retrospectively score them against the future y[t+h]. Adds the
    /// h>1 log-likelihood contributions (weighted) to cum_log_liks so
    /// the softmax reflects long-horizon accuracy, not just 1-step.
    multi_h_scoring: bool,
    /// Explicit scoring horizon for multi-h scoring; overrides the
    /// default `per_h_horizon.clamp(4, 24)`. Set by `with_scoring_horizon`.
    scoring_horizon: Option<usize>,
    /// Sliding-window LL: when `Some(w)`, `cum_log_liks[i]` is a moving
    /// sum of the last `w` `logpdf` values instead of a cumulative sum.
    /// Set by `with_scoring_window`.
    scoring_window: Option<usize>,
    /// Per-leaf ring buffer of the last `w` `logpdf` values, used to
    /// implement `scoring_window`.
    scoring_window_hist: Vec<std::collections::VecDeque<f64>>,
    /// Accuracy-audit #1 (ensemble stacking): enable OLS-based blend
    /// weight learning at end of fit. When true and enough training data
    /// is available, `stacking_weights` is populated with per-leaf
    /// linear-combination weights minimising `||y_train - X · w||²`
    /// (with non-negativity + simplex projection). Used at
    /// `forecast_dist` time in place of softmax weights for the mean
    /// blend.
    stacking_enabled: bool,
    /// Filled at end of fit when stacking is enabled and the training
    /// window has enough observations. Length = `leaves.len()`.
    stacking_weights: Option<Vec<f64>>,
    /// Per-leaf, per-step 1-step-ahead prediction history collected
    /// during fit — the design matrix for the stacking OLS. Only
    /// populated when `stacking_enabled` is true.
    predictions_history: Vec<Vec<f64>>,
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
            seasonal_batch_init: false,
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
            theta_alphas: Vec::new(),
            yj_coord_lambdas: Vec::new(),
            standardize_ema_alphas: Vec::new(),
            fast_slow_slow_alphas: Vec::new(),
            seasonal_diff_ema: Vec::new(),
            diff_ema_alphas: Vec::new(),
            drift_alphas: Vec::new(),
            // PR #5 of #180 defaults kept at the historical values so
            // existing callers see the same behavior. `.skaters()`,
            // `.learning_rate(η)`, `.log_clamp(b)` opt into the new
            // mechanism.
            learning_rate: 1.0,
            log_clamp: f64::NEG_INFINITY,
            frac_diff_variants: Vec::new(),
            garch_ema_alphas: Vec::new(),
            power_ema: Vec::new(),
            yj_ema: Vec::new(),
            yj_diff_ema: Vec::new(),
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
            terminal_crps: None,
            sticky: None,
            sticky_auto_gate: false,
            stl_period: None,
            scratch_per_leaf: Vec::new(),
            scratch_ln_std: Vec::new(),
            scratch_weights: Vec::new(),
            stacking_enabled: false,
            stacking_weights: None,
            predictions_history: Vec::new(),
            multi_h_scoring: false,
            scoring_horizon: None,
            scoring_window: None,
            scoring_window_hist: Vec::new(),
        }
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

    /// Enable the CRPS-gradient terminal leaf (PR #7 of #180).
    ///
    /// Same fixed-scale mixture shape as [`Self::with_terminal_scale_mixture`],
    /// but the component weights are updated by **exponentiated-gradient
    /// descent on the closed-form mixture CRPS** rather than
    /// likelihood-EM. Uses 15 log-spaced scale components
    /// (`c = 0.4 · 1.28^i` for `i ∈ 0..15`) vs. 5 in the likelihood
    /// variant — more granular tail coverage.
    ///
    /// Ports skaters' `crps_leaf`. Takes precedence over the
    /// likelihood-EM terminal when both are configured. Enabled
    /// automatically by `.skaters()`.
    pub fn with_terminal_crps(mut self) -> Self {
        self.terminal_crps = Some(TerminalCrpsMixture::new());
        self
    }

    /// Same as [`Self::with_terminal_crps`] but exposes the two
    /// rate parameters. Defaults `(scale_alpha=0.01, eta=1.0)` match
    /// skaters' `crps_leaf`.
    pub fn with_terminal_crps_params(mut self, scale_alpha: f64, eta: f64) -> Self {
        self.terminal_crps = Some(TerminalCrpsMixture::with_params(scale_alpha, eta));
        self
    }

    /// Enable the sticky lattice projection (PR #7 of #180).
    ///
    /// Adds near-Dirac atoms at revisited exact-value observations so
    /// a continuous mixture doesn't pay density mass on discrete
    /// values the series keeps returning to (0 on M5 first-differenced
    /// counts, integer prices, etc.). Mean-preserving — the atoms plus
    /// the recentered continuous part have the same expected value as
    /// the original mixture. Ports skaters' `sticky` wrapper with its
    /// defaults `(propensity_alpha=0.05, spike_frac=0.005,
    /// thresh_mult=1.8, max_atoms=6)`.
    ///
    /// On continuous series no value gets revisited, no atom fires,
    /// and the wrapper vanishes.
    ///
    /// Enabled automatically by `.skaters()`.
    pub fn with_sticky(mut self) -> Self {
        self.sticky = Some(StickyState::new());
        // Explicit user choice — honor it, don't auto-gate.
        self.sticky_auto_gate = false;
        self
    }

    /// Disable the sticky-lattice projection (PR #7 follow-up).
    ///
    /// Sticky is enabled by default in [`Self::skaters`] because it
    /// dramatically improves LL / MASE on discrete-count panels (M5,
    /// exchange_rate, dominick). On **continuous smooth panels** the
    /// atoms are placed on spurious repeated values and quantile mass
    /// concentrates in the wrong places — WQL blows up 100-1800× on
    /// short-history yearly panels like `m1_yearly`, `tourism_yearly`,
    /// `cif_2016` (fev-27 benchmark).
    ///
    /// Use `.skaters().no_sticky()` on continuous data — keeps the
    /// fixed pool, terminal scale-mixture, and shrunk-softmax
    /// mechanism, drops the atom projection. Call **after** `.skaters()`
    /// since that builder turns sticky back on.
    pub fn no_sticky(mut self) -> Self {
        self.sticky = None;
        // Explicit user choice — honor it, don't auto-gate.
        self.sticky_auto_gate = false;
        self
    }

    /// Enable multi-horizon retrospective scoring (accuracy-audit #2).
    ///
    /// During fit, periodically snapshots each leaf's h-step predictions.
    /// After the observation loop, iterates the snapshots and adds a
    /// weighted h-step log-likelihood contribution to `cum_log_liks`
    /// for each leaf. Score at horizon h uses `y[snapshot_step + h - 1]`
    /// against the leaf's h-step prediction from `snapshot_step`.
    ///
    /// The h-step contribution is weighted by `η · 1/h` so far horizons
    /// don't dominate. Total effect: leaves that flat-line at long h
    /// (e.g. slow EMAs) get down-weighted in the final softmax.
    pub fn with_multi_h_scoring(mut self) -> Self {
        self.multi_h_scoring = true;
        self
    }

    /// Score the softmax on h-step LL where `h` equals the caller's
    /// forecast horizon. Enables multi-h scoring and overrides the
    /// default `per_h_horizon.clamp(4, 24)` depth. Useful when
    /// `forecast_dist(H)` is called at a specific known H and you want
    /// the leaf weights driven by H-step accuracy, not 1-step LL.
    pub fn with_scoring_horizon(mut self, horizon: usize) -> Self {
        self.scoring_horizon = Some(horizon.max(1));
        self.multi_h_scoring = true;
        self
    }

    /// Replace the cumulative softmax score with a **sliding-window
    /// sum** of the last `w` 1-step log-likelihoods. Each new
    /// observation adds its `logpdf`, and once the window fills, the
    /// oldest `logpdf` is subtracted off. Softmax weights then reflect
    /// only recent leaf performance instead of the entire training
    /// history. Zero `learning_rate` decay is applied in this mode.
    pub fn with_scoring_window(mut self, window: usize) -> Self {
        self.scoring_window = Some(window.max(1));
        self
    }

    /// Enable ensemble stacking on top of the softmax blend
    /// (accuracy-audit #1).
    ///
    /// After the streaming fit completes, solves an OLS problem for
    /// per-leaf blend weights `w` minimising `||y_train - X · w||²`
    /// where `X[t][i] = leaf_i.predict_one_at_step(t)`. Weights are
    /// projected onto the non-negative simplex via `nnls_simplex`.
    /// At `forecast_dist` time these weights replace the softmax
    /// weights for the mean blend (σ mixture still uses softmax).
    ///
    /// **Requires N ≥ 60 training obs**. Below that, softmax is used
    /// throughout — the ridge would overfit on short series.
    ///
    /// **Cost**: N × N_leaves × 8 bytes storage during fit
    /// (~300 KB on M5). O(N × N_leaves²) solve at end of fit.
    pub fn with_stacking(mut self) -> Self {
        self.stacking_enabled = true;
        self
    }

    /// Add an STL-decomposition leaf at the given period (opt-in).
    ///
    /// `StlDecompLeaf(period)` runs STL on a rolling buffer of the last
    /// `10 * period` observations and extrapolates a linear trend plus
    /// cyclic seasonal pattern. Useful when the series has strong
    /// deterministic seasonality that our streaming leaves can't
    /// capture cleanly (M-competition monthly / quarterly panels are
    /// candidates).
    ///
    /// **NOT auto-enabled by `.auto()` or `.skaters()`.** An earlier
    /// attempt to auto-enable it based on `seasonality_strength > 0.30`
    /// caused a 31 % MASE regression on `tourism_monthly` in the fev-27
    /// bakeoff: the STL leaf's linear-trend extrapolation is aggressive
    /// at long horizons and compounds on short-history seasonal panels
    /// (150-300 obs, H=24). Aggregate fev-27 also regressed 1.7 %
    /// (geomean MASE 6.085 → 6.186). Documented in
    /// `docs/SOTA_POSITIONING.md`.
    ///
    /// Use this only when you've verified STL helps on your specific
    /// data.
    pub fn with_stl(mut self, period: usize) -> Self {
        if period >= 2 {
            self.stl_period = Some(period);
        }
        self
    }

    /// Enable Theta-method leaves at the given α values (PR #3 of #180).
    ///
    /// Ports skaters' `theta(α)` transform. Each variant is a SES level
    /// plus a running-OLS half-slope drift extrapolation — the best
    /// simple univariate method in M3, near-best in M4. Skaters' pool
    /// ships `α ∈ {0.05, 0.1, 0.3}`.
    ///
    /// Enabled automatically by `.auto()` at that same 3-α pool.
    pub fn with_theta(mut self, alphas: &[f64]) -> Self {
        self.theta_alphas = alphas.iter().copied().filter(|a| a.is_finite()).collect();
        self
    }

    /// Enable standardize + EMA depth-2 compositions (PR #4 of #180).
    ///
    /// For each α in `ema_alphas`, adds a `StandardizeWrapper(EmaLeaf(α), 0.05)`
    /// candidate. Ports skaters' `α ∈ {0.05, 0.1}` pool. The standardize
    /// transform tracks the running mean+variance so the inner EMA sees
    /// a stationary, unit-variance stream.
    ///
    /// Enabled automatically by `.auto()` at the standard 2-α pool.
    pub fn with_standardize_ema(mut self, ema_alphas: &[f64]) -> Self {
        self.standardize_ema_alphas = ema_alphas
            .iter()
            .copied()
            .filter(|a| a.is_finite() && *a > 0.0)
            .collect();
        self
    }

    /// Enable seasonal-diff + EMA depth-2 compositions (PR #4 of #180).
    ///
    /// For each `(period, α)`, adds a
    /// `SeasonalDifferenceWrapper(EmaLeaf(α), period)` candidate. Ports
    /// skaters' `{7, 12, 24} × {0.05, 0.1}` = 6 candidates. Removes an
    /// s-lag seasonal from the series so the inner EMA models the
    /// deseasonalised residual.
    ///
    /// Enabled automatically by `.auto()` at the standard 6-candidate pool.
    pub fn with_seasonal_diff_ema(mut self, pairs: &[(usize, f64)]) -> Self {
        self.seasonal_diff_ema = pairs
            .iter()
            .filter(|(p, a)| *p >= 1 && a.is_finite() && *a > 0.0)
            .copied()
            .collect();
        self
    }

    /// Override the Bayesian-ensemble learning rate (PR #5 of #180).
    ///
    /// The per-observation log-weight update is
    ///
    /// ```text
    ///   log_w[i] += η · logpdf_i(y)
    /// ```
    ///
    /// At `η = 1.0` (our historical default) this is exact cumulative
    /// log-likelihood updating — a single peaked candidate can pull all
    /// weight quickly. At `η = 0.5` (skaters' default) the update is
    /// XGBoost-shrunk: the ensemble stays adaptive to regime change at
    /// the cost of slower convergence to the best single candidate.
    ///
    /// Clamped to `(0, 1]`.
    pub fn learning_rate(mut self, eta: f64) -> Self {
        self.learning_rate = eta.clamp(1e-4, 1.0);
        self
    }

    /// Set a lower bound on per-observation log-likelihood contributions
    /// (PR #5 of #180).
    ///
    /// Each candidate's `lp = logpdf(y)` is clamped to
    /// `max(lp, log_clamp)` before its cumulative-weight update. Bounds
    /// catastrophic single-observation losses so a candidate can recover
    /// from one bad prediction. Skaters ships `-20.0` (about 5σ into the
    /// tail of `N(0, 1)`); `f64::NEG_INFINITY` disables the clamp (our
    /// historical default).
    pub fn log_clamp(mut self, bound: f64) -> Self {
        self.log_clamp = bound;
        self
    }

    /// Skaters-style ensemble configuration (PR #5 of #180).
    ///
    /// Runs the **full fixed candidate pool** with skaters' softmax
    /// mechanism:
    ///
    /// - **All candidates on, always** (no data-heuristic gating) —
    ///   ~30 leaves matching the depth-1 and depth-2 slices we've
    ///   ported: EMA (3 speeds), Drift (3 speeds), AR(1), Theta (3 α),
    ///   Standardize+EMA (2), Seasonal-diff+EMA (6 at {7, 12, 24} × {0.05, 0.1}),
    ///   Diff+EMA (3), Multi-speed drift (3).
    /// - **Terminal scale-mixture** on top (matches skaters).
    /// - **Learning rate `η = 0.5`** (XGBoost-shrunk log-weight updates).
    /// - **Log-clamp `-20.0`** (bounded single-observation losses).
    ///
    /// Contrast with [`Self::auto`] which uses data-heuristic inclusion.
    /// Skaters' philosophy: trust the softmax; our `.auto()`'s
    /// philosophy: filter first. Both are legitimate. See #180 for
    /// bakeoff comparisons.
    pub fn skaters(mut self) -> Self {
        self.learning_rate = 0.5;
        self.log_clamp = -20.0;
        // PR #7 of #180: skaters' default terminal is `crps_leaf`
        // (CRPS-gradient) but empirically on M5 first-differenced counts
        // the likelihood-EM `scale_mixture_leaf` is better. `.skaters()`
        // uses the likelihood variant by default; opt in to CRPS via
        // `.with_terminal_crps()` for continuous / heavy-tailed data.
        if self.terminal.is_none() && self.terminal_crps.is_none() {
            self.terminal = Some(TerminalScaleMixture::new());
        }
        // PR #7 of #180: sticky lattice on by default in .skaters().
        // Fev-27 follow-up: auto-gate at fit time — sticky stays on
        // only if data looks discrete-count-like. On continuous
        // panels (m1_yearly, cif_2016, tourism_yearly) sticky would
        // otherwise blow up WQL. Callers can override with
        // `.with_sticky()` (force on) or `.no_sticky()` (force off).
        if self.sticky.is_none() {
            self.sticky = Some(StickyState::new());
        }
        self.sticky_auto_gate = true;
        // Populate the full fixed pool, matching skaters' candidate
        // types (excluding items that don't shift M5 auto-enable per
        // PR #4 empirical decisions — but still on here because
        // skaters' style is "everything always on").
        if self.theta_alphas.is_empty() {
            self.theta_alphas = vec![0.05, 0.1, 0.3];
        }
        // Accuracy-audit #4b: REVERTED. Added damped Holt(0.3, 0.1, 0.9)
        // unconditionally in `.skaters()` — caused a +1.5 % geomean MASE
        // regression on fev-27 (see docs/ACCURACY_AUDIT.md). Callers who
        // want damped Holt in the skaters pool can add it explicitly via
        // `.with_holt(0.3, 0.1, 0.9).skaters()`.
        if self.standardize_ema_alphas.is_empty() {
            self.standardize_ema_alphas = vec![0.05, 0.1];
        }
        if self.seasonal_diff_ema.is_empty() {
            self.seasonal_diff_ema = vec![
                (7, 0.05),
                (7, 0.1),
                (12, 0.05),
                (12, 0.1),
                (24, 0.05),
                (24, 0.1),
            ];
        }
        if self.diff_ema_alphas.is_empty() {
            self.diff_ema_alphas = vec![0.05, 0.1, 0.3];
        }
        if self.drift_alphas.is_empty() {
            self.drift_alphas = vec![0.01, 0.002, 0.0005];
        }
        // PR #6 of #180: fractional-diff variants at skaters' 2 d values.
        // Composed with EMA at α = 0.1 internally (FractionalDiffLeaf
        // takes (d, α_mean, α_diff)).
        if self.frac_diff_variants.is_empty() {
            self.frac_diff_variants = vec![(0.2, 0.1, 0.1), (0.4, 0.1, 0.1)];
        }
        // PR #6 of #180: GARCH + EMA (1 candidate at skaters' default).
        if self.garch_ema_alphas.is_empty() {
            self.garch_ema_alphas = vec![0.1];
        }
        // PR #6 of #180: PowerTransform(0.5) + EMA (1 candidate).
        if self.power_ema.is_empty() {
            self.power_ema = vec![(0.5, 0.1)];
        }
        // PR #6 of #180: YJ coordinate compositions (4 candidates —
        // skaters' `{0.0, 0.5} × {diff, EMA}`).
        if self.yj_ema.is_empty() {
            self.yj_ema = vec![(0.0, 0.1), (0.5, 0.1)];
        }
        if self.yj_diff_ema.is_empty() {
            self.yj_diff_ema = vec![(0.0, 0.1), (0.5, 0.1)];
        }
        // Fast-slow family (10 candidates — skaters' 12 minus the two
        // `difference()` trackers we can't compose 3-deep yet). Ports
        // `_build_candidates`'s `fast_slow` group from skaters' api.py.
        if self.fast_slow_slow_alphas.is_empty() {
            self.fast_slow_slow_alphas = vec![0.02, 0.05];
        }
        // Do NOT set self.use_auto — the heuristic path is orthogonal
        // and the caller may pipe `.skaters().auto()` if they want both.
        self
    }

    /// Enable diff + EMA depth-2 compositions (PR #4 of #180, opt-in).
    ///
    /// Adds a `SeasonalDifferenceWrapper(EmaLeaf(α), 1)` for each α.
    /// **Not auto-enabled** — on M5's zero-heavy first-differenced
    /// counts these diluted the softmax without adding LL signal.
    /// Available for callers on continuous / trending data.
    pub fn with_diff_ema(mut self, alphas: &[f64]) -> Self {
        self.diff_ema_alphas = alphas
            .iter()
            .copied()
            .filter(|a| a.is_finite() && *a > 0.0)
            .collect();
        self
    }

    /// Enable a multi-speed drift grid (PR #4 of #180, opt-in).
    ///
    /// Adds one `DriftLeaf(α)` per entry. **Not auto-enabled** — same
    /// M5 bakeoff finding as [`Self::with_diff_ema`]. Available for
    /// callers on data where drift matters.
    pub fn with_drift_alphas(mut self, alphas: &[f64]) -> Self {
        self.drift_alphas = alphas
            .iter()
            .copied()
            .filter(|a| a.is_finite() && *a > 0.0)
            .collect();
        self
    }

    /// Enable a Yeo-Johnson coordinate composition (PR #3 of #180).
    ///
    /// Wraps every base leaf with each λ in `lambdas`, adding them as
    /// *additional* softmax candidates (existing base leaves stay).
    /// Skaters ships `λ ∈ {0.0, 0.5}` composed with `{diff, ema}` in
    /// its depth-2 pool. This is our looser "wrap all base leaves"
    /// approximation.
    ///
    /// Different from [`Self::with_yeo_johnson_grid`] (which *replaces*
    /// the base list and can dilute the pool 2×+). This one *adds*.
    pub fn with_yj_coord(mut self, lambdas: &[f64]) -> Self {
        self.yj_coord_lambdas = lambdas.iter().copied().filter(|l| l.is_finite()).collect();
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
        // Accuracy-audit #7: import the XGBoost-shrunk log-weight update
        // from `.skaters()` — smaller η + log-clamp cap prevents the
        // softmax from over-concentrating on a single winner when the
        // pool has many correlated candidates. Historical defaults were
        // `η=1.0, clamp=-∞` (exact cumulative log-likelihood).
        self.learning_rate = 0.5;
        self.log_clamp = -20.0;
        // PR #1 of #180: terminal scale-mixture leaf — reshape the
        // predictive density once at the top. Cheap in fit time
        // (5-component EWMA + weight vector), meaningful LL win.
        if self.terminal.is_none() {
            self.terminal = Some(TerminalScaleMixture::new());
        }
        // PR #3 of #180: Theta-method leaves at skaters' 3 α values.
        // Cheap (level + running-OLS accumulators), covers the SES +
        // half-slope forecaster that Theta is best-known for.
        if self.theta_alphas.is_empty() {
            self.theta_alphas = vec![0.05, 0.1, 0.3];
        }
        // PR #4 of #180: standardize + EMA depth-2 compositions.
        if self.standardize_ema_alphas.is_empty() {
            self.standardize_ema_alphas = vec![0.05, 0.1];
        }
        // PR #4 of #180: seasonal-diff + EMA depth-2 compositions at
        // skaters' 3 periods × 2 α values = 6 candidates. Actual seasonal
        // period is auto-detected below; this is the coarse fallback grid.
        if self.seasonal_diff_ema.is_empty() {
            self.seasonal_diff_ema = vec![
                (7, 0.05),
                (7, 0.1),
                (12, 0.05),
                (12, 0.1),
                (24, 0.05),
                (24, 0.1),
            ];
        }
        // Note: diff + EMA + multi-speed drift stayed opt-in. M5 bakeoff
        // showed both dilute the softmax without adding signal on
        // first-differenced counts (LL regressed 0.003 nats). Available
        // via `.with_diff_ema(&[...])` / `.with_drift_alphas(&[...])`
        // for callers on data types where they should help.
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
        if self.theta_alphas.is_empty() {
            self.theta_alphas = vec![0.05, 0.1, 0.3];
        }
        if self.standardize_ema_alphas.is_empty() {
            self.standardize_ema_alphas = vec![0.05, 0.1];
        }
        if self.seasonal_diff_ema.is_empty() {
            self.seasonal_diff_ema = vec![
                (7, 0.05),
                (7, 0.1),
                (12, 0.05),
                (12, 0.1),
                (24, 0.05),
                (24, 0.1),
            ];
        }
        // Note: diff_ema / multi-speed drift stay opt-in in auto_aid
        // too — same bakeoff finding as .auto() (softmax dilution).
        self
    }

    /// Commit to a seasonal period used by BOTH [`Self::auto`] and
    /// [`Self::skaters`]. Sets both:
    ///
    /// - `auto_seasonal_period` — the fallback period `.auto()`'s ACF
    ///   scan uses when it can't detect one from data.
    /// - `seasonal_period` — the period consumed by the leaf-pool
    ///   builder to add a [`super::leaves::SeasonalEmaLeaf`].
    ///
    /// Set to 12 for monthly, 24 for hourly-with-daily, 4 for
    /// quarterly, etc. Values `< 2` set only the auto fallback (yearly
    /// data has no per-cycle seasonality to model).
    ///
    /// Fix for issue #195: previously this only set the auto-selector
    /// fallback, so `.skaters().auto_with_seasonal_period(12)` didn't
    /// actually add a seasonal-EMA leaf to the pool. Level-tracker
    /// leaves then dominated and the forecast collapsed to a flat line
    /// on amplitude-declining series (bug reproducer:
    /// `examples/issue_195_amplitude_decline.rs`).
    pub fn auto_with_seasonal_period(mut self, period: usize) -> Self {
        self.auto_seasonal_period = period.max(2);
        if period >= 2 && self.seasonal_period.is_none() {
            self.seasonal_period = Some(period);
        }
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
    ///
    /// **Since v0.15.2**, also defaults [`Self::with_seasonal_batch_init`]
    /// on. Reason: on committed-period series, batch init from the last
    /// training cycle closes the cold-start handicap that was causing
    /// near-flat forecasts on N=48-60 monthly series (issue #198). Opt
    /// out with [`Self::no_seasonal_batch_init`] if your amplitude is
    /// growing or the seasonal phase is shifting.
    ///
    /// **Contrast with [`Self::auto_with_seasonal_period`]**: that
    /// builder does NOT enable batch init, because measurement showed it
    /// regressed fev-27 (m4_quarterly WQL +214 %, `.skaters()` aggregate
    /// WQL +14.6 %). The `auto_with_seasonal_period` path is used by
    /// benchmarks and heterogeneous panels where the last-cycle prior
    /// can be actively misleading; only the user's explicit
    /// `.with_seasonal(p)` commitment is a strong-enough signal.
    pub fn with_seasonal(mut self, period: usize) -> Self {
        if period >= 2 {
            self.seasonal_period = Some(period);
            self.seasonal_batch_init = true;
        }
        self
    }

    /// Batch-initialize the seasonal-EMA / multiplicative-seasonal
    /// leaves' phase levels from the last training cycle. Closes the
    /// softmax cold-start handicap where the seasonal leaf spends
    /// its first cycle producing fallback predictions and
    /// permanently lags plain EMA/Drift in `cum_log_liks` — the
    /// mechanism behind reports of near-flat forecasts on N=48
    /// monthly (period=12) data despite obvious seasonal structure.
    ///
    /// **Opt-in.** Not enabled by default. Trade-offs, measured on
    /// `examples/monthly_48_seasonal_diagnostic.rs` and
    /// `examples/issue_195_amplitude_decline.rs`:
    ///
    /// - Constant / declining amplitude: large MAE win (2.18 → 0.07 on
    ///   strong-seasonal N=48; 5.49× → 1.10× peak-ratio on regime-change).
    /// - **Growing amplitude** (retail expanding): batch init makes the
    ///   softmax switch from `seasonal_ema` to a differenced-EMA
    ///   leaf, collapsing the forecast to flat. Do NOT enable on
    ///   growing-amplitude series.
    /// - **Phase-shifted seasonality**: same failure mode as growing —
    ///   softmax abandons `seasonal_ema` for a level tracker.
    /// - Trending panels (M-competition tourism-shape) unconditionally:
    ///   the batch-initialised additive seasonal-EMA leaf tends to
    ///   displace the multiplicative-seasonal leaf, which fits
    ///   trending × seasonal data better.
    ///
    /// Rule of thumb: safe on stationary or declining-amplitude
    /// seasonal series. Risky on growing / phase-shifting series.
    ///
    /// Requires a period to be set via [`Self::with_seasonal`],
    /// [`Self::auto_with_seasonal_period`],
    /// [`Self::with_seasonal_multi`], or `.auto()`'s auto-detection.
    /// Without a period the flag is a no-op.
    pub fn with_seasonal_batch_init(mut self) -> Self {
        self.seasonal_batch_init = true;
        self
    }

    /// Opt out of seasonal batch initialisation.
    ///
    /// As of v0.15.2, [`Self::with_seasonal`] and
    /// [`Self::auto_with_seasonal_period`] enable seasonal batch init
    /// by default (closes issues #195/#198). Call this after those
    /// builders to revert to cold-start behaviour — useful when the
    /// series has **growing amplitude** or **shifting seasonal phase**,
    /// where a last-cycle prior misleads the softmax into abandoning
    /// the seasonal leaf.
    pub fn no_seasonal_batch_init(mut self) -> Self {
        self.seasonal_batch_init = false;
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
    #[allow(dead_code)] // Retained: callers should use build_base_leaves_with_batch(None).
    fn build_base_leaves(&self) -> Vec<super::leaf_enum::LeafEnum> {
        self.build_base_leaves_with_batch(None)
    }

    /// Same as [`Self::build_base_leaves`] but optionally batch-inits
    /// Drift + Holt candidates from the given training values (yearly
    /// Trick 1). When `batch` is `Some`, Drift/Holt use `from_batch`
    /// instead of `new`.
    fn build_base_leaves_with_batch(
        &self,
        batch: Option<&[f64]>,
    ) -> Vec<super::leaf_enum::LeafEnum> {
        use super::leaf_enum::LeafEnum;
        // Local shims: `mk_drift(α)` and `mk_holt(α, β, φ)` fall back to
        // `::new` when there's no batch. This keeps the pool-construction
        // code below unchanged (which is heavily tuned per `.auto()` /
        // `.skaters()` variant).
        // NOTE on `batch`: this parameter used to drive DriftLeaf and
        // HoltLeaf `from_batch` too (yearly Trick 1). That was reverted
        // after it regressed cif_2016 by 280% — the `looks_trending`
        // gate was too permissive. `mk_drift` / `mk_holt` therefore
        // ALWAYS use `::new` and ignore `batch`. The `batch` slice is
        // used only by the seasonal closures below, where the
        // batch-mean-per-phase computation has no gate (once you know
        // the period, the per-phase mean is unambiguously a better
        // starting point than the cold zero).
        let mk_drift = |a: f64| -> DriftLeaf { DriftLeaf::new(a) };
        let mk_holt = |a: f64, b: f64, p: f64| -> HoltLeaf { HoltLeaf::new(a, b, p) };
        // Seasonal batch init — closes the softmax cold-start handicap
        // where an un-warmed SeasonalEmaLeaf / MultiplicativeSeasonalLeaf
        // spends one full cycle producing fallback predictions and
        // permanently lags plain Drift/EMA in cum_log_liks. On N=48
        // monthly (period=12) this was catastrophic — the seasonal leaf
        // never won the softmax and the forecast collapsed to a
        // near-straight line.
        let mk_seasonal_ema = |period: usize, a: f64| -> SeasonalEmaLeaf {
            match batch {
                Some(v) => SeasonalEmaLeaf::from_batch(period, a, v),
                None => SeasonalEmaLeaf::new(period, a),
            }
        };
        let mk_seasonal_mult = |period: usize, a: f64| -> MultiplicativeSeasonalLeaf {
            match batch {
                Some(v) => MultiplicativeSeasonalLeaf::from_batch(period, a, v),
                None => MultiplicativeSeasonalLeaf::new(period, a),
            }
        };
        let _ = &mk_drift; // silence unused-if-no-hits warnings
        let _ = &mk_holt;
        let _ = &mk_seasonal_ema;
        let _ = &mk_seasonal_mult;
        let mut leaves: Vec<LeafEnum> = if self.use_populations_wide {
            vec![
                LeafEnum::Ema(EmaLeaf::new(0.02)),
                LeafEnum::Ema(EmaLeaf::new(0.10)),
                LeafEnum::Ema(EmaLeaf::new(0.25)),
                LeafEnum::Ema(EmaLeaf::new(0.45)),
                LeafEnum::Ema(EmaLeaf::new(0.60)),
                LeafEnum::Drift(mk_drift(0.03)),
                LeafEnum::Drift(mk_drift(0.10)),
                LeafEnum::Drift(mk_drift(0.25)),
                LeafEnum::Ar1(Ar1Leaf::new(0.03)),
                LeafEnum::Ar1(Ar1Leaf::new(0.10)),
                LeafEnum::Ar1(Ar1Leaf::new(0.25)),
            ]
        } else if self.use_populations {
            vec![
                LeafEnum::Ema(EmaLeaf::new(0.05)),
                LeafEnum::Ema(EmaLeaf::new(0.20)),
                LeafEnum::Ema(EmaLeaf::new(0.50)),
                LeafEnum::Drift(mk_drift(0.05)),
                LeafEnum::Drift(mk_drift(0.15)),
                LeafEnum::Ar1(Ar1Leaf::new(0.05)),
                LeafEnum::Ar1(Ar1Leaf::new(0.15)),
            ]
        } else {
            vec![
                LeafEnum::Ema(EmaLeaf::new(self.ema_alpha)),
                LeafEnum::Drift(mk_drift(self.drift_alpha)),
                LeafEnum::Ar1(Ar1Leaf::new(self.ar_alpha_mean)),
            ]
        };
        if let Some((a, b, phi)) = self.holt {
            leaves.push(LeafEnum::Holt(mk_holt(a, b, phi)));
        }
        if let Some(a) = self.ar2 {
            leaves.push(LeafEnum::Ar2(Ar2Leaf::new(a)));
        }
        if let Some((d, am, ad)) = self.frac_diff {
            leaves.push(LeafEnum::FracDiff(FractionalDiffLeaf::new(d, am, ad)));
        }
        if let Some(a) = self.ou {
            leaves.push(LeafEnum::Ou(OuLeaf::new(a)));
        }
        // PR #3 of #180: Theta-method leaves (SES + half OLS slope).
        for &a in &self.theta_alphas {
            leaves.push(LeafEnum::Theta(ThetaLeaf::new(a)));
        }
        // Fev-27 follow-up (#9): STL-decomposition leaf. Batch fitter
        // dressed as a streaming leaf; runs STL on the rolling buffer
        // at predict time. Closes the M-competition monthly/quarterly
        // gap where our streaming leaves lose 30-50 % MASE to
        // AutoTheta's proper seasonal decomposition.
        if let Some(p) = self.stl_period {
            if p >= 2 {
                leaves.push(LeafEnum::Stl(StlDecompLeaf::new(p)));
            }
        }
        // PR #4 of #180: standardize + EMA depth-2 compositions.
        for &alpha in &self.standardize_ema_alphas {
            leaves.push(LeafEnum::Wrapped(Box::new(StandardizeWrapper::new(
                Box::new(EmaLeaf::new(alpha)),
                0.05,
            ))));
        }
        // v0.15.4: fast_slow family — fast tracker (EMA/Holt/AR1/Drift)
        // wrapped by SlowStandardizeWrapper (slow residual variance).
        // Ports skaters' `fast_slow` group. 5 fast trackers × N slow
        // alphas. Difference-based fast tracker skipped (needs a
        // 3-deep composition we don't have as a single leaf yet).
        for &slow_alpha in &self.fast_slow_slow_alphas {
            leaves.push(LeafEnum::Wrapped(Box::new(SlowStandardizeWrapper::new(
                Box::new(EmaLeaf::new(0.3)),
                slow_alpha,
            ))));
            leaves.push(LeafEnum::Wrapped(Box::new(SlowStandardizeWrapper::new(
                Box::new(EmaLeaf::new(0.5)),
                slow_alpha,
            ))));
            leaves.push(LeafEnum::Wrapped(Box::new(SlowStandardizeWrapper::new(
                Box::new(HoltLeaf::new(0.4, 0.2, 1.0)),
                slow_alpha,
            ))));
            leaves.push(LeafEnum::Wrapped(Box::new(SlowStandardizeWrapper::new(
                Box::new(Ar1Leaf::new(0.1)),
                slow_alpha,
            ))));
            leaves.push(LeafEnum::Wrapped(Box::new(SlowStandardizeWrapper::new(
                Box::new(DriftLeaf::new(0.05)),
                slow_alpha,
            ))));
        }
        // PR #4 of #180: seasonal-diff + EMA depth-2 compositions.
        for &(period, alpha) in &self.seasonal_diff_ema {
            leaves.push(LeafEnum::Wrapped(Box::new(SeasonalDifferenceWrapper::new(
                Box::new(EmaLeaf::new(alpha)),
                period,
            ))));
        }
        // PR #4 of #180: diff + EMA depth-2 (period=1 == plain differencing).
        //
        // Fix for issue #195 fourth pathology (drift + seasonal): the
        // period-1 diff-EMA family are excellent 1-step level trackers,
        // and on a trending seasonal series they win the softmax by
        // producing zero-mean differences → flat multi-step forecast
        // ignoring the seasonal component. When the caller has
        // committed to a seasonal period, exclude this family; the
        // seasonal-EMA leaf below is what they wanted.
        if self.seasonal_period.is_none() {
            for &alpha in &self.diff_ema_alphas {
                leaves.push(LeafEnum::Wrapped(Box::new(SeasonalDifferenceWrapper::new(
                    Box::new(EmaLeaf::new(alpha)),
                    1,
                ))));
            }
        }
        // PR #4 of #180: multi-speed drift grid.
        for &alpha in &self.drift_alphas {
            leaves.push(LeafEnum::Drift(mk_drift(alpha)));
        }
        // PR #6 of #180: fractional-diff variants.
        for &(d, am, ad) in &self.frac_diff_variants {
            leaves.push(LeafEnum::FracDiff(FractionalDiffLeaf::new(d, am, ad)));
        }
        // PR #6 of #180: GARCH + EMA composition.
        for &alpha in &self.garch_ema_alphas {
            leaves.push(LeafEnum::Wrapped(Box::new(
                GarchWrappedLeaf::with_defaults(Box::new(EmaLeaf::new(alpha))),
            )));
        }
        // PR #6 of #180: PowerTransform + EMA composition.
        for &(p, alpha) in &self.power_ema {
            leaves.push(LeafEnum::Wrapped(Box::new(PowerTransformWrapper::new(
                Box::new(EmaLeaf::new(alpha)),
                p,
            ))));
        }
        // PR #6 of #180: YJ + EMA composition — the "coordinate prior"
        // (skaters composes YJ only with {diff, ema}; this is the EMA half).
        for &(lam, alpha) in &self.yj_ema {
            leaves.push(LeafEnum::Wrapped(Box::new(YjWrappedLeaf::new(
                Box::new(EmaLeaf::new(alpha)),
                lam,
            ))));
        }
        // PR #6 of #180: YJ + diff + EMA composition — the diff half of
        // skaters' YJ coordinate prior. Same gate as the plain diff-EMA
        // family above: skip when the caller committed to a seasonal
        // period (see issue #195 fourth pathology).
        if self.seasonal_period.is_none() {
            for &(lam, alpha) in &self.yj_diff_ema {
                let inner: Box<dyn Leaf + Send> = Box::new(SeasonalDifferenceWrapper::new(
                    Box::new(EmaLeaf::new(alpha)),
                    1,
                ));
                leaves.push(LeafEnum::Wrapped(Box::new(YjWrappedLeaf::new(inner, lam))));
            }
        }
        // PR #3 of #180: Yeo-Johnson coordinate composition — for each λ,
        // append a wrapped copy of every base leaf so far.
        if !self.yj_coord_lambdas.is_empty() {
            let base_count = leaves.len();
            for &lam in &self.yj_coord_lambdas {
                for i in 0..base_count {
                    let _ = i;
                }
                leaves.push(LeafEnum::Wrapped(Box::new(YjWrappedLeaf::new(
                    Box::new(EmaLeaf::new(self.ema_alpha)),
                    lam,
                ))));
                leaves.push(LeafEnum::Wrapped(Box::new(YjWrappedLeaf::new(
                    Box::new(DriftLeaf::new(self.drift_alpha)),
                    lam,
                ))));
            }
        }
        if let Some(a) = self.intermittent {
            leaves.push(LeafEnum::Intermittent(IntermittentLeaf::new(a)));
        }
        if let Some((p, a)) = self.seasonal_intermittent {
            leaves.push(LeafEnum::SeasonalIntermittent(
                SeasonalIntermittentLeaf::new(p, a),
            ));
        }
        if let Some(a) = self.poisson {
            leaves.push(LeafEnum::Poisson(PoissonLeaf::new(a)));
        }
        if let Some(a) = self.neg_binomial {
            leaves.push(LeafEnum::NegativeBinomial(NegativeBinomialLeaf::new(a)));
        }
        if let Some(a) = self.lognormal {
            leaves.push(LeafEnum::LogNormal(LogNormalLeaf::new(a)));
        }
        if let Some(a) = self.gamma {
            leaves.push(LeafEnum::Gamma(GammaLeaf::new(a)));
        }
        if let Some(a) = self.rectified_normal {
            leaves.push(LeafEnum::RectifiedNormal(RectifiedNormalLeaf::new(a)));
        }
        if let Some(a) = self.zip {
            leaves.push(LeafEnum::Zip(ZeroInflatedPoissonLeaf::new(a)));
        }
        if let Some(a) = self.zinb {
            leaves.push(LeafEnum::Zinb(ZeroInflatedNegativeBinomialLeaf::new(a)));
        }
        if let Some(a) = self.student_t {
            leaves.push(LeafEnum::StudentT(StudentTLeaf::new(a)));
        }
        if let Some(a) = self.beta {
            leaves.push(LeafEnum::Beta(BetaLeaf::new(a)));
        }
        if let Some((a, p)) = self.tweedie {
            leaves.push(LeafEnum::Tweedie(TweedieLeaf::new(a, p)));
        }
        if let Some(a) = self.skew_normal {
            leaves.push(LeafEnum::SkewNormal(SkewNormalLeaf::new(a)));
        }
        if self.discrete_uniform {
            leaves.push(super::leaf_enum::LeafEnum::DiscreteUniform(
                DiscreteUniformLeaf::new(),
            ));
        }
        if let Some(p) = self.seasonal_period {
            leaves.push(super::leaf_enum::LeafEnum::SeasonalEma(mk_seasonal_ema(
                p,
                self.seasonal_alpha,
            )));
        }
        for &p in &self.seasonal_periods_multi {
            leaves.push(super::leaf_enum::LeafEnum::SeasonalEma(mk_seasonal_ema(
                p,
                self.seasonal_alpha,
            )));
        }
        if let Some((p, a)) = self.seasonal_mult {
            leaves.push(super::leaf_enum::LeafEnum::SeasonalMult(mk_seasonal_mult(
                p, a,
            )));
        }
        leaves
    }

    #[allow(dead_code)] // Kept as a convenience for callers that hoist init out of fit().
    fn init_leaves(&mut self) {
        self.init_leaves_maybe_batch(None);
    }

    /// Init with optional batch values for yearly-Trick 1 (batch OLS
    /// initialization of Drift + Holt trends). When `Some(values)` is
    /// passed AND the series appears trending AND is short (N < 60),
    /// Drift/Holt candidates start from the OLS-fitted slope rather
    /// than from zero.
    fn init_leaves_maybe_batch(&mut self, batch_values: Option<&[f64]>) {
        use super::leaf_enum::LeafEnum;
        // Batch init drives ONLY seasonal-EMA / multiplicative-seasonal
        // phase levels now (Drift/Holt were reverted after regressing
        // cif_2016 — see comment in build_base_leaves_with_batch).
        // Seasonal batch init has no gate: computing per-phase means
        // from training data is unambiguously a better start than the
        // cold zero, whenever the caller has committed to a `period`.
        // Require at least 5 obs for a meaningful mean.
        let batch: Option<&[f64]> = batch_values.filter(|v| v.len() >= 5);
        let leaves = self.build_base_leaves_with_batch(batch);
        let leaves = if !self.yj_grid.is_empty() {
            let mut wrapped: Vec<LeafEnum> = Vec::with_capacity(leaves.len() * self.yj_grid.len());
            for lam in self.yj_grid.clone() {
                let per_lambda = self.build_base_leaves_with_batch(batch);
                for l in per_lambda {
                    let boxed: Box<dyn Leaf + Send> = Box::new(l);
                    wrapped.push(LeafEnum::Wrapped(Box::new(YjWrappedLeaf::new(boxed, lam))));
                }
            }
            wrapped
        } else {
            leaves
        };
        self.cum_log_liks = vec![0.0; leaves.len()];
        self.leaves = leaves;
    }

    fn weights(&self) -> Vec<f64> {
        softmax(&self.cum_log_liks)
    }

    fn per_leaf_horizons(&self, horizon: usize) -> Vec<Vec<super::dist::Gaussian>> {
        self.leaves.iter().map(|l| l.predict(horizon)).collect()
    }

    /// Absorb one **transformed-space** observation `y` into all
    /// leaves + terminals + sticky lattice. The shared work of both
    /// the batch `fit()` loop and the public streaming
    /// [`Self::observe`]. O(N_leaves) per call.
    ///
    /// Returns the transformed-space one-step mixture mean at this step
    /// so the caller can compute residuals / fitted values / calibration
    /// snapshots in the original space.
    fn absorb_one(&mut self, y: f64) -> f64 {
        // 1-step predictions from each leaf, before observing y.
        self.scratch_per_leaf.clear();
        for l in self.leaves.iter() {
            self.scratch_per_leaf.push(l.predict_one());
        }
        let per_leaf = self.scratch_per_leaf.as_slice();
        softmax_into(&self.cum_log_liks, &mut self.scratch_weights);
        let weights = self.scratch_weights.as_slice();
        let mixture_is_empty = per_leaf.is_empty();
        let mixture_mean: f64 = weights
            .iter()
            .zip(per_leaf.iter())
            .map(|(w, g)| w * g.mean)
            .sum();
        // Score + absorb per leaf. Learning-rate shrinkage and
        // log-clamp applied to the cumulative-weight update.
        let eta = eta_schedule(self.learning_rate, self.n_obs);
        let clamp = self.log_clamp;
        let sw = self.scoring_window;
        if let Some(w_size) = sw {
            if self.scoring_window_hist.len() != self.leaves.len() {
                self.scoring_window_hist = (0..self.leaves.len())
                    .map(|_| std::collections::VecDeque::with_capacity(w_size + 1))
                    .collect();
            }
        }
        for (i, leaf) in self.leaves.iter_mut().enumerate() {
            let g = per_leaf[i];
            let lp_raw = g.logpdf(y);
            if lp_raw.is_finite() {
                let lp_clamped = if lp_raw < clamp { clamp } else { lp_raw };
                if let Some(w_size) = sw {
                    let hist = &mut self.scoring_window_hist[i];
                    hist.push_back(lp_clamped);
                    self.cum_log_liks[i] += lp_clamped;
                    if hist.len() > w_size {
                        if let Some(old) = hist.pop_front() {
                            self.cum_log_liks[i] -= old;
                        }
                    }
                } else {
                    self.cum_log_liks[i] += eta * lp_clamped;
                }
            }
            leaf.observe(y);
        }
        // Terminals: residual is transformed-space (y - mixture_mean).
        let residual = if mixture_is_empty {
            0.0
        } else {
            y - mixture_mean
        };
        if let Some(t) = self.terminal.as_mut() {
            t.observe(residual);
        }
        if let Some(t) = self.terminal_crps.as_mut() {
            t.observe(residual);
        }
        self.n_obs += 1;
        mixture_mean
    }

    /// Streaming observe — absorb a **single original-space** observation
    /// into all leaves + terminals + sticky lattice.
    ///
    /// This is the O(N_leaves) counterpart to [`Forecaster::fit`]'s batch
    /// loop. Call this once per new observation to update model state
    /// without a full refit. Skaters' equivalent primitive:
    /// `f(y, state) -> (dist, new_state)`.
    ///
    /// # Requirements
    /// - [`Forecaster::fit`] must have been called first (initializes
    ///   the leaf pool). Streaming from empty state is not supported.
    /// - This path **skips** batch-only features:
    ///     - Yeo-Johnson transform (`with_yeo_johnson*`)
    ///     - Exog OLS pre-regression
    ///     - Per-horizon calibration snapshots
    ///     - Fitted-values / residuals bookkeeping
    ///
    ///   Configure the model without these when planning to stream.
    ///
    /// # Errors
    /// - `FitRequired` if the leaf pool hasn't been initialized.
    pub fn observe(&mut self, y: f64) -> Result<()> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if !y.is_finite() {
            return Ok(());
        }
        // Absorb (transformed space == original space when YJ disabled,
        // which the streaming API requires).
        let _ = self.absorb_one(y);
        // Sticky lattice always tracks original-space values.
        if let Some(s) = self.sticky.as_mut() {
            s.observe(y);
        }
        Ok(())
    }

    /// Streaming observe of an entire slice — convenience wrapper around
    /// [`Self::observe`]. Same O(N_obs · N_leaves) cost as a batch fit
    /// on the same-length window, but *incremental*: previous
    /// observations are preserved in state.
    pub fn observe_slice(&mut self, ys: &[f64]) -> Result<()> {
        for &y in ys {
            self.observe(y)?;
        }
        Ok(())
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

        // Fev-27 follow-up (#9): STL leaf auto-detection was REMOVED.
        // Adding an auto-STL leaf caused a large tourism_monthly
        // regression on the fev-27 panel (MASE 2.34 → 3.08, +31 %) —
        // the STL leaf's linear-trend extrapolation is aggressive at
        // long horizons, compounding on short-history seasonal panels
        // (150-300 obs). Aggregate fev-27 also regressed (geomean
        // MASE 6.085 → 6.186). See `docs/SOTA_POSITIONING.md`
        // "Deferred / future work" for the full story.
        //
        // `StlDecompLeaf` and the `stl_period` field remain so
        // callers can opt in via `.with_stl(period)` on data they
        // know behaves well with STL. It is NOT auto-enabled by
        // `.auto()` or `.skaters()`.

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
            // Fev-27 follow-up (#9): STL leaf auto-detection REMOVED.
            // See earlier note above and docs/SOTA_POSITIONING.md.
            // Available opt-in via `.with_stl(period)`.
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

        // Fix for issue #195 third pathology: intermittent-bursty
        // series (case study `FOODS_3_444_WI_2`: 37% zeros alternating
        // with 400-1900 spikes) that .skaters() left un-Crostoned
        // because the auto-selector doesn't run under .skaters(). Runs
        // outside the `use_auto` gate so both paths benefit. Uses the
        // same 0.4 threshold as .auto()'s existing zero_fraction gate.
        if self.intermittent.is_none() && !values.is_empty() {
            let zero_count = values.iter().filter(|y| y.abs() < 1e-9).count();
            let zero_frac = zero_count as f64 / values.len() as f64;
            if zero_frac > 0.4 {
                self.intermittent = Some(0.1);
            }
            // If we've committed to a seasonal period, prefer the
            // seasonal Croston variant which retains the phase shape.
            if let Some(period) = self.seasonal_period {
                if self.seasonal_intermittent.is_none() && zero_frac > 0.3 && period >= 2 {
                    self.seasonal_intermittent = Some((period, 0.1));
                }
            }
        }

        // Fix for issue #195 second pathology: all-positive training
        // data (retail counts, M5-monthly-shape series) auto-enables
        // the non-negative clamp on the output mixture — regardless of
        // whether `use_auto` is set, since `.skaters()` skips the auto
        // block but is even more likely to produce negatives (measured:
        // 1.9% of M5-monthly `.auto()` series produced ≥1 negative
        // forecast; 5.0% for `.skaters()`). Runs on both `.auto()` and
        // `.skaters()` paths.
        if !self.non_negative
            && !values.is_empty()
            && values.iter().all(|y| y.is_finite() && *y >= 0.0)
            && values.iter().any(|y| *y > 0.0)
        {
            self.non_negative = true;
        }

        // Seasonal batch init is opt-in via `.with_seasonal_batch_init()`.
        // When enabled and a period is set, pre-fills the seasonal-EMA /
        // multiplicative-seasonal leaves' phase levels from the last
        // training cycle to close the softmax cold-start handicap.
        // Documented on N=48 monthly (period=12) synthetic data.
        //
        // Note: Drift/Holt build closures IGNORE `batch` regardless
        // (yearly-Trick 1 regressed cif_2016 by 280% — see
        // docs/ACCURACY_AUDIT.md).
        let batch_arg = if self.seasonal_batch_init {
            Some(values)
        } else {
            None
        };
        self.init_leaves_maybe_batch(batch_arg);
        // Fev-27 follow-up: auto-gate sticky. When `.skaters()` set
        // `sticky_auto_gate = true`, decide sticky based on training
        // data characteristics. Discrete-count-like → keep sticky
        // (M5, dominick). Continuous smooth → disable (m1_yearly,
        // tourism_*, cif_2016 all had catastrophic WQL blowups
        // with sticky on).
        if self.sticky_auto_gate && self.sticky.is_some() && !looks_discrete_count(values) {
            self.sticky = None;
        }
        // Accuracy-audit #3a: warm-start the terminal σ from the
        // training values' MAD (1.4826 × median absolute deviation
        // from the median). Skips the terminal's 1/n bootstrap on
        // short-history panels where the first 30 obs' EWMA would
        // otherwise miscalibrate the mixture spread.
        if values.len() >= 10 {
            let mad = compute_mad(values);
            if mad > 0.0 && mad.is_finite() {
                if let Some(t) = self.terminal.as_mut() {
                    t.warm_start(mad, 30);
                }
            }
        }
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
        // Accuracy-audit #2: multi-horizon retrospective scoring
        // snapshots. Only populated when `self.multi_h_scoring` is on.
        // Structure: (step, per_leaf_h_predictions[leaf_idx][h_idx]).
        let mut mh_snapshots: Vec<(usize, Vec<Vec<Gaussian>>)> = Vec::new();
        // Snapshot cadence: every 20 steps starting from step 60.
        // Limited to `values.len() / 15` snapshots to bound cost.
        let mh_stride: usize = 20;
        let mh_horizon: usize = self
            .scoring_horizon
            .unwrap_or_else(|| per_h_horizon.clamp(4, 24));

        for (step, &y_orig) in values.iter().enumerate() {
            let y = match yj {
                Some(l) => yj_forward(y_orig, l),
                None => y_orig,
            };

            // Accuracy-audit #2: multi-horizon scoring snapshot.
            if self.multi_h_scoring
                && step >= 60
                && step % mh_stride == 0
                && step + mh_horizon <= values.len()
                && mh_snapshots.len() < 200
            {
                let per_leaf_h: Vec<Vec<Gaussian>> =
                    self.leaves.iter().map(|l| l.predict(mh_horizon)).collect();
                mh_snapshots.push((step, per_leaf_h));
            }

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
            // Perf: fill the reusable scratch buffer (sized to leaf count
            // in fit's initialization) instead of `collect`ing a fresh Vec.
            self.scratch_per_leaf.clear();
            for l in self.leaves.iter() {
                self.scratch_per_leaf.push(l.predict_one());
            }
            let per_leaf = self.scratch_per_leaf.as_slice();
            // Accuracy-audit #1: stacking history snapshot. Store each
            // leaf's 1-step-ahead prediction MEAN for the OLS solve at
            // end of fit. Skip on very short series (< 60 obs) where
            // stacking would overfit.
            if self.stacking_enabled && values.len() >= 60 {
                if self.predictions_history.len() != per_leaf.len() {
                    self.predictions_history.clear();
                    self.predictions_history
                        .resize(per_leaf.len(), Vec::with_capacity(values.len()));
                }
                for (i, g) in per_leaf.iter().enumerate() {
                    self.predictions_history[i].push(g.mean);
                }
            }
            // Perf: softmax weights into a reused scratch buffer to skip
            // the per-iteration `Vec<f64>` alloc.
            softmax_into(&self.cum_log_liks, &mut self.scratch_weights);
            let weights = self.scratch_weights.as_slice();

            // Perf: inline mixture mean / variance instead of building a
            // GaussianMixture struct — we only need mean/std/is_empty here,
            // not the components vec.
            let mixture_is_empty = per_leaf.is_empty();
            let mixture_mean: f64 = weights
                .iter()
                .zip(per_leaf.iter())
                .map(|(w, g)| w * g.mean)
                .sum();
            // Fitted / residuals: expose in ORIGINAL space so downstream
            // consumers (Explanation, tests, callers computing MAE) see
            // the same scale as the training values.
            let fitted_orig = if mixture_is_empty {
                y_orig
            } else {
                let m_trans = mixture_mean;
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
                let (mu_trans, sigma_trans) = if mixture_is_empty {
                    (y, 1.0)
                } else {
                    // Inline mixture variance to skip mixture allocation.
                    let mu = mixture_mean;
                    let var: f64 = weights
                        .iter()
                        .zip(per_leaf.iter())
                        .map(|(w, g)| w * (g.std * g.std + (g.mean - mu).powi(2)))
                        .sum();
                    (mu, var.sqrt())
                };
                self.predictive_stds.push(sigma_trans);
                self.predictive_residuals_trans.push(y - mu_trans);
            }

            // Score each leaf on this y, then absorb.
            // PR #5 of #180: apply learning_rate shrinkage and log-clamp
            // to the cumulative-weight update — skaters' XGBoost-style
            // ensemble regularization. Defaults (η=1.0, clamp=−∞)
            // preserve the historical behavior.
            // Fev-27 follow-up (#5): warmup schedule for η. For
            // n < 30 obs we use η=1.0 (fast learning — the softmax
            // needs to move away from uniform quickly). For
            // 30 <= n < 100 linearly decay to self.learning_rate.
            // For n >= 100 hold at self.learning_rate. Prevents the
            // short-history yearly regression that Fix B introduced.
            //
            // Yearly-Trick 3 was REVERTED after cif_2016 regression
            // caused by Trick 1. May have contributed independently
            // to WQL regression; safer to revert together and revisit
            // separately with proper isolation.
            let eta = eta_schedule(self.learning_rate, self.n_obs);
            let clamp = self.log_clamp;
            let sw = self.scoring_window;
            if let Some(w_size) = sw {
                if self.scoring_window_hist.len() != self.leaves.len() {
                    self.scoring_window_hist = (0..self.leaves.len())
                        .map(|_| std::collections::VecDeque::with_capacity(w_size + 1))
                        .collect();
                }
            }
            for (i, leaf) in self.leaves.iter_mut().enumerate() {
                let g = per_leaf[i];
                let lp_raw = g.logpdf(y);
                if lp_raw.is_finite() {
                    let lp_clamped = if lp_raw < clamp { clamp } else { lp_raw };
                    if let Some(w_size) = sw {
                        let hist = &mut self.scoring_window_hist[i];
                        hist.push_back(lp_clamped);
                        self.cum_log_liks[i] += lp_clamped;
                        if hist.len() > w_size {
                            if let Some(old) = hist.pop_front() {
                                self.cum_log_liks[i] -= old;
                            }
                        }
                    } else {
                        self.cum_log_liks[i] += eta * lp_clamped;
                    }
                }
                leaf.observe(y);
            }
            // Terminal scale-mixture: absorb the residual (transformed
            // space) between the softmax mixture mean and y. This leaf
            // tracks the residual's own distribution independently of
            // the individual leaves' Gaussian assumptions.
            let residual = if mixture_is_empty {
                0.0
            } else {
                y - mixture_mean
            };
            if let Some(t) = self.terminal.as_mut() {
                t.observe(residual);
            }
            // PR #7 of #180: CRPS-gradient terminal in parallel. Absorbs
            // the same residual; forecast_dist picks whichever is set
            // (crps takes precedence when both are configured).
            if let Some(t) = self.terminal_crps.as_mut() {
                t.observe(residual);
            }
            // PR #7 of #180: sticky lattice — update the recency table
            // with the ORIGINAL-space y (not the transformed value), so
            // atoms fire on actual observation values.
            if let Some(s) = self.sticky.as_mut() {
                s.observe(y_orig);
            }
            self.n_obs += 1;
        }

        // Accuracy-audit #2: multi-horizon retrospective scoring pass.
        // For each snapshot at step t, score each leaf's h-step
        // prediction against `values[t + h - 1]` (the actual). Weight
        // the contribution by `η / h` so 1-step still dominates but
        // long-horizon accuracy shifts the ensemble.
        if self.multi_h_scoring && !mh_snapshots.is_empty() {
            let eta = self.learning_rate;
            let clamp = self.log_clamp;
            for (step, per_leaf_h) in &mh_snapshots {
                for h in 1..=mh_horizon {
                    let target_step = step + h - 1;
                    if target_step >= values.len() {
                        break;
                    }
                    let y_target = match yj {
                        Some(l) => yj_forward(values[target_step], l),
                        None => values[target_step],
                    };
                    if !y_target.is_finite() {
                        continue;
                    }
                    let h_weight = eta / h as f64;
                    for (leaf_idx, preds) in per_leaf_h.iter().enumerate() {
                        if leaf_idx >= self.cum_log_liks.len() {
                            break;
                        }
                        if let Some(g) = preds.get(h - 1) {
                            let lp = g.logpdf(y_target);
                            if lp.is_finite() {
                                let lp_c = if lp < clamp { clamp } else { lp };
                                self.cum_log_liks[leaf_idx] += h_weight * lp_c;
                            }
                        }
                    }
                }
            }
        }

        // Accuracy-audit #1: ensemble stacking solve. If we collected
        // per-leaf predictions during the fit loop and have enough
        // observations, solve OLS for the blend weights and project
        // onto the non-negative simplex.
        if self.stacking_enabled && !self.predictions_history.is_empty() {
            // Skip the first BURN steps so leaves have warmed up.
            const BURN: usize = 30;
            let n_leaves = self.predictions_history.len();
            let n_steps = self.predictions_history[0].len();
            if n_steps > BURN + n_leaves {
                let stacking_weights = solve_stacking(&self.predictions_history, values, BURN);
                self.stacking_weights = Some(stacking_weights);
            }
            // Free the history buffer — only needed once.
            self.predictions_history.clear();
            self.predictions_history.shrink_to_fit();
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
        // Accuracy-audit #1: prefer stacking weights (learned by OLS
        // on training predictions) over softmax when available.
        // Stacking directly optimizes point-forecast MSE, closer to
        // the MASE / WAPE metrics than softmax's 1-step-log-likelihood.
        let softmax_weights = if let Some(sw) = self.stacking_weights.as_ref() {
            sw.clone()
        } else {
            self.weights()
        };
        let per_leaf = self.per_leaf_horizons(horizon);
        let scale = self.calibration_scale;
        let per_h = &self.calibration_scale_per_h;
        let yj = self.fitted_yj_lambda;
        let trans_range = self.yj_trans_range;
        let non_negative = self.non_negative;
        Ok((0..horizon)
            .map(|h| {
                let m = blend_horizon(&softmax_weights, &per_leaf, h);
                // Terminal scale-mixture: replace the softmax blend's
                // shape with a fixed-scale mixture centered at its mean.
                // Mean-preserving; only reshapes the density.
                // PR #7 of #180: CRPS terminal takes precedence over
                // the likelihood-EM terminal when both are configured.
                let m = if let Some(t) = self.terminal_crps.as_ref() {
                    if t.n_obs() > 5 && !m.is_empty() {
                        t.predict_shifted(m.mean())
                    } else {
                        m
                    }
                } else if let Some(t) = self.terminal.as_ref() {
                    if t.n_obs() > 5 && !m.is_empty() {
                        t.predict_shifted(m.mean())
                    } else {
                        m
                    }
                } else {
                    m
                };
                // Fev-27 follow-up (#3): multi-horizon terminal σ scaling.
                // The terminal tracks 1-step residual variance; at h > 1
                // the true predictive spread grows. Assume random-walk
                // residuals and scale std by √(h+1). Closes WQL underfit
                // at long horizons (h + 1 since the closure's `h` is
                // 0-based).
                //
                // Accuracy-audit #5: if terminal tracks AR(1) φ, use
                // `√((1 − φ^(2(h+1))) / (1 − φ²))` — the true AR(1)
                // h-step predictive std. Falls back to `√(h+1)` when
                // φ ≈ 0 (IID case). Tighter spread on mean-reverting
                // residuals (φ < 0), wider on persistent (φ > 0).
                let m = if h > 0 {
                    let scale = self
                        .terminal
                        .as_ref()
                        .map(|t| t.h_step_std_scale(h + 1))
                        .unwrap_or_else(|| ((h + 1) as f64).sqrt());
                    let inflated = m
                        .components
                        .into_iter()
                        .map(|(w, g)| (w, super::dist::Gaussian::new(g.mean, g.std * scale)));
                    GaussianMixture::new(inflated)
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
                let mix = GaussianMixture::new(components);
                // PR #7 of #180: sticky lattice — project onto revisited
                // exact values. No-op if no atoms have fired. Fix A of
                // fev-27 follow-up: horizon-decayed atom mass (`h + 1`
                // since the closure's `h` is 0-based).
                if let Some(s) = self.sticky.as_ref() {
                    s.project(&mix, h + 1)
                } else {
                    mix
                }
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

    /// Streaming `observe()` should produce bit-identical predictions
    /// to a batch `fit()` on the same total window. Two forecasters:
    /// (A) fit on values[0..k], stream values[k..n];
    /// (B) fit on values[0..n] in one shot.
    /// Their `forecast_dist(1)` mean must match to ~1e-9.
    #[test]
    fn streaming_observe_matches_batch_fit() {
        let ts_full = ts_ar1(250, 0.6);
        let values = ts_full.primary_values().to_vec();
        let split = 200;
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let stamps_a: Vec<_> = (0..split)
            .map(|i| base + Duration::hours(i as i64))
            .collect();
        let ts_a = TimeSeries::univariate(stamps_a, values[..split].to_vec()).unwrap();

        // Path A: fit on first 200, stream the last 50.
        let mut fa = LaplaceForecaster::new();
        fa.fit(&ts_a).unwrap();
        for &y in &values[split..] {
            fa.observe(y).unwrap();
        }
        let m_a = fa.forecast_dist(1).unwrap()[0].mean();

        // Path B: batch fit on all 250.
        let mut fb = LaplaceForecaster::new();
        fb.fit(&ts_full).unwrap();
        let m_b = fb.forecast_dist(1).unwrap()[0].mean();

        assert!(
            (m_a - m_b).abs() < 1e-9,
            "streaming ({m_a:.9}) != batch ({m_b:.9})"
        );
    }

    #[test]
    fn observe_returns_error_before_fit() {
        let mut f = LaplaceForecaster::new();
        assert!(f.observe(1.0).is_err());
    }

    #[test]
    fn observe_ignores_nan() {
        let ts = ts_ar1(100, 0.5);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        let m_before = f.forecast_dist(1).unwrap()[0].mean();
        // NaN / inf are silently ignored (matches leaf-level behavior).
        f.observe(f64::NAN).unwrap();
        f.observe(f64::INFINITY).unwrap();
        let m_after = f.forecast_dist(1).unwrap()[0].mean();
        assert!(
            (m_before - m_after).abs() < 1e-12,
            "NaN observe changed state: {m_before} vs {m_after}"
        );
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
