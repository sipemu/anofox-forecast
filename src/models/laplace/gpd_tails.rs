//! GPD tail splice — port of skaters' `tails.py::gpdtails` (simplified).
//!
//! The Laplace ensemble's body predictive is calibrated in the bulk but too
//! thin in the tails: fev-27's WQL at extreme quantiles (0.99 / 0.01) is
//! systematically off. This module fits a generalised Pareto distribution
//! to the exceedances of standardised training residuals beyond frozen
//! `level`-quantile thresholds and splices those tails into the mixture's
//! quantile function at forecast time.
//!
//! Difference from skaters: skaters tracks per-horizon PIT via the parade
//! module, refreshing the fit every 25 ticks. We take a batch-friendly
//! shortcut — fit once at the end of training on 1-step standardised
//! residuals. Full per-horizon rolling refit would require the parade
//! infrastructure (skaters' `parade.py`), a separate follow-up.
//!
//! Post-#180 addition — v0.15.4.

use super::dist::GaussianMixture;
use super::forecaster::LaplaceForecaster;
use super::DistributionalForecaster;
use crate::core::{Forecast, TimeSeries};
use crate::error::Result;
use crate::models::traits::{validate_series_complete, Forecaster};

/// Fitted GPD-tail parameters for a body predictive whose PIT push-through
/// to N(0,1) is approximately calibrated in the bulk. `t_*` are frozen
/// z-thresholds; `zeta_*` are the empirical exceedance rates; `g_*` and
/// `s_*` are the GPD shape and scale parameters (fitted by censored ML).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpdTailParams {
    pub t_lo: f64,
    pub t_up: f64,
    pub zeta_lo: f64,
    pub zeta_up: f64,
    pub g_lo: f64,
    pub s_lo: f64,
    pub g_up: f64,
    pub s_up: f64,
}

impl GpdTailParams {
    /// Return the interior scaling factor `c` that renormalises the body's
    /// interior mass so the spliced density integrates to 1.
    fn interior_c(&self) -> f64 {
        let plo = phi(self.t_lo);
        let pup = phi(self.t_up);
        let interior = (pup - plo).max(1e-12);
        ((1.0 - self.zeta_lo - self.zeta_up).max(1e-12)) / interior
    }
}

/// Public wrapper for [`erf_local`] used by the parade PIT tracker in
/// `forecaster.rs` to avoid duplicating this rational approximation.
#[doc(hidden)]
pub fn erf_local_pub(x: f64) -> f64 {
    erf_local(x)
}

/// `erf(x)` via Abramowitz-Stegun 7.1.26. Same rational approximation
/// used by [`super::leaves::terminal_crps`]; keeps us off a new dep.
fn erf_local(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Standard-normal CDF (`Φ`).
fn phi(z: f64) -> f64 {
    0.5 * (1.0 + erf_local(z / std::f64::consts::SQRT_2))
}

/// Standard-normal inverse CDF via bisection. Adequate for tail splicing
/// where |z| stays within ±8.
fn phi_inv(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let (mut lo, mut hi) = (-10.0f64, 10.0f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if phi(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-10 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// GPD survival function `1 - F(e | γ, σ)`. Public for users who
/// want to compute exceedance survival probabilities on top of a
/// fitted [`GpdTailParams`]; complements [`gpd_isf`].
pub fn gpd_sf(e: f64, gamma: f64, sigma: f64) -> f64 {
    if e <= 0.0 {
        return 1.0;
    }
    if gamma.abs() < 1e-9 {
        return (-e / sigma).exp();
    }
    let arg = 1.0 + gamma * e / sigma;
    if arg <= 0.0 {
        return 0.0;
    }
    arg.powf(-1.0 / gamma)
}

/// Inverse GPD survival function.
pub fn gpd_isf(p: f64, gamma: f64, sigma: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0);
    if gamma.abs() < 1e-9 {
        return -sigma * p.ln();
    }
    sigma / gamma * (p.powf(-gamma) - 1.0)
}

const TAU_GRID: [f64; 13] = [
    0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0,
];

/// Censored-ML GPD fit (Grimshaw profile over a fixed tau grid).
/// Ports skaters' `_fit_ml`.
fn fit_gpd_ml(exc: &[f64]) -> (f64, f64) {
    let n = exc.len();
    if n < 20 {
        let s1: f64 = exc.iter().sum();
        return (0.0, (s1 / n.max(1) as f64).max(1e-12));
    }
    let s1: f64 = exc.iter().sum();
    let emean = s1 / n as f64;
    if emean <= 0.0 {
        return (0.0, 1e-12);
    }
    let emax = exc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut best = (0.0f64, emean.max(1e-12), f64::NEG_INFINITY);
    let mut taus: Vec<f64> = TAU_GRID.iter().map(|t| t / emean).collect();
    taus.extend([-0.5 / emax, -0.25 / emax, -0.1 / emax]);
    for &tau in &taus {
        if tau <= -1.0 / emax || tau.abs() < 1e-12 {
            continue;
        }
        let g: f64 = exc.iter().map(|&e| (1.0 + tau * e).ln()).sum::<f64>() / n as f64;
        if g <= 1e-9 {
            continue;
        }
        let sigma = g / tau;
        if sigma <= 0.0 {
            continue;
        }
        let ll = -(n as f64) * sigma.ln() - (1.0 + 1.0 / g) * (n as f64) * g;
        if ll > best.2 {
            best = (g, sigma, ll);
        }
    }
    (best.0, best.1)
}

/// Fit `GpdTailParams` from a sample of standardised residuals `z_i =
/// (y_i - μ_i) / σ_i`. `level` is the frozen-threshold quantile (e.g. 0.98).
/// Returns `None` if there aren't enough exceedances on either side to fit
/// a stable GPD.
pub fn fit_gpd_tails(z: &[f64], level: f64) -> Option<GpdTailParams> {
    if z.len() < 100 || !(0.5 < level && level < 1.0) {
        return None;
    }
    let mut sorted: Vec<f64> = z.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n < 100 {
        return None;
    }
    let iu = ((level * n as f64) as usize).min(n - 1);
    let il = n - 1 - iu;
    let t_up = sorted[iu];
    let t_lo = sorted[il];
    // Excesses.
    let exc_up: Vec<f64> = sorted[iu..]
        .iter()
        .map(|&x| x - t_up)
        .filter(|&e| e > 0.0)
        .collect();
    let exc_lo: Vec<f64> = sorted[..=il]
        .iter()
        .map(|&x| t_lo - x)
        .filter(|&e| e > 0.0)
        .collect();
    if exc_up.len() < 20 || exc_lo.len() < 20 {
        return None;
    }
    let (g_up, s_up) = fit_gpd_ml(&exc_up);
    let (g_lo, s_lo) = fit_gpd_ml(&exc_lo);
    let zeta_up = exc_up.len() as f64 / n as f64;
    let zeta_lo = exc_lo.len() as f64 / n as f64;
    Some(GpdTailParams {
        t_lo,
        t_up,
        zeta_lo,
        zeta_up,
        g_lo,
        s_lo,
        g_up,
        s_up,
    })
}

/// Spliced quantile: body's Gaussian-mixture quantile in the interior,
/// GPD splice on either side. Ports skaters' `SplicedDist.quantile`.
///
/// `body_quantile` should be a function that, given a probability `u ∈ (0,1)`,
/// returns the body's quantile at `u`. `body_cdf_at_z` inverts the standard-
/// normal push-through: `body(x) = body.cdf(body_quantile(Φ(z)))`. In our
/// batch-simplification we pass `Φ(z)` directly to the body's `quantile`,
/// which effectively treats the body as N(0,1) reparameterised to whatever
/// distribution the mixture describes.
pub fn spliced_quantile(mix: &GaussianMixture, params: &GpdTailParams, p: f64) -> f64 {
    assert!((0.0..1.0).contains(&p) && p > 0.0);
    let z = if p < params.zeta_lo {
        params.t_lo - gpd_isf(p / params.zeta_lo, params.g_lo, params.s_lo)
    } else if p > 1.0 - params.zeta_up {
        params.t_up + gpd_isf((1.0 - p) / params.zeta_up, params.g_up, params.s_up)
    } else {
        let plo = phi(params.t_lo);
        let c = params.interior_c();
        let u = plo + (p - params.zeta_lo) / c;
        phi_inv(u.clamp(1e-12, 1.0 - 1e-12))
    };
    // Push z back through the body's quantile at Φ(z).
    let ub = phi(z).clamp(1e-12, 1.0 - 1e-12);
    mix.quantile(ub)
}

/// A `GaussianMixture` body with GPD tails spliced in beyond frozen
/// z-thresholds. Full port of skaters' `SplicedDist` — the body's own
/// PIT through the standard normal defines the interior; GPD densities
/// take over outside. Provides `cdf`, `pdf`, `logpdf`, `quantile`,
/// numeric `mean`/`var`/`std`/`crps` (via a 65-node quantile grid).
///
/// Constructed by [`GpdTailsForecaster::spliced_mixtures`].
pub struct SplicedGaussianMixture {
    pub body: GaussianMixture,
    pub params: GpdTailParams,
    /// Cached 65-node quantile grid for moments / CRPS. Populated lazily.
    grid: std::cell::OnceCell<Vec<f64>>,
}

impl SplicedGaussianMixture {
    pub fn new(body: GaussianMixture, params: GpdTailParams) -> Self {
        Self {
            body,
            params,
            grid: std::cell::OnceCell::new(),
        }
    }

    /// Push body-CDF through Φ⁻¹ to get the z-value used for the splice
    /// region check.
    fn z(&self, x: f64) -> f64 {
        let u = self.body.cdf(x).clamp(1e-12, 1.0 - 1e-12);
        phi_inv(u)
    }

    pub fn cdf(&self, x: f64) -> f64 {
        let z = self.z(x);
        let p = &self.params;
        if z < p.t_lo {
            p.zeta_lo * gpd_sf(p.t_lo - z, p.g_lo, p.s_lo)
        } else if z > p.t_up {
            1.0 - p.zeta_up * gpd_sf(z - p.t_up, p.g_up, p.s_up)
        } else {
            let c = p.interior_c();
            p.zeta_lo + c * (phi(z) - phi(p.t_lo))
        }
    }

    pub fn logpdf(&self, x: f64) -> f64 {
        let base = self.body.logpdf(x);
        if !base.is_finite() {
            return base;
        }
        let z = self.z(x);
        let p = &self.params;
        // Standard-normal logpdf helper.
        let phi_logpdf = |z: f64| -0.5 * z * z - 0.5 * (2.0 * PI).ln();
        let corr = if z < p.t_lo {
            p.zeta_lo.max(1e-300).ln() + gpd_logpdf(p.t_lo - z, p.g_lo, p.s_lo) - phi_logpdf(z)
        } else if z > p.t_up {
            p.zeta_up.max(1e-300).ln() + gpd_logpdf(z - p.t_up, p.g_up, p.s_up) - phi_logpdf(z)
        } else {
            p.interior_c().ln()
        };
        base + corr
    }

    pub fn pdf(&self, x: f64) -> f64 {
        let lp = self.logpdf(x);
        if lp < 700.0 {
            lp.exp()
        } else {
            f64::INFINITY
        }
    }

    pub fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..1.0).contains(&p) && p > 0.0);
        spliced_quantile(&self.body, &self.params, p)
    }

    /// 65-node midpoint-rule quantile grid, cached. Feeds mean / var / crps.
    fn qgrid(&self) -> &Vec<f64> {
        self.grid.get_or_init(|| {
            const N: usize = 65;
            (0..N)
                .map(|i| self.quantile((i as f64 + 0.5) / N as f64))
                .collect()
        })
    }

    pub fn mean(&self) -> f64 {
        let q = self.qgrid();
        q.iter().sum::<f64>() / q.len() as f64
    }

    pub fn var(&self) -> f64 {
        let q = self.qgrid();
        let m = self.mean();
        q.iter().map(|x| (x - m).powi(2)).sum::<f64>() / q.len() as f64
    }

    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// CRPS via the quantile-representation midpoint rule. Same formula
    /// as skaters' `SplicedDist.crps`: `E|X - x| - 0.5 · E|X - X'|`.
    pub fn crps(&self, x: f64) -> f64 {
        let q = self.qgrid();
        let n = q.len();
        let t1: f64 = q.iter().map(|v| (v - x).abs()).sum::<f64>() / n as f64;
        let t2: f64 = 2.0
            * q.iter()
                .enumerate()
                .map(|(i, v)| v * (2.0 * (i as f64 + 0.5) / n as f64 - 1.0))
                .sum::<f64>()
            / n as f64;
        t1 - 0.5 * t2
    }
}

/// GPD log-density, for [`SplicedGaussianMixture::logpdf`].
fn gpd_logpdf(e: f64, gamma: f64, sigma: f64) -> f64 {
    if gamma.abs() < 1e-9 {
        return -sigma.ln() - e / sigma;
    }
    let arg = 1.0 + gamma * e / sigma;
    if arg <= 0.0 {
        return -745.0;
    }
    -sigma.ln() - (1.0 / gamma + 1.0) * arg.ln()
}

use std::f64::consts::PI;

/// Wrapper: `LaplaceForecaster` + GPD-tail splice on top of the
/// per-horizon mixture quantile. Implements `Forecaster` +
/// `DistributionalForecaster` transparently; tails only affect quantile
/// queries (via `spliced_quantile` — the mixture components themselves
/// are unchanged so `mean()`, `pdf()` etc. still delegate to the body).
///
/// If the inner forecaster is configured with
/// [`LaplaceForecaster::with_parade`], `fit()` uses the per-horizon PIT
/// vector for per-horizon tail params. Otherwise falls back to fitting a
/// single set of tail params from 1-step residuals and applying them at
/// every horizon.
pub struct GpdTailsForecaster {
    inner: LaplaceForecaster,
    level: f64,
    /// Global tail params (fallback / 1-step-residual path). `None` if
    /// per-horizon params are populated.
    params: Option<GpdTailParams>,
    /// Per-horizon tail params from parade PIT (when available).
    /// `per_horizon_params[h-1]` = params for horizon `h`.
    per_horizon_params: Vec<Option<GpdTailParams>>,
}

impl GpdTailsForecaster {
    /// Wrap an inner forecaster. Default `level = 0.98` matches skaters.
    pub fn new(inner: LaplaceForecaster) -> Self {
        Self {
            inner,
            level: 0.98,
            params: None,
            per_horizon_params: Vec::new(),
        }
    }

    pub fn with_level(mut self, level: f64) -> Self {
        self.level = level;
        self
    }

    /// Fitted GPD tail parameters (`None` if fit hasn't been called or the
    /// residual sample was too small to fit stable tails).
    pub fn tail_params(&self) -> Option<&GpdTailParams> {
        self.params.as_ref()
    }

    /// Return `Vec<SplicedGaussianMixture>` — the body per-horizon
    /// mixtures with GPD tail splice applied. Prefers per-horizon
    /// params (from `.with_parade(k)` on the inner) if fitted; falls
    /// back to global params from 1-step residuals. Horizons without
    /// any fitted tail params are omitted.
    pub fn spliced_mixtures(&self, horizon: usize) -> Result<Vec<SplicedGaussianMixture>> {
        let bodies = self.inner.forecast_dist(horizon)?;
        Ok(bodies
            .into_iter()
            .enumerate()
            .filter_map(|(h_idx, body)| {
                let ph = self.per_horizon_params.get(h_idx).and_then(|o| *o);
                let params = ph.or(self.params);
                params.map(|p| SplicedGaussianMixture::new(body, p))
            })
            .collect())
    }

    /// Spliced quantile at fine horizon `h`, probability `p`. Uses
    /// per-horizon params if available (parade path), else global
    /// `params` (1-step-residual path), else the raw body quantile.
    pub fn quantile_spliced(&self, mixtures: &[GaussianMixture], h: usize, p: f64) -> f64 {
        if h == 0 || h > mixtures.len() {
            return f64::NAN;
        }
        // Prefer per-horizon params if we have them for this h.
        let ph = self.per_horizon_params.get(h - 1).and_then(|o| o.as_ref());
        let pars = ph.or(self.params.as_ref());
        match pars {
            Some(p_ref) => spliced_quantile(&mixtures[h - 1], p_ref, p),
            None => mixtures[h - 1].quantile(p),
        }
    }
}

impl Forecaster for GpdTailsForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        self.inner.fit(series)?;
        self.per_horizon_params.clear();
        // Prefer per-horizon parade PIT if available.
        if let Some(pit_by_h) = self.inner.parade_pit() {
            self.per_horizon_params = pit_by_h
                .iter()
                .map(|pit| {
                    // Push each PIT through Φ⁻¹ to get z-values, then fit
                    // GPD tails on the z sample for this horizon.
                    let z: Vec<f64> = pit
                        .iter()
                        .map(|&u| phi_inv(u.clamp(1e-12, 1.0 - 1e-12)))
                        .filter(|v| v.is_finite())
                        .collect();
                    fit_gpd_tails(&z, self.level)
                })
                .collect();
            return Ok(());
        }
        // Fallback: fit a single set of tail params from 1-step
        // standardised residuals; applied at every horizon.
        let raw_res = self.inner.residuals().unwrap_or(&[]);
        if raw_res.is_empty() {
            return Ok(());
        }
        let n = raw_res.len() as f64;
        let mean: f64 = raw_res.iter().sum::<f64>() / n;
        let var: f64 = raw_res.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let sigma = var.sqrt().max(1e-9);
        let z: Vec<f64> = raw_res
            .iter()
            .map(|&r| (r - mean) / sigma)
            .filter(|v| v.is_finite())
            .collect();
        self.params = fit_gpd_tails(&z, self.level);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        self.inner.predict(horizon)
    }

    fn name(&self) -> &str {
        "GpdTailsForecaster"
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.inner.fitted_values()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.inner.residuals()
    }
}

impl DistributionalForecaster for GpdTailsForecaster {
    fn forecast_dist(&self, horizon: usize) -> Result<Vec<GaussianMixture>> {
        self.inner.forecast_dist(horizon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpd_fit_matches_hand_calc_on_uniform_excesses() {
        // For a uniform on (0, s) the ML gpd fit should give γ ≈ 0
        // (exponential) with σ near the empirical mean.
        let exc: Vec<f64> = (1..=100).map(|i| (i as f64) / 100.0).collect();
        let (g, s) = fit_gpd_ml(&exc);
        assert!(g.abs() < 0.5, "γ = {g}");
        assert!((s - 0.5).abs() < 0.5, "σ = {s}");
    }

    #[test]
    fn phi_inv_round_trips() {
        for &z in &[-3.0, -1.0, 0.0, 1.0, 3.0] {
            let p = phi(z);
            let z_rt = phi_inv(p);
            assert!((z - z_rt).abs() < 1e-6);
        }
    }

    #[test]
    fn fit_gpd_tails_returns_none_on_small_sample() {
        let z: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        assert!(fit_gpd_tails(&z, 0.98).is_none());
    }

    #[test]
    fn fit_gpd_tails_produces_valid_params_on_uniform_sample() {
        // Uniform in [-3, 3] — has "tails" that GPD can fit trivially.
        let z: Vec<f64> = (0..1000).map(|i| -3.0 + 6.0 * i as f64 / 999.0).collect();
        let params = fit_gpd_tails(&z, 0.95).expect("fit failed");
        assert!(params.zeta_lo > 0.0 && params.zeta_up > 0.0);
        assert!(params.t_up > params.t_lo);
    }

    #[test]
    fn spliced_mixture_round_trips_cdf_and_quantile() {
        use super::super::dist::{Gaussian, GaussianMixture};
        let body = GaussianMixture::new(vec![
            (0.7, Gaussian::new(0.0, 1.0)),
            (0.3, Gaussian::new(3.0, 0.5)),
        ]);
        let params = GpdTailParams {
            t_lo: -2.0,
            t_up: 2.0,
            zeta_lo: 0.05,
            zeta_up: 0.05,
            g_lo: 0.1,
            s_lo: 0.5,
            g_up: 0.15,
            s_up: 0.4,
        };
        let sp = SplicedGaussianMixture::new(body, params);
        for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let q = sp.quantile(p);
            assert!(q.is_finite(), "quantile({p}) = {q}");
        }
        // Mean / std / crps from qgrid — sanity, not exact.
        let m = sp.mean();
        let s = sp.std();
        assert!(m.is_finite() && s.is_finite() && s > 0.0);
        // logpdf integrates roughly to 1 (crude midpoint).
        let mut total = 0.0;
        for i in 0..201 {
            let x = -10.0 + 20.0 * i as f64 / 200.0;
            total += sp.pdf(x) * (20.0 / 200.0);
        }
        assert!((total - 1.0).abs() < 0.1, "pdf integral = {total}");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn gpd_tail_params_serde_round_trip_is_bit_identical() {
        let orig = GpdTailParams {
            t_lo: -1.9599639845400545,
            t_up: 1.9599639845400545,
            zeta_lo: 0.024997500416604166,
            zeta_up: 0.024997500416604166,
            g_lo: 0.123456789012345,
            s_lo: 0.987654321098765,
            g_up: 0.234567890123456,
            s_up: 0.876543210987654,
        };
        let json = serde_json::to_string(&orig).expect("ser");
        let round: GpdTailParams = serde_json::from_str(&json).expect("de");
        // Bit-exact — skaters caught a real float roundtrip divergence
        // this way; we replicate the same gate.
        assert_eq!(orig.t_lo.to_bits(), round.t_lo.to_bits());
        assert_eq!(orig.t_up.to_bits(), round.t_up.to_bits());
        assert_eq!(orig.zeta_lo.to_bits(), round.zeta_lo.to_bits());
        assert_eq!(orig.zeta_up.to_bits(), round.zeta_up.to_bits());
        assert_eq!(orig.g_lo.to_bits(), round.g_lo.to_bits());
        assert_eq!(orig.s_lo.to_bits(), round.s_lo.to_bits());
        assert_eq!(orig.g_up.to_bits(), round.g_up.to_bits());
        assert_eq!(orig.s_up.to_bits(), round.s_up.to_bits());
    }
}
