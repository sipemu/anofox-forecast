//! Streaming Mahalanobis anomaly detector.
//!
//! Port of [microprediction/timemachines' `mahalanobis`](
//! https://github.com/microprediction/timemachines/blob/main/src/timemachines/heads/mahalanobis.py).
//!
//! Wraps a fitted [`crate::models::laplace::LaplaceForecaster`] via a
//! [`Parade`] and emits, per tick:
//!
//! - `d2` — Mahalanobis distance of the k-vector of standardised
//!   surprises;
//! - `p_value` — calibrated tail probability (Uniform(0,1) under a
//!   well-specified forecaster);
//! - `run` — consecutive ticks above the guard level (1 → point
//!   outlier, growing → changepoint).
//!
//! ## Calibration layers
//!
//! - **Bulk** (Welch-Satterthwaite): tracks running mean `m2` and
//!   variance `v2` of `d²`; matches to scaled chi-square `c · χ²_ν`.
//! - **Tail** (POT/GPD): excesses over the `pot_level` empirical-null
//!   quantile fit a Generalized Pareto Distribution; deep-tail p-values
//!   are `p = zeta · GPD_sf(d² − t_pot)`.
//!
//! ## Robustness
//!
//! - Huberized updates: ticks with `d² > q_guard` are downweighted by
//!   `q_guard / d²`.
//! - Changepoint escape: `adapt_after` consecutive guarded ticks
//!   resume full-weight updates so a structural break doesn't stay
//!   anomalous forever.
//! - Winsorized null-moment update: variance update is quadratic in
//!   `d²`, so downweighting an outlier still lets it widen its own
//!   null through v2. Clip the update value at `q_guard` outright.
//! - Deep-evidence (nlp) channel: `-logpdf` of `y` under the 1-step
//!   predictive; own POT tail; Bonferroni-combined with the
//!   Mahalanobis p-value. Restores resolution when `|z|` saturates at
//!   the parade's ±7σ clamp.

use std::collections::VecDeque;

use super::chi2::{chi2_ppf, chi2_sf};
use super::gpd::{gpd_fit_pwm, gpd_sf};
use super::linalg::{cholesky, mahal2, solve_sym, top_factors};
use super::parade::Parade;
use crate::core::TimeSeries;
use crate::error::Result;
use crate::models::laplace::dist::GaussianMixture;
use crate::models::laplace::LaplaceForecaster;

/// Two ways to tame the near-singular scatter matrix at scoring time.
#[derive(Clone, Copy, Debug)]
pub enum ScatterMode {
    /// `Σ ≈ Σⱼ λⱼ vⱼvⱼᵀ + D`. Leading eigenpairs by power iteration,
    /// residual per-horizon variances on the diagonal, exact inverse
    /// via Woodbury. Recommended: models the "everything surprised
    /// together" degeneracy instead of flooring it.
    ///
    /// `factors`: number of leading eigenpairs (1 is enough for a
    /// single parade; 2-4 for a bank). `dfloor`: relative floor on
    /// residual diagonal variances, as a fraction of the mean diagonal.
    Factor { factors: usize, dfloor: f64 },
    /// `(1-δ)·S + δ·I` — plain Ledoit-Wolf-style shrinkage toward
    /// identity. Simpler but suppresses "the forecasts disagreed"
    /// directions.
    Shrink { delta: f64 },
}

/// Configuration for [`MahalanobisDetector`]. Defaults mirror the
/// Python reference exactly.
#[derive(Clone, Debug)]
pub struct MahalanobisConfig {
    /// Forecast horizon of the parade. Passed to the base forecaster.
    pub k: usize,
    /// EWMA rate for the location/scatter of z. Memory ~ 1/α.
    pub alpha: f64,
    /// Scatter regularization mode.
    pub scatter: ScatterMode,
    /// Chi-square quantile above which a tick's update is Huberized.
    pub guard_p: f64,
    /// Consecutive guarded ticks after which the run is treated as a
    /// changepoint and updates resume at full weight.
    pub adapt_after: usize,
    /// POT threshold quantile of the empirical null.
    pub pot_level: f64,
    /// Excesses required before the GPD tail is used (bulk fit is used
    /// below that).
    pub min_exc: usize,
}

impl MahalanobisConfig {
    /// Reference defaults from the Python source.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            alpha: 0.02,
            scatter: ScatterMode::Factor {
                factors: 1,
                dfloor: 1e-3,
            },
            guard_p: 0.99,
            adapt_after: 10,
            pot_level: 0.98,
            min_exc: 30,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    pub fn with_scatter(mut self, s: ScatterMode) -> Self {
        self.scatter = s;
        self
    }
    pub fn with_guard_p(mut self, g: f64) -> Self {
        self.guard_p = g;
        self
    }
    pub fn with_pot_level(mut self, p: f64) -> Self {
        self.pot_level = p;
        self
    }
}

/// Anomaly detector output at the current tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnomalyOutput {
    /// Mahalanobis distance of the current tick's z-vector. `None`
    /// while any horizon of the parade is still warming up.
    pub d2: Option<f64>,
    /// Calibrated tail probability under the empirical null.
    /// `None` under warmup; otherwise `∈ (0, 1)`.
    pub p_value: Option<f64>,
    /// Consecutive ticks with `d² > q_guard`. `1-2` reads as a point
    /// outlier; a growing run reads as a changepoint.
    pub run: usize,
}

/// Streaming multivariate anomaly detector wrapping a
/// [`LaplaceForecaster`] via a [`Parade`].
pub struct MahalanobisDetector {
    parade: Parade,
    cfg: MahalanobisConfig,
    // ---- streaming state ----
    /// EWMA of z-vector location, length k.
    mu: Vec<f64>,
    /// EWMA of z-vector scatter, k*k flat row-major.
    sigma: Vec<f64>,
    /// Bulk null: EWMA mean of d² (seeded at k, the exact chi²_k mean).
    m2: f64,
    /// Bulk null: EWMA variance of d² (seeded at 2k, the exact chi²_k variance).
    v2: f64,
    /// POT excesses buffer (bounded FIFO, cap 250).
    exc: VecDeque<f64>,
    /// EWMA of the exceedance indicator, seeded at 1 - pot_level.
    zeta: f64,
    /// Consecutive guarded ticks.
    run: usize,
    /// Deep-evidence (nlp) channel: previous tick's 1-step predictive.
    /// Used to compute nlp = -logpdf(y). `None` before first predict.
    pend1: Option<GaussianMixture>,
    /// nlp channel: EWMA mean of nlp.
    nm: f64,
    /// nlp channel: EWMA variance of nlp.
    nv: f64,
    /// nlp channel: excesses.
    n_exc: VecDeque<f64>,
    /// nlp channel: exceedance rate.
    n_zeta: f64,
    /// nlp channel: counter (min_exc gate).
    n_n: usize,
    /// Latest output.
    last: AnomalyOutput,
    /// Count of non-finite ticks (diagnostic).
    skipped: usize,
}

const EXC_CAP: usize = 250;
const NLP_Z_THRESH: f64 = 2.33; // ~ N(0,1) 0.99 quantile
const D2_EXCESS_CAP: f64 = 50.0;
const NLP_EXCESS_CAP: f64 = 50.0;
const NLP_WINSOR_SIGMAS: f64 = 6.0;

impl MahalanobisDetector {
    /// Fit the base forecaster on `series`, then wrap it. `cfg.k` is
    /// the parade horizon.
    pub fn fit_and_wrap(
        base: LaplaceForecaster,
        series: &TimeSeries,
        cfg: MahalanobisConfig,
    ) -> Result<Self> {
        assert!(cfg.k >= 1);
        assert!(cfg.alpha > 0.0 && cfg.alpha < 1.0);
        assert!(cfg.guard_p > 0.0 && cfg.guard_p < 1.0);
        assert!(cfg.pot_level > 0.0 && cfg.pot_level < 1.0);
        assert!(
            cfg.min_exc >= 2,
            "min_exc < 2 divides by zero in the GPD fit"
        );
        let parade = Parade::fit_and_wrap(base, series, cfg.k)?;
        Ok(Self::from_parade(parade, cfg))
    }

    /// Wrap a pre-built `Parade` — useful when the parade needs custom
    /// setup (e.g. warm-started from a saved state).
    pub fn from_parade(parade: Parade, cfg: MahalanobisConfig) -> Self {
        let k = cfg.k;
        // Identity prior scatter — the parade standardises each margin
        // so I is not arbitrary but the calibrated prior; shrink toward
        // it is a principled Ledoit-Wolf-style target.
        let mut sigma = vec![0.0f64; k * k];
        for i in 0..k {
            sigma[i * k + i] = 1.0;
        }
        Self {
            parade,
            m2: k as f64,
            v2: 2.0 * k as f64,
            zeta: 1.0 - cfg.pot_level,
            n_zeta: 1.0 - cfg.pot_level,
            cfg,
            mu: vec![0.0; k],
            sigma,
            exc: VecDeque::with_capacity(EXC_CAP),
            run: 0,
            pend1: None,
            nm: 0.0,
            nv: 1.0,
            n_exc: VecDeque::with_capacity(EXC_CAP),
            n_n: 0,
            last: AnomalyOutput::default(),
            skipped: 0,
        }
    }

    /// Absorb one observation and score it. `state()` yields the
    /// updated `AnomalyOutput` afterward.
    pub fn observe(&mut self, y: f64) -> Result<()> {
        // Non-finite gate: hold state, no score.
        if !y.is_finite() {
            self.skipped += 1;
            self.last.d2 = None;
            self.last.p_value = None;
            return Ok(());
        }

        // Deep-evidence: compute nlp of `y` under the 1-step predictive
        // BEFORE the parade absorbs `y` (mustn't defend itself).
        let nlp = self.pend1.as_ref().map(|d| {
            let lp = d.logpdf(y);
            if lp.is_finite() {
                -lp
            } else {
                1e6
            }
        });

        // Snapshot pend1 for next tick — done here so parade.observe
        // below produces the new one.
        // Advance the parade: it computes PIT/z and refreshes its
        // internal 1-step predictive.
        self.parade.observe(y)?;
        // Refresh pend1 for the NEXT tick.
        self.pend1 = self.parade.pending_one_step().cloned();

        // If any horizon's z is unavailable, warmup — no score.
        let z_opt = self.parade.z();
        let z: Vec<f64> = if z_opt.iter().all(|v| v.is_some()) {
            z_opt.iter().map(|v| v.unwrap()).collect()
        } else {
            self.last.d2 = None;
            self.last.p_value = None;
            return Ok(());
        };

        // === Score BEFORE any update. ===
        let k = self.cfg.k;
        let v: Vec<f64> = (0..k).map(|i| z[i] - self.mu[i]).collect();
        let d2 = self.compute_d2(&v);
        self.last.d2 = Some(d2);

        // Bulk null (Welch-Satterthwaite).
        let m2 = self.m2.max(1e-9);
        let v2 = self.v2.max(1e-9);
        let c = (v2 / (2.0 * m2)).max(1e-9);
        let nu = (2.0 * m2 * m2 / v2).clamp(0.5, 1000.0);
        let t_pot = c * chi2_ppf(self.cfg.pot_level, nu);
        let t_scale = t_pot.max(1e-9);

        // Tail p-value: bulk unless GPD is authoritative.
        let mut p_value = if d2 > t_pot && self.exc.len() >= self.cfg.min_exc {
            let exc_v: Vec<f64> = self.exc.iter().copied().collect();
            let (gamma, sigma_g) = gpd_fit_pwm(&exc_v);
            (self.zeta.max(1e-12) * gpd_sf((d2 - t_pot) / t_scale, gamma, sigma_g)).min(1.0)
        } else {
            chi2_sf(d2 / c, nu)
        };

        // Deep-evidence (nlp) channel — Bonferroni-combined.
        if let Some(nlp_val) = nlp {
            if self.n_n >= self.cfg.min_exc {
                let ns = self.nv.max(1e-12).sqrt();
                let t_n = self.nm + NLP_Z_THRESH * ns;
                if nlp_val > t_n && self.n_exc.len() >= self.cfg.min_exc {
                    let n_exc_v: Vec<f64> = self.n_exc.iter().copied().collect();
                    let (g2, s2) = gpd_fit_pwm(&n_exc_v);
                    let denom = (t_n - self.nm).max(1e-9);
                    let p_n = self.n_zeta.max(1e-12) * gpd_sf((nlp_val - t_n) / denom, g2, s2);
                    p_value = p_value.min(2.0 * p_n);
                }
            }
        }
        self.last.p_value = Some(p_value.clamp(1e-300, 1.0));

        // === Huberized update rate. ===
        let q_guard = c * chi2_ppf(self.cfg.guard_p, nu);
        let w = if d2 > q_guard {
            self.run += 1;
            if self.run > self.cfg.adapt_after {
                1.0
            } else {
                q_guard / d2
            }
        } else {
            self.run = 0;
            1.0
        };
        self.last.run = self.run;
        let a = self.cfg.alpha * w;

        // Null moments: WINSORIZE (not just downweight) so an outlier's
        // v2 update can't widen its own null.
        let d2n = if w == 1.0 { d2 } else { d2.min(q_guard) };
        let dm = d2n - self.m2;
        self.m2 += self.cfg.alpha * dm;
        self.v2 = (1.0 - self.cfg.alpha) * self.v2 + self.cfg.alpha * dm * (d2n - self.m2);

        // POT layer maintenance.
        let aw = self.cfg.alpha * w;
        let exceed = if d2 > t_pot { 1.0 } else { 0.0 };
        self.zeta = (1.0 - aw) * self.zeta + aw * exceed;
        if d2 > t_pot {
            let e = ((d2 - t_pot) / t_scale).min(D2_EXCESS_CAP);
            self.exc.push_back(e);
            if self.exc.len() > EXC_CAP {
                self.exc.pop_front();
            }
        }

        // nlp channel maintenance.
        if let Some(nlp_val) = nlp {
            self.n_n += 1;
            let ns = self.nv.max(1e-12).sqrt();
            let t_n = self.nm + NLP_Z_THRESH * ns;
            // Winsorized moments: cap at NLP_WINSOR_SIGMAS σ above mean.
            let nw = nlp_val.min(self.nm + NLP_WINSOR_SIGMAS * ns);
            let dn = nw - self.nm;
            self.nm += self.cfg.alpha * dn;
            self.nv = (1.0 - self.cfg.alpha) * self.nv + self.cfg.alpha * dn * (nw - self.nm);
            let n_exceed = if nlp_val > t_n { 1.0 } else { 0.0 };
            self.n_zeta = (1.0 - aw) * self.n_zeta + aw * n_exceed;
            if nlp_val > t_n {
                let denom = (t_n - self.nm).max(1e-9);
                let e = ((nlp_val - t_n) / denom).min(NLP_EXCESS_CAP);
                self.n_exc.push_back(e);
                if self.n_exc.len() > EXC_CAP {
                    self.n_exc.pop_front();
                }
            }
        }

        // Location / scatter update: delta pre- and post-update.
        let delta_pre: Vec<f64> = (0..k).map(|i| z[i] - self.mu[i]).collect();
        for i in 0..k {
            self.mu[i] += a * delta_pre[i];
        }
        let delta_post: Vec<f64> = (0..k).map(|i| z[i] - self.mu[i]).collect();
        for i in 0..k {
            for j in 0..k {
                self.sigma[i * k + j] =
                    (1.0 - a) * self.sigma[i * k + j] + a * delta_pre[i] * delta_post[j];
            }
        }

        Ok(())
    }

    /// Latest scoring output.
    pub fn state(&self) -> &AnomalyOutput {
        &self.last
    }

    /// Pass-through: latest k-vector forecast from the base forecaster.
    pub fn forecast_dist(&self, h: usize) -> Result<Vec<GaussianMixture>> {
        self.parade.forecast_dist(h)
    }

    /// Immutable access to the wrapped parade.
    pub fn parade(&self) -> &Parade {
        &self.parade
    }

    fn compute_d2(&self, v: &[f64]) -> f64 {
        let k = self.cfg.k;
        match self.cfg.scatter {
            ScatterMode::Factor { factors, dfloor } => {
                // Sigma ≈ Σⱼ λⱼ vⱼvⱼᵀ + D. Woodbury inverse.
                let fac = top_factors(&self.sigma, k, factors);
                let mean_diag: f64 = (0..k).map(|i| self.sigma[i * k + i]).sum::<f64>() / k as f64;
                let floor = (dfloor * mean_diag).max(1e-12);
                // D[i] = S[i,i] - Σⱼ λⱼ · vⱼ[i]²  (floored).
                let d: Vec<f64> = (0..k)
                    .map(|i| {
                        let diag = self.sigma[i * k + i];
                        let sub: f64 = fac.iter().map(|(lam, w)| lam * w[i] * w[i]).sum();
                        (diag - sub).max(floor)
                    })
                    .collect();
                let q1: f64 = (0..k).map(|i| v[i] * v[i] / d[i]).sum();
                if fac.is_empty() {
                    return q1;
                }
                let r = fac.len();
                // b_j = wⱼ' D⁻¹ v
                let b: Vec<f64> = fac
                    .iter()
                    .map(|(_, w)| (0..k).map(|i| w[i] * v[i] / d[i]).sum::<f64>())
                    .collect();
                // B = diag(1/λ) + W' D⁻¹ W  (r×r).
                let mut big_b = vec![0.0f64; r * r];
                for a_ in 0..r {
                    big_b[a_ * r + a_] = 1.0 / fac[a_].0;
                    for c_ in a_..r {
                        let g: f64 = (0..k).map(|i| fac[a_].1[i] * fac[c_].1[i] / d[i]).sum();
                        big_b[a_ * r + c_] += g;
                        if c_ != a_ {
                            big_b[c_ * r + a_] += g;
                        }
                    }
                }
                let x = solve_sym(&big_b, &b, r);
                q1 - (0..r).map(|j| b[j] * x[j]).sum::<f64>()
            }
            ScatterMode::Shrink { delta } => {
                let ssh: Vec<f64> = (0..k)
                    .flat_map(|i| {
                        (0..k).map(move |j| {
                            (1.0 - delta) * self.sigma[i * k + j] + if i == j { delta } else { 0.0 }
                        })
                    })
                    .collect();
                let l = cholesky(&ssh, k, 1e-12);
                mahal2(&l, v, k)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use chrono::{Duration, TimeZone, Utc};

    fn synthetic_iid_gaussian(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let seed = (i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let u1 = ((seed >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
                let seed2 = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u2 = ((seed2 >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect()
    }

    fn ts_from(vals: Vec<f64>) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
        let stamps: Vec<_> = (0..vals.len())
            .map(|i| base + Duration::hours(i as i64))
            .collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn detector_warmup_returns_none() {
        let vals = synthetic_iid_gaussian(200);
        let train = ts_from(vals[..150].to_vec());
        let base = LaplaceForecaster::new().auto();
        let cfg = MahalanobisConfig::new(4);
        let mut det = MahalanobisDetector::fit_and_wrap(base, &train, cfg).unwrap();
        // First 3 ticks: at least one horizon still None → no d2.
        for i in 0..3 {
            det.observe(vals[150 + i]).unwrap();
            assert!(det.state().d2.is_none(), "tick {i} should be warmup");
        }
        // 4th tick: all 4 horizons matured → d2 present.
        det.observe(vals[153]).unwrap();
        assert!(det.state().d2.is_some());
        assert!(det.state().p_value.is_some());
    }

    #[test]
    fn detector_flags_injected_spike() {
        let mut vals = synthetic_iid_gaussian(500);
        // Warm the detector for ~200 ticks past parade warmup, then
        // inject a 20σ spike.
        let spike_idx = 400;
        vals[spike_idx] = 20.0;
        let train = ts_from(vals[..200].to_vec());
        let base = LaplaceForecaster::new().auto();
        let cfg = MahalanobisConfig::new(4);
        let mut det = MahalanobisDetector::fit_and_wrap(base, &train, cfg).unwrap();
        let mut min_p_before = 1.0;
        for i in 200..spike_idx {
            det.observe(vals[i]).unwrap();
            if let Some(p) = det.state().p_value {
                if p < min_p_before {
                    min_p_before = p;
                }
            }
        }
        det.observe(vals[spike_idx]).unwrap();
        let p_spike = det.state().p_value.unwrap();
        assert!(
            p_spike < 0.01,
            "spike p-value {p_spike} should be < 0.01 (min pre-spike was {min_p_before})",
        );
        assert!(det.state().run >= 1);
    }

    #[test]
    fn detector_p_value_uniform_ish_under_null() {
        // Feed a long stream of iid Gaussian noise; p-values should
        // roughly Uniform(0,1). Loose check: at least 30% of ticks in
        // (0.1, 0.9), no more than 20% below 0.01 (false-alarm rate).
        let vals = synthetic_iid_gaussian(3000);
        let train = ts_from(vals[..500].to_vec());
        let base = LaplaceForecaster::new().auto();
        let cfg = MahalanobisConfig::new(4);
        let mut det = MahalanobisDetector::fit_and_wrap(base, &train, cfg).unwrap();
        let mut ps = Vec::new();
        for i in 500..vals.len() {
            det.observe(vals[i]).unwrap();
            if let Some(p) = det.state().p_value {
                ps.push(p);
            }
        }
        assert!(
            ps.len() > 2000,
            "expected > 2000 p-values, got {}",
            ps.len()
        );
        let mid: usize = ps.iter().filter(|p| **p > 0.1 && **p < 0.9).count();
        let low: usize = ps.iter().filter(|p| **p < 0.01).count();
        let mid_frac = mid as f64 / ps.len() as f64;
        let low_frac = low as f64 / ps.len() as f64;
        assert!(
            mid_frac > 0.3,
            "only {mid_frac:.2} of p-values in (0.1, 0.9)"
        );
        assert!(
            low_frac < 0.25,
            "false-alarm rate {low_frac:.3} is too high",
        );
    }
}
