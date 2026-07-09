//! The prediction parade — online calibration diagnostics on any
//! streaming distributional forecaster.
//!
//! Port of [microprediction/timemachines' `parade`](
//! https://github.com/microprediction/timemachines/blob/main/src/timemachines/parade.py).
//!
//! Each incoming observation `y` is resolved against the predictives
//! previously made *for it* — the m-step-ahead predictive issued m
//! ticks ago, for `m = 1..=k`. After [`Parade::observe`]:
//!
//! - [`Parade::pit`] holds the probability-integral-transform value
//!   per horizon — Uniform(0,1) when the corresponding predictive is
//!   calibrated;
//! - [`Parade::z`] holds the standard-normal quantile of the PIT —
//!   N(0,1) when calibrated, so `|z|` reads directly as "how
//!   surprising was this point under the m-step-ahead predictive".
//!
//! Entries are `None` until the corresponding prediction has matured
//! (the first `m` observations for horizon `m`) and on any tick where
//! a horizon's `z` is unavailable (degenerate cdf, non-finite result).
//!
//! The parade is **pass-through** for the forecasts themselves —
//! [`Parade::forecast_dist`] returns the base's output unchanged.

use std::collections::VecDeque;

use super::quantile::standard_normal_quantile;
use crate::core::TimeSeries;
use crate::error::Result;
use crate::models::laplace::dist::GaussianMixture;
use crate::models::laplace::LaplaceForecaster;
use crate::models::{DistributionalForecaster, Forecaster};

/// Clamp the PIT away from `{0, 1}`: `|z|` is then bounded by the
/// standard-normal quantile at 1e-12, about 7.03. No input can produce
/// an infinite `z`.
const PIT_EPS: f64 = 1e-12;

/// Streaming parade wrapper around a [`LaplaceForecaster`].
pub struct Parade {
    /// Base distributional forecaster. Must have been `.fit()` before
    /// the parade is wrapped around it (so it can emit predictives
    /// immediately).
    base: LaplaceForecaster,
    /// Forecast horizon. Every tick emits (and the ring buffer stores)
    /// exactly this many predictives.
    k: usize,
    /// Ring buffer of the last `k` predicted k-vectors. Newest at the
    /// back. Element at index `i` is the k-vector issued `n − i` ticks
    /// ago, where `n = pending.len()`.
    pending: VecDeque<Vec<GaussianMixture>>,
    /// Latest PIT per horizon (index 0 = 1-step-ahead).
    pit: Vec<Option<f64>>,
    /// Latest standard-normal z per horizon (index 0 = 1-step-ahead).
    z: Vec<Option<f64>>,
    /// Latest forecast returned to the caller (for the pass-through
    /// contract).
    last_dists: Option<Vec<GaussianMixture>>,
}

impl Parade {
    /// Wrap a **fitted** `LaplaceForecaster`. `k` must match the horizon
    /// the base can predict; typical range 1..32.
    pub fn wrap(base: LaplaceForecaster, k: usize) -> Result<Self> {
        assert!(k >= 1, "parade requires k >= 1");
        let mut parade = Self {
            base,
            k,
            pending: VecDeque::with_capacity(k),
            pit: vec![None; k],
            z: vec![None; k],
            last_dists: None,
        };
        // Emit the initial k-vector so subsequent `observe` has something
        // to PIT against on tick 1.
        if let Ok(dists) = parade.base.forecast_dist(k) {
            if dists.len() == k {
                parade.pending.push_back(dists.clone());
                parade.last_dists = Some(dists);
            }
        }
        Ok(parade)
    }

    /// Convenience: fit and wrap in one call.
    pub fn fit_and_wrap(
        mut base: LaplaceForecaster,
        series: &TimeSeries,
        k: usize,
    ) -> Result<Self> {
        base.fit(series)?;
        Self::wrap(base, k)
    }

    /// Absorb one new observation. Updates PIT/z from matured
    /// predictions, feeds `y` to the base forecaster, pushes the new
    /// k-vector into the ring buffer.
    ///
    /// Non-finite `y` is skipped: the previous forecasts stay put,
    /// PIT/z blank for this tick.
    pub fn observe(&mut self, y: f64) -> Result<()> {
        if !y.is_finite() {
            self.pit = vec![None; self.k];
            self.z = vec![None; self.k];
            return Ok(());
        }
        let n = self.pending.len();
        let mut pit = vec![None; self.k];
        let mut z = vec![None; self.k];
        for m in 1..=self.k {
            if m > n {
                break;
            }
            // Prediction issued `m` ticks ago, horizon `m`.
            let d = &self.pending[n - m][m - 1];
            let u = d.cdf(y);
            if !u.is_finite() {
                continue;
            }
            let u = u.clamp(PIT_EPS, 1.0 - PIT_EPS);
            pit[m - 1] = Some(u);
            z[m - 1] = Some(standard_normal_quantile(u));
        }
        self.pit = pit;
        self.z = z;

        // Magnitude-relative winsorization before the base sees `y`.
        // See the reference for the rationale: after a degenerate-
        // variance stretch a legitimate value can sit billions of
        // sigmas out and must pass; twelve orders above the current
        // level is unreachable by data but far below the ~1e77 jump
        // ratio where doubles die.
        let mut y_fed = y;
        if n > 0 {
            let d1 = &self.pending[n - 1][0];
            // Access mixture mean and total std via cheap accessors.
            let (mp, sp) = mixture_moments(d1);
            if mp.is_finite() && sp.is_finite() {
                let w = 1e12 * (1.0 + mp.abs() + sp);
                y_fed = y_fed.clamp(mp - w, mp + w);
            }
        }

        // Feed the base forecaster, get the new k-vector, push into ring.
        self.base.observe(y_fed)?;
        let dists = self.base.forecast_dist(self.k)?;
        if dists.len() == self.k {
            self.pending.push_back(dists.clone());
            if self.pending.len() > self.k {
                self.pending.pop_front();
            }
            self.last_dists = Some(dists);
        }
        Ok(())
    }

    /// Standard-normal surprises for each matured horizon.
    ///
    /// Index `m − 1` is the surprise of the current observation under
    /// the m-step-ahead predictive issued m ticks ago. `None` while
    /// that horizon is still warming up.
    pub fn z(&self) -> &[Option<f64>] {
        &self.z
    }

    /// Probability-integral-transform values for each matured horizon
    /// (Uniform(0,1) under calibration).
    pub fn pit(&self) -> &[Option<f64>] {
        &self.pit
    }

    /// Forecast horizon `k` this parade was built for.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Pass-through: latest k-vector predictive from the base
    /// forecaster (identical to `base.forecast_dist(k)?`).
    pub fn forecast_dist(&self, h: usize) -> Result<Vec<GaussianMixture>> {
        self.base.forecast_dist(h)
    }

    /// Immutable access to the wrapped base forecaster.
    pub fn base(&self) -> &LaplaceForecaster {
        &self.base
    }

    /// The 1-step-ahead predictive issued at the previous tick — the
    /// input to the deep-evidence (nlp) channel of the Mahalanobis
    /// detector. `None` before the parade has emitted any prediction.
    pub fn pending_one_step(&self) -> Option<&GaussianMixture> {
        self.pending.back().and_then(|k_vec| k_vec.first())
    }
}

/// Cheap mixture moments: mean = Σ w_i · μ_i, std = √(Σ w_i · (σ_i² + μ_i²) − mean²).
fn mixture_moments(m: &GaussianMixture) -> (f64, f64) {
    let comps = &m.components;
    if comps.is_empty() {
        return (0.0, 0.0);
    }
    let mut mean = 0.0;
    let mut second = 0.0;
    for (w, g) in comps.iter() {
        mean += w * g.mean;
        second += w * (g.std * g.std + g.mean * g.mean);
    }
    let var = (second - mean * mean).max(0.0);
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use chrono::{Duration, TimeZone, Utc};

    fn synthetic_iid_gaussian(n: usize) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
        let vals: Vec<f64> = (0..n)
            .map(|i| {
                // Deterministic Box-Muller from a simple LCG so tests
                // are reproducible without external RNG deps.
                let seed = (i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let u1 = ((seed >> 33) as f64 / (1u64 << 31) as f64)
                    .max(1e-12)
                    .min(1.0 - 1e-12);
                let seed2 = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u2 = ((seed2 >> 33) as f64 / (1u64 << 31) as f64)
                    .max(1e-12)
                    .min(1.0 - 1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn warmup_returns_none_before_maturity() {
        let n = 200;
        let ts = synthetic_iid_gaussian(n);
        let train_len = 150;
        let train = TimeSeries::univariate(
            ts.timestamps()[..train_len].to_vec(),
            ts.primary_values()[..train_len].to_vec(),
        )
        .unwrap();
        let base = LaplaceForecaster::new().auto();
        let mut parade = Parade::fit_and_wrap(base, &train, 4).unwrap();
        // Tick 1: 1-step should have z (predictive from initial forecast),
        // but 2/3/4-step still None.
        parade.observe(ts.primary_values()[train_len]).unwrap();
        assert!(parade.z()[0].is_some());
        assert!(parade.z()[1].is_none());
        assert!(parade.z()[3].is_none());
        // Tick 4: all four should be present.
        for i in 1..4 {
            parade.observe(ts.primary_values()[train_len + i]).unwrap();
        }
        for (i, z) in parade.z().iter().enumerate() {
            assert!(z.is_some(), "horizon {} still None after 4 ticks", i + 1);
        }
    }

    #[test]
    fn z_is_finite_and_bounded() {
        let n = 400;
        let ts = synthetic_iid_gaussian(n);
        let train_len = 200;
        let train = TimeSeries::univariate(
            ts.timestamps()[..train_len].to_vec(),
            ts.primary_values()[..train_len].to_vec(),
        )
        .unwrap();
        let base = LaplaceForecaster::new().auto();
        let mut parade = Parade::fit_and_wrap(base, &train, 4).unwrap();
        for i in 0..100 {
            parade.observe(ts.primary_values()[train_len + i]).unwrap();
        }
        for z in parade.z() {
            let z = z.unwrap();
            assert!(z.is_finite(), "z = {z}");
            // PIT clamped at 1e-12 → |z| ≤ 7.04.
            assert!(z.abs() <= 7.5, "|z| = {} exceeds parade clamp", z.abs());
        }
    }

    #[test]
    fn nan_observation_blanks_z_but_survives() {
        let n = 100;
        let ts = synthetic_iid_gaussian(n);
        let base = LaplaceForecaster::new().auto();
        let mut parade = Parade::fit_and_wrap(base, &ts, 4).unwrap();
        parade.observe(ts.primary_values()[0]).unwrap();
        parade.observe(f64::NAN).unwrap();
        for z in parade.z() {
            assert!(z.is_none());
        }
        // Next real observation should still work.
        parade.observe(ts.primary_values()[1]).unwrap();
        assert!(parade.z()[0].is_some());
    }

    #[test]
    fn forecast_dist_pass_through() {
        let n = 100;
        let ts = synthetic_iid_gaussian(n);
        let base = LaplaceForecaster::new().auto();
        let parade = Parade::fit_and_wrap(base, &ts, 4).unwrap();
        let dists = parade.forecast_dist(4).unwrap();
        assert_eq!(dists.len(), 4);
        for d in &dists {
            let (mean, std) = mixture_moments(d);
            assert!(mean.is_finite() && std.is_finite() && std > 0.0);
        }
    }
}
