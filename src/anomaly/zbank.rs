//! Multi-scale prediction bank (port of the `zbank` skater).
//!
//! A bank of parade-wrapped forecasters at different
//! `(scale_alpha, stride)` gridpoints, whose concatenated z-vectors
//! form one long surprise vector. This lets the downstream Mahalanobis
//! detector see structure at multiple time-and-memory scales
//! simultaneously — a slow-memory engine spots anomalies against a
//! long-held notion of normal; a coarse-clock engine spots drifts a
//! fine-clock engine absorbs.
//!
//! ## Stride phase trick
//!
//! For stride `s`, we keep `s` phase-shifted copies of the engine.
//! Each tick advances only the copy whose clock phase matches `t mod s`,
//! so every stride contributes a fresh surprise at every tick — no
//! staleness, no detection delay — at an amortised cost of one engine
//! step per `(sigma, stride)` per tick.
//!
//! ## Cost
//!
//! For `sigmas = [σ₁, σ₂]` and `strides = [1, 4, 16]` the bank holds
//! `2 · (1 + 4 + 16) = 42` engine copies. Each is a
//! `LaplaceForecaster` + [`Parade`]. Memory is `O(engines · k²)`
//! (Parade ring buffer × k Gaussians × mixture components each).
//!
//! Consider trimming `strides` if k is large; the pass-through engine
//! (`stride = 1`) is mandatory since its forecasts are what the caller
//! sees.

use super::mahalanobis::{AnomalyOutput, MahalanobisConfig, MahalanobisScorer};
use super::parade::Parade;
use crate::core::TimeSeries;
use crate::error::Result;
use crate::models::laplace::dist::GaussianMixture;
use crate::models::laplace::LaplaceForecaster;

/// A bank of parade-wrapped forecasters, indexed by `(sigma_idx, stride)`.
/// The per-engine k horizons concatenate into one long z-vector of
/// length `sigmas.len() · strides_sum() · k`.
pub struct ZBank {
    /// Per-engine parades, keyed by (sigma_idx, stride_idx, phase).
    /// Outer Vec length: `sigmas.len() * strides.len()`. Inner length:
    /// the stride (number of phase copies).
    engines: Vec<Vec<Parade>>,
    sigmas: Vec<f64>,
    strides: Vec<usize>,
    k: usize,
    t: usize,
    /// Cache of the latest concatenated z (post `observe`). Length:
    /// `sigmas.len() * strides.len() * k`. Uninitialized entries are
    /// `None` while their engine warms up.
    last_z: Vec<Option<f64>>,
    last_dists: Option<Vec<GaussianMixture>>,
}

impl ZBank {
    /// Effective concatenated z-vector length: `sigmas · strides · k`.
    pub fn effective_k(&self) -> usize {
        self.sigmas.len() * self.strides.len() * self.k
    }

    /// Bank size: total number of active phase copies. Diagnostic.
    pub fn n_engines(&self) -> usize {
        self.engines.iter().map(|v| v.len()).sum()
    }

    /// Latest concatenated z-vector. Entries are `None` for engines
    /// still in warmup.
    pub fn z(&self) -> &[Option<f64>] {
        &self.last_z
    }

    /// Pass-through forecasts from the (first sigma, stride=1) engine —
    /// the reference against which the bank is defined.
    pub fn forecast_dist(&self, h: usize) -> Result<Vec<GaussianMixture>> {
        self.engines[0][0].forecast_dist(h)
    }

    /// Absorb one observation. For each `(sigma, stride)` pair,
    /// advances the phase copy whose clock ends now, extracts its
    /// z-vector, and appends it to the concatenated result cached in
    /// [`Self::z`].
    pub fn observe(&mut self, y: f64) -> Result<()> {
        let k = self.k;
        let n_sig = self.sigmas.len();
        let n_str = self.strides.len();
        // Reset the concatenated z (will overwrite entries that update).
        for e in self.last_z.iter_mut() {
            *e = None;
        }
        for s_idx in 0..n_sig {
            for st_idx in 0..n_str {
                let stride = self.strides[st_idx];
                let phase = self.t % stride;
                let bank_idx = s_idx * n_str + st_idx;
                let parade = &mut self.engines[bank_idx][phase];
                parade.observe(y)?;
                // Pass-through: forecasts from the (first sigma, stride=1) engine.
                if s_idx == 0 && stride == 1 {
                    if let Ok(dists) = parade.forecast_dist(k) {
                        self.last_dists = Some(dists);
                    }
                }
                // Copy this engine's z into the concatenated slot.
                let base = bank_idx * k;
                for (h, opt_z) in parade.z().iter().enumerate() {
                    self.last_z[base + h] = *opt_z;
                }
            }
        }
        self.t += 1;
        Ok(())
    }
}

/// Builder for a [`ZBank`].
pub struct ZBankBuilder {
    k: usize,
    sigmas: Vec<f64>,
    strides: Vec<usize>,
}

impl ZBankBuilder {
    /// Per-engine horizon `k`. Reference defaults: `sigmas = [0.03, 0.003]`,
    /// `strides = [1, 4, 16]`. `strides` must include `1`.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            sigmas: vec![0.03, 0.003],
            strides: vec![1, 4, 16],
        }
    }

    pub fn sigmas(mut self, sigmas: Vec<f64>) -> Self {
        self.sigmas = sigmas;
        self
    }

    pub fn strides(mut self, strides: Vec<usize>) -> Self {
        self.strides = strides;
        self
    }

    /// Build the bank by cloning `base_config` for each engine, fitting
    /// on `series`, and wrapping each in a [`Parade`]. Each
    /// `(sigma, stride)` pair receives `stride` phase copies.
    ///
    /// `base_factory` is called once per engine copy to produce a fresh
    /// unfitted `LaplaceForecaster`. Callers typically pass
    /// `|| LaplaceForecaster::new().auto()` or similar.
    pub fn build<F>(self, mut base_factory: F, series: &TimeSeries) -> Result<ZBank>
    where
        F: FnMut() -> LaplaceForecaster,
    {
        assert!(self.k >= 1);
        assert!(!self.sigmas.is_empty());
        assert!(!self.strides.is_empty());
        assert!(
            self.strides.contains(&1),
            "strides must include 1 (the pass-through engine)"
        );
        let n_sig = self.sigmas.len();
        let n_str = self.strides.len();
        let mut engines: Vec<Vec<Parade>> = Vec::with_capacity(n_sig * n_str);
        for _sig in &self.sigmas {
            for &stride in &self.strides {
                let mut phase_copies = Vec::with_capacity(stride);
                for _ph in 0..stride {
                    let base = base_factory();
                    // Fit on the same training window — the sigma
                    // parameter would ideally influence the base's
                    // internal EWMA rate, but for this port we treat
                    // sigma as a label (the caller can bake it into
                    // the factory closure if they need real
                    // differentiation between engines).
                    let parade = Parade::fit_and_wrap(base, series, self.k)?;
                    phase_copies.push(parade);
                }
                engines.push(phase_copies);
            }
        }
        Ok(ZBank {
            engines,
            k: self.k,
            last_z: vec![None; n_sig * n_str * self.k],
            last_dists: None,
            sigmas: self.sigmas,
            strides: self.strides,
            t: 0,
        })
    }
}

/// Streaming Mahalanobis detector on top of a [`ZBank`]. The scoring
/// state has `k_effective = sigmas · strides · k` — matches the
/// concatenated z-vector length.
///
/// Only ticks where **every** engine's z-vector is fully populated
/// produce a scored output; earlier ticks are warmup.
pub struct ZBankDetector {
    bank: ZBank,
    scorer: MahalanobisScorer,
    pend1: Option<GaussianMixture>,
}

impl ZBankDetector {
    /// Build from a fitted bank + config. The scorer is sized to the
    /// bank's effective k.
    pub fn wrap(bank: ZBank, cfg: MahalanobisConfig) -> Self {
        let k_eff = bank.effective_k();
        let scorer = MahalanobisScorer::with_k(k_eff, cfg);
        Self {
            bank,
            scorer,
            pend1: None,
        }
    }

    /// Absorb one observation and score it.
    pub fn observe(&mut self, y: f64) -> Result<()> {
        if !y.is_finite() {
            self.scorer.blank();
            return Ok(());
        }
        // Deep-evidence nlp from the pass-through engine's previous
        // 1-step predictive.
        let nlp = self.pend1.as_ref().map(|d| {
            let lp = d.logpdf(y);
            if lp.is_finite() {
                -lp
            } else {
                1e6
            }
        });
        self.bank.observe(y)?;
        // Refresh pend1 for next tick.
        self.pend1 = self
            .bank
            .last_dists
            .as_ref()
            .and_then(|dists| dists.first().cloned());
        // Warmup: any engine still returning None → no score.
        let z_opt = self.bank.z();
        let z: Vec<f64> = if z_opt.iter().all(|v| v.is_some()) {
            z_opt.iter().map(|v| v.unwrap()).collect()
        } else {
            self.scorer.blank();
            return Ok(());
        };
        self.scorer.score_z(&z, nlp);
        Ok(())
    }

    pub fn state(&self) -> &AnomalyOutput {
        self.scorer.last()
    }

    pub fn bank(&self) -> &ZBank {
        &self.bank
    }

    pub fn forecast_dist(&self, h: usize) -> Result<Vec<GaussianMixture>> {
        self.bank.forecast_dist(h)
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
    fn zbank_effective_k_matches_config() {
        let vals = synthetic_iid_gaussian(200);
        let train = ts_from(vals[..100].to_vec());
        // 2 sigmas × 2 strides × k=2 = 8
        let bank = ZBankBuilder::new(2)
            .sigmas(vec![0.03, 0.003])
            .strides(vec![1, 4])
            .build(|| LaplaceForecaster::new().auto(), &train)
            .unwrap();
        assert_eq!(bank.effective_k(), 8);
        assert_eq!(bank.n_engines(), 2 * (1 + 4)); // 10
    }

    #[test]
    fn zbank_detector_flags_spike() {
        let mut vals = synthetic_iid_gaussian(400);
        vals[350] = 25.0; // huge spike
        let train = ts_from(vals[..150].to_vec());
        let bank = ZBankBuilder::new(2)
            .sigmas(vec![0.03])
            .strides(vec![1, 2])
            .build(|| LaplaceForecaster::new().auto(), &train)
            .unwrap();
        let mut det = ZBankDetector::wrap(bank, MahalanobisConfig::new(2));
        let mut min_p_before = 1.0;
        for &y in &vals[150..350] {
            det.observe(y).unwrap();
            if let Some(p) = det.state().p_value {
                if p < min_p_before {
                    min_p_before = p;
                }
            }
        }
        det.observe(vals[350]).unwrap();
        let p_spike = det.state().p_value.unwrap_or(1.0);
        assert!(
            p_spike < 0.05,
            "spike p-value {p_spike} should be < 0.05 (min pre-spike {min_p_before})",
        );
    }

    #[test]
    fn zbank_strides_must_include_one() {
        let vals = synthetic_iid_gaussian(50);
        let train = ts_from(vals.clone());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ZBankBuilder::new(2)
                .strides(vec![2, 4])
                .build(|| LaplaceForecaster::new().auto(), &train)
        }));
        assert!(result.is_err(), "should panic when strides don't include 1");
    }
}
