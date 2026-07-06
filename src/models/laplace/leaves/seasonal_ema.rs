//! Seasonal-EMA leaf.
//!
//! Maintains a per-phase EMA of the level: for a series with period `p`,
//! phase `k ∈ 0..p` gets its own exponentially-weighted mean. The h-step
//! forecast at phase `(now + h) mod p` reads that phase's EMA; unseen
//! phases (cold-start of the calendar cycle) fall back to a global EMA.
//!
//! Predictive std grows as `σ · √h` where `σ²` is running residual
//! variance on the phase-matched prediction — same convention as the
//! other leaves.
//!
//! The period is supplied by the caller — no auto-detection.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

pub struct SeasonalEmaLeaf {
    period: usize,
    alpha: f64,
    phase_level: Vec<f64>,
    phase_seen: Vec<bool>,
    phase_step: usize,
    global_ema: f64,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl SeasonalEmaLeaf {
    pub fn new(period: usize, alpha: f64) -> Self {
        let period = period.max(1);
        Self {
            period,
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            phase_level: vec![0.0; period],
            phase_seen: vec![false; period],
            phase_step: 0,
            global_ema: 0.0,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
        }
    }

    pub fn period(&self) -> usize {
        self.period
    }

    fn sigma(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).sqrt().max(1e-9)
    }

    fn level_for_phase(&self, phase: usize) -> f64 {
        if self.phase_seen[phase] {
            self.phase_level[phase]
        } else {
            self.global_ema
        }
    }
}

impl Leaf for SeasonalEmaLeaf {
    fn name(&self) -> &'static str {
        "seasonal_ema"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let base = self.sigma();
        (1..=horizon)
            .map(|h| {
                let phase = (self.phase_step + h - 1) % self.period;
                Gaussian::new(self.level_for_phase(phase), base * (h as f64).sqrt())
            })
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let phase = self.phase_step;
        let predicted = self.level_for_phase(phase);
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        if self.phase_seen[phase] {
            self.phase_level[phase] = self.alpha * y + (1.0 - self.alpha) * self.phase_level[phase];
        } else {
            self.phase_level[phase] = y;
            self.phase_seen[phase] = true;
        }

        if self.n == 1 {
            self.global_ema = y;
        } else {
            self.global_ema = self.alpha * y + (1.0 - self.alpha) * self.global_ema;
        }

        self.phase_step = (self.phase_step + 1) % self.period;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_onto_pure_period_signal() {
        // Repeat [10, 20, 30] many times → phase EMAs converge to those.
        let cycle = [10.0, 20.0, 30.0];
        let mut leaf = SeasonalEmaLeaf::new(3, 0.4);
        for _ in 0..200 {
            for &y in &cycle {
                leaf.observe(y);
            }
        }
        let preds = leaf.predict(3);
        assert!(
            (preds[0].mean - 10.0).abs() < 0.5,
            "h=1 → {}",
            preds[0].mean
        );
        assert!(
            (preds[1].mean - 20.0).abs() < 0.5,
            "h=2 → {}",
            preds[1].mean
        );
        assert!(
            (preds[2].mean - 30.0).abs() < 0.5,
            "h=3 → {}",
            preds[2].mean
        );
    }

    #[test]
    fn cold_start_uses_global_ema_for_unseen_phases() {
        let mut leaf = SeasonalEmaLeaf::new(4, 0.4);
        // Only feed 2 observations — phases 2 and 3 are unseen.
        leaf.observe(5.0);
        leaf.observe(7.0);
        let preds = leaf.predict(4);
        // preds[2] and preds[3] should be finite and near the observed range,
        // not NaN, because global_ema fallback applies.
        for (i, p) in preds.iter().enumerate() {
            assert!(p.mean.is_finite(), "h={}: mean not finite", i + 1);
            assert!(p.std.is_finite() && p.std > 0.0, "h={}: std invalid", i + 1);
        }
    }

    #[test]
    fn period_one_degenerates_to_plain_ema() {
        let mut seasonal = SeasonalEmaLeaf::new(1, 0.3);
        for y in [1.0, 2.0, 3.0, 4.0, 5.0] {
            seasonal.observe(y);
        }
        // Both horizons should give the same mean (period=1 → always same phase).
        let preds = seasonal.predict(2);
        assert!((preds[0].mean - preds[1].mean).abs() < 1e-12);
    }
}
