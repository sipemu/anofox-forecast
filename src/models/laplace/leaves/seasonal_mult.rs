//! Multiplicative seasonal-EMA leaf.
//!
//! Retail seasonality is often proportional (peak week = 3× baseline,
//! not baseline + 5). The additive [`SeasonalEmaLeaf`](super::SeasonalEmaLeaf)
//! misfits this on retail. Here we track a per-phase *multiplier* on a
//! shared level:
//!
//! * `level`: a running EMA of the deseasonalised series (`y / factor[phase]`)
//! * `factor[phase]`: a running EMA of `y / level` for that phase
//! * h-step forecast at phase `(now + h - 1) mod period`: `level · factor[phase]`
//!
//! Predictive std uses `σ · √h` on the residual against the multiplicative
//! forecast — same convention as the other leaves.
//!
//! Small numerical guard: when `level` is near zero (e.g. cold start or an
//! all-zeros run) the multiplier update is skipped for that step so the
//! ratio doesn't blow up.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const LEVEL_TOL: f64 = 1e-6;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiplicativeSeasonalLeaf {
    period: usize,
    alpha: f64,
    level: f64,
    initialized_level: bool,
    factor: Vec<f64>,
    factor_seen: Vec<bool>,
    phase_step: usize,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl MultiplicativeSeasonalLeaf {
    pub fn new(period: usize, alpha: f64) -> Self {
        let period = period.max(1);
        Self {
            period,
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            level: 0.0,
            initialized_level: false,
            factor: vec![1.0; period],
            factor_seen: vec![false; period],
            phase_step: 0,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
        }
    }

    fn sigma(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).sqrt().max(1e-9)
    }

    fn factor_at(&self, phase: usize) -> f64 {
        if self.factor_seen[phase] {
            self.factor[phase]
        } else {
            1.0
        }
    }
}

impl Leaf for MultiplicativeSeasonalLeaf {
    fn name(&self) -> &'static str {
        "seasonal_mult"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let level = if self.initialized_level {
            self.level
        } else {
            0.0
        };
        let base = self.sigma();
        (1..=horizon)
            .map(|h| {
                let phase = (self.phase_step + h - 1) % self.period;
                let mean = level * self.factor_at(phase);
                Gaussian::new(mean, base * (h as f64).sqrt())
            })
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let phase = self.phase_step;
        let factor = self.factor_at(phase);
        let predicted = if self.initialized_level {
            self.level * factor
        } else {
            y
        };
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        // Update level on the deseasonalised value.
        let de_seasonalised = if factor.abs() > LEVEL_TOL {
            y / factor
        } else {
            y
        };
        if !self.initialized_level {
            self.level = de_seasonalised;
            self.initialized_level = true;
        } else {
            self.level = self.alpha * de_seasonalised + (1.0 - self.alpha) * self.level;
        }

        // Update the phase multiplier (guarded against small level).
        if self.level.abs() > LEVEL_TOL {
            let new_factor = y / self.level;
            if self.factor_seen[phase] {
                self.factor[phase] =
                    self.alpha * new_factor + (1.0 - self.alpha) * self.factor[phase];
            } else {
                self.factor[phase] = new_factor;
                self.factor_seen[phase] = true;
            }
        }

        self.phase_step = (self.phase_step + 1) % self.period;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_onto_a_multiplicative_seasonal() {
        let mut leaf = MultiplicativeSeasonalLeaf::new(3, 0.4);
        // Series with baseline 10 and multipliers [0.5, 2.0, 1.5].
        for _ in 0..200 {
            for &m in &[0.5, 2.0, 1.5] {
                leaf.observe(10.0 * m);
            }
        }
        let preds = leaf.predict(3);
        assert!(
            (preds[0].mean - 5.0).abs() < 1.0,
            "h=1 → {}, target 5.0",
            preds[0].mean
        );
        assert!(
            (preds[1].mean - 20.0).abs() < 2.0,
            "h=2 → {}, target 20.0",
            preds[1].mean
        );
        assert!(
            (preds[2].mean - 15.0).abs() < 1.5,
            "h=3 → {}, target 15.0",
            preds[2].mean
        );
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = MultiplicativeSeasonalLeaf::new(4, 0.3);
        leaf.observe(5.0);
        leaf.observe(0.0);
        let preds = leaf.predict(6);
        for (h, p) in preds.iter().enumerate() {
            assert!(p.mean.is_finite(), "h={}", h + 1);
            assert!(p.std.is_finite() && p.std > 0.0, "h={}", h + 1);
        }
    }
}
