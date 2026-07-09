//! Seasonal-Croston leaf — intermittent demand with a per-phase modifier.
//!
//! Classic Croston ([`IntermittentLeaf`](super::IntermittentLeaf)) predicts
//! a constant demand-per-period `demand_ema / interval_ema` at every
//! horizon. On retail SKU data the real pattern is usually `weekend
//! spike, weekday quiet, weekend spike…` — a flat constant misses badly
//! on both peaks and troughs.
//!
//! This leaf tracks a **per-phase demand-size EMA** on top of the shared
//! interval-EMA. h-step forecast at phase `p`:
//!
//! ```text
//!   ŷ_{t+h} = demand_ema[p] * (n_nonzero_at_p / n_obs_at_p)
//! ```
//!
//! The `demand_ema[p]` captures typical size when demand *does* land on
//! that phase; the empirical rate captures how often it does.
//!
//! Predictive std: `σ · √h` on residuals, same convention as other leaves.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const ZERO_TOL: f64 = 1e-9;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeasonalIntermittentLeaf {
    period: usize,
    alpha: f64,
    /// Per-phase EMA of the non-zero demand size at that phase.
    demand_ema: Vec<f64>,
    /// Per-phase count of observations we've seen at that phase.
    phase_obs: Vec<usize>,
    /// Per-phase count of *non-zero* observations at that phase.
    phase_nonzero: Vec<usize>,
    /// Which phases have received at least one non-zero observation.
    phase_seen: Vec<bool>,
    /// Global (cross-phase) demand EMA — fallback for unseen phases.
    global_demand_ema: f64,
    /// True after the first non-zero observation.
    global_initialized: bool,
    phase_step: usize,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl SeasonalIntermittentLeaf {
    pub fn new(period: usize, alpha: f64) -> Self {
        let period = period.max(1);
        Self {
            period,
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            demand_ema: vec![0.0; period],
            phase_obs: vec![0; period],
            phase_nonzero: vec![0; period],
            phase_seen: vec![false; period],
            global_demand_ema: 0.0,
            global_initialized: false,
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

    fn point_at(&self, phase: usize) -> f64 {
        let obs = self.phase_obs[phase];
        if obs == 0 {
            return if self.global_initialized {
                self.global_demand_ema * 0.1 // Very conservative fallback for unseen phase.
            } else {
                0.0
            };
        }
        let rate = self.phase_nonzero[phase] as f64 / obs as f64;
        let size = if self.phase_seen[phase] {
            self.demand_ema[phase]
        } else {
            self.global_demand_ema
        };
        size * rate
    }
}

impl Leaf for SeasonalIntermittentLeaf {
    fn name(&self) -> &'static str {
        "seasonal_intermittent"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let base = self.sigma();
        (1..=horizon)
            .map(|h| {
                let phase = (self.phase_step + h - 1) % self.period;
                Gaussian::new(self.point_at(phase), base * (h as f64).sqrt())
            })
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let phase = self.phase_step;
        let predicted = self.point_at(phase);
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        self.phase_obs[phase] += 1;
        if y > ZERO_TOL {
            self.phase_nonzero[phase] += 1;
            if self.phase_seen[phase] {
                self.demand_ema[phase] =
                    self.alpha * y + (1.0 - self.alpha) * self.demand_ema[phase];
            } else {
                self.demand_ema[phase] = y;
                self.phase_seen[phase] = true;
            }
            if self.global_initialized {
                self.global_demand_ema =
                    self.alpha * y + (1.0 - self.alpha) * self.global_demand_ema;
            } else {
                self.global_demand_ema = y;
                self.global_initialized = true;
            }
        }
        self.phase_step = (self.phase_step + 1) % self.period;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_per_phase_demand_size_times_rate() {
        // Period 7. Phases 0,1: always 5.0. Phases 2..6: always 0.
        // Expected: phase 0 forecast = 5.0 * (1/1) = 5.0.
        //           phase 2 forecast = 0.
        let mut leaf = SeasonalIntermittentLeaf::new(7, 0.1);
        for _ in 0..100 {
            for phase in 0..7 {
                let y = if phase < 2 { 5.0 } else { 0.0 };
                leaf.observe(y);
            }
        }
        let preds = leaf.predict(7);
        assert!(
            (preds[0].mean - 5.0).abs() < 0.5,
            "h=1 phase 0 → {}, want ~5",
            preds[0].mean
        );
        assert!(
            (preds[1].mean - 5.0).abs() < 0.5,
            "h=2 phase 1 → {}, want ~5",
            preds[1].mean
        );
        assert!(
            preds[2].mean.abs() < 0.5,
            "h=3 phase 2 → {}, want ~0",
            preds[2].mean
        );
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = SeasonalIntermittentLeaf::new(7, 0.1);
        leaf.observe(0.0);
        leaf.observe(5.0);
        leaf.observe(0.0);
        let preds = leaf.predict(10);
        for p in preds {
            assert!(p.mean.is_finite());
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
