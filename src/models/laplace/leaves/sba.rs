//! Syntetos-Boylan Approximation (SBA) leaf.
//!
//! SBA (Syntetos & Boylan, 2005) is Croston's method with a bias
//! correction. Croston's `demand_ema / interval_ema` overestimates the
//! true expected demand-per-period by ~α/2 · (demand_ema / interval_ema²)
//! (Syntetos & Boylan 2001). SBA multiplies Croston's forecast by
//! `(1 - α/2)` to remove the leading-order bias.
//!
//! On the M4 competition intermittent panels SBA has slightly better
//! MASE than plain Croston. In our ensemble it adds diversity — SBA
//! and Croston vote differently on the same series.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const ZERO_TOL: f64 = 1e-9;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SbaLeaf {
    alpha: f64,
    demand_ema: f64,
    interval_ema: f64,
    steps_since_demand: usize,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl SbaLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            demand_ema: 0.0,
            interval_ema: 1.0,
            steps_since_demand: 0,
            initialized: false,
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

    fn point(&self) -> f64 {
        if !self.initialized || self.interval_ema < ZERO_TOL {
            return 0.0;
        }
        // SBA correction: multiply Croston forecast by (1 - α/2)
        let croston = self.demand_ema / self.interval_ema;
        croston * (1.0 - self.alpha / 2.0)
    }
}

impl Leaf for SbaLeaf {
    fn name(&self) -> &'static str {
        "sba"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let point = self.point();
        let base = self.sigma();
        (1..=horizon)
            .map(|h| Gaussian::new(point, base * (h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let predicted = self.point();
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        if y > ZERO_TOL {
            let interval = (self.steps_since_demand + 1) as f64;
            if !self.initialized {
                self.demand_ema = y;
                self.interval_ema = interval;
                self.initialized = true;
            } else {
                self.demand_ema = self.alpha * y + (1.0 - self.alpha) * self.demand_ema;
                self.interval_ema = self.alpha * interval + (1.0 - self.alpha) * self.interval_ema;
            }
            self.steps_since_demand = 0;
        } else {
            self.steps_since_demand += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sba_forecast_is_croston_scaled_by_bias_correction() {
        let mut sba = SbaLeaf::new(0.1);
        // Same series as Croston test: 10, 0, 0, 0, 0, repeat.
        for _ in 0..80 {
            sba.observe(10.0);
            for _ in 0..4 {
                sba.observe(0.0);
            }
        }
        let preds = sba.predict(3);
        // Croston would give 2.0; SBA gives 2.0 * (1 - 0.1/2) = 1.9.
        for p in preds {
            assert!(
                (p.mean - 1.9).abs() < 0.3,
                "SBA expected ~1.9, got {}",
                p.mean
            );
        }
    }
}
