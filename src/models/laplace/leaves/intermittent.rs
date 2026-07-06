//! Intermittent-demand leaf (Croston-flavored).
//!
//! Standard Croston (1972) tracks two EMAs separately: the non-zero
//! **demand size** and the **inter-demand interval**. The point forecast
//! is `demand_ema / interval_ema` — the expected demand per period. This
//! is dramatically better than a level-EMA on zero-inflated series (SKU
//! sales with 40-70% zero days) because the level-EMA gets dragged
//! toward zero by the zero periods and under-predicts on the non-zero
//! ones.
//!
//! Predictive std is fit from residuals on the running forecast (same
//! convention as the other leaves). h-step forecast is constant across
//! horizons — Croston's demand-per-period is inherently unbiased for
//! h-step aggregates but not for the per-h shape.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const ZERO_TOL: f64 = 1e-9;

pub struct IntermittentLeaf {
    alpha: f64,
    demand_ema: f64,
    interval_ema: f64,
    steps_since_demand: usize,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl IntermittentLeaf {
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
            0.0
        } else {
            self.demand_ema / self.interval_ema
        }
    }
}

impl Leaf for IntermittentLeaf {
    fn name(&self) -> &'static str {
        "intermittent"
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
    fn intermittent_point_matches_demand_per_interval() {
        // Series: 1 non-zero of size 10, then 4 zeros, repeat.
        // Expected demand-per-period = 10/5 = 2.0.
        let mut leaf = IntermittentLeaf::new(0.1);
        for _ in 0..80 {
            leaf.observe(10.0);
            for _ in 0..4 {
                leaf.observe(0.0);
            }
        }
        let preds = leaf.predict(3);
        for p in preds {
            assert!((p.mean - 2.0).abs() < 0.5, "expected ~2.0, got {}", p.mean);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = IntermittentLeaf::new(0.1);
        leaf.observe(0.0);
        leaf.observe(0.0);
        leaf.observe(5.0);
        let preds = leaf.predict(4);
        for p in preds {
            assert!(p.mean.is_finite());
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
