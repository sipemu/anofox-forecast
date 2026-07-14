//! Teunter-Syntetos-Babai (TSB) leaf.
//!
//! TSB (Teunter, Syntetos & Babai, 2011) is Croston's method reworked
//! for **obsolescence-aware** intermittent forecasting. Instead of
//! tracking demand INTERVALS (which never resets — Croston can never
//! forecast zero), TSB tracks the demand **probability** each period.
//!
//! Two EMAs:
//! - `demand_size`: EMA of non-zero demand sizes (updated only when y > 0).
//! - `prob`: EMA of the binary indicator `{y > 0}` (updated **every** period).
//!
//! Forecast = `prob · demand_size`. As demand stops, `prob` decays toward
//! zero and the forecast trends to zero — this is what Croston cannot do.
//! TSB is the intermittent-forecasting method of choice for retail SKUs
//! that go obsolete (short life-cycle products, seasonal end-of-life,
//! declining categories).
//!
//! Typical rates: `alpha` (size) 0.05-0.3, `beta` (prob) usually smaller
//! 0.02-0.1 — probability drifts should be slower than size updates.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const ZERO_TOL: f64 = 1e-9;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TsbLeaf {
    alpha: f64,
    beta: f64,
    demand_size: f64,
    prob: f64,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl TsbLeaf {
    /// `alpha`: demand-size EMA rate. `beta`: demand-probability EMA rate
    /// (usually smaller than α — probability drifts slower than size).
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            beta: beta.clamp(1e-3, 1.0 - 1e-3),
            demand_size: 0.0,
            prob: 0.0,
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
        self.prob * self.demand_size
    }
}

impl Leaf for TsbLeaf {
    fn name(&self) -> &'static str {
        "tsb"
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

        // Prob updates every period (unlike Croston / SBA):
        let indicator = if y > ZERO_TOL { 1.0 } else { 0.0 };
        if !self.initialized {
            self.prob = indicator;
        } else {
            self.prob = self.beta * indicator + (1.0 - self.beta) * self.prob;
        }
        // Size only on non-zero:
        if y > ZERO_TOL {
            if !self.initialized {
                self.demand_size = y;
                self.initialized = true;
            } else {
                self.demand_size = self.alpha * y + (1.0 - self.alpha) * self.demand_size;
            }
        } else if !self.initialized && self.demand_size == 0.0 {
            // Not-yet-initialized on all-zero prefix: mark initialized so
            // prob EMA takes effect and forecast can be > 0 once a non-
            // zero arrives.
            self.initialized = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsb_forecast_trends_to_zero_on_obsolescent_series() {
        // 40 obs of demand-every-5th-period at level 10, then 60 obs of
        // pure zeros. Croston would keep forecasting; TSB's prob should
        // decay so forecast trends to zero.
        let mut tsb = TsbLeaf::new(0.1, 0.05);
        for _ in 0..40 {
            tsb.observe(10.0);
            for _ in 0..4 {
                tsb.observe(0.0);
            }
        }
        let active_forecast = tsb.predict(1)[0].mean;
        // After 60 pure zeros, prob should have decayed and forecast → 0.
        for _ in 0..60 {
            tsb.observe(0.0);
        }
        let decayed_forecast = tsb.predict(1)[0].mean;
        assert!(
            decayed_forecast < 0.5 * active_forecast,
            "TSB should decay: active {active_forecast:.3} → decayed {decayed_forecast:.3}"
        );
    }

    #[test]
    fn tsb_cold_start_produces_finite_predictions() {
        let mut leaf = TsbLeaf::new(0.1, 0.05);
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
