//! Damped-Holt leaf — level + trend + damping.
//!
//! Standard damped-trend recursion:
//!
//!   ℓ_t = α·y_t + (1−α)·(ℓ_{t−1} + φ·b_{t−1})
//!   b_t = β·(ℓ_t − ℓ_{t−1}) + (1−β)·φ·b_{t−1}
//!   ŷ_{t+h} = ℓ_t + (φ + φ² + … + φ^h) · b_t
//!
//! With φ = 1 this collapses to Holt's linear method (pure Holt). With
//! φ < 1 the trend decays — appropriate for series where a long-run
//! extrapolation of the current slope would overshoot.
//!
//! Predictive std grows as `σ · √h` where `σ²` is running residual
//! variance on the 1-step damped forecast — same convention as the
//! other leaves.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

pub struct HoltLeaf {
    alpha: f64,
    beta: f64,
    phi: f64,
    level: Option<f64>,
    trend: f64,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl HoltLeaf {
    /// `alpha` = level smoothing, `beta` = trend smoothing, `phi` = damping factor.
    /// `phi = 1.0` gives pure Holt; `phi ∈ (0.5, 1.0)` gives damped-trend.
    pub fn new(alpha: f64, beta: f64, phi: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            beta: beta.clamp(1e-3, 1.0 - 1e-3),
            phi: phi.clamp(0.5, 1.0),
            level: None,
            trend: 0.0,
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

    /// Geometric sum φ + φ² + … + φ^h.
    fn damped_sum(&self, h: usize) -> f64 {
        if (self.phi - 1.0).abs() < 1e-12 {
            h as f64
        } else {
            self.phi * (1.0 - self.phi.powi(h as i32)) / (1.0 - self.phi)
        }
    }
}

impl Leaf for HoltLeaf {
    fn name(&self) -> &'static str {
        "holt_damped"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let level = self.level.unwrap_or(0.0);
        let base = self.sigma();
        (1..=horizon)
            .map(|h| {
                let mean = level + self.damped_sum(h) * self.trend;
                Gaussian::new(mean, base * (h as f64).sqrt())
            })
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let level = self.level.unwrap_or(0.0);
        Gaussian::new(level + self.phi * self.trend, self.sigma())
    }

    fn observe(&mut self, y: f64) {
        let level_prev = self.level.unwrap_or(y);
        let predicted = level_prev + self.phi * self.trend;
        let resid = y - predicted;

        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        let new_level = self.alpha * y + (1.0 - self.alpha) * (level_prev + self.phi * self.trend);
        let new_trend =
            self.beta * (new_level - level_prev) + (1.0 - self.beta) * self.phi * self.trend;
        self.level = Some(new_level);
        self.trend = new_trend;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_onto_a_linear_trend_pure_holt() {
        let mut leaf = HoltLeaf::new(0.3, 0.1, 1.0);
        for i in 0..400 {
            leaf.observe(5.0 + 2.0 * i as f64);
        }
        let preds = leaf.predict(3);
        // Next observation should be 5 + 2*400 = 805; step size ≈ 2.
        assert!(
            (preds[0].mean - 805.0).abs() < 1.0,
            "h=1 pred {} vs expected ~805",
            preds[0].mean
        );
        assert!(
            (preds[1].mean - preds[0].mean - 2.0).abs() < 0.05,
            "step h=1→h=2 should be ~2, got {}",
            preds[1].mean - preds[0].mean
        );
    }

    #[test]
    fn damping_shrinks_far_horizon_trend() {
        let mut damped = HoltLeaf::new(0.3, 0.1, 0.9);
        let mut pure = HoltLeaf::new(0.3, 0.1, 1.0);
        for i in 0..300 {
            let y = 10.0 + 2.0 * i as f64;
            damped.observe(y);
            pure.observe(y);
        }
        let dp = damped.predict(20);
        let pp = pure.predict(20);
        // At h=20 the damped forecast should be materially below the
        // pure Holt forecast because the trend has decayed.
        assert!(dp[19].mean < pp[19].mean);
    }

    #[test]
    fn flat_series_gives_flat_forecast() {
        let mut leaf = HoltLeaf::new(0.3, 0.1, 0.98);
        for _ in 0..200 {
            leaf.observe(42.0);
        }
        let preds = leaf.predict(5);
        for p in preds {
            assert!((p.mean - 42.0).abs() < 0.5);
        }
    }
}
