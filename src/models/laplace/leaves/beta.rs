//! Beta leaf — moment-matched Gaussian output for bounded `[0, 1]` data.
//!
//! Useful when the demand quantity is a **rate** or **proportion**:
//! promotional-uplift ratios, service levels, capacity fills, conversion
//! rates. The user is responsible for scaling into `[0, 1]`; observations
//! outside are clamped.
//!
//! Beta with mean μ and variance σ² has parameters `(α, β)` where
//!
//! ```text
//!   α + β = μ(1 − μ) / σ² − 1
//!   α     = μ · (α + β)
//!   β     = (1 − μ) · (α + β)
//! ```
//!
//! We track μ and σ² directly via EMA + Welford; the shape parameters
//! are recoverable at forecast time but not stored explicitly (moment
//! matching to Gaussian is equivalent for the output projection).
//!
//! Predictive std: `√(σ² · h)`. When variance grows past the
//! maximum-possible Beta variance `μ · (1 − μ)`, we clamp — this
//! keeps the output distribution self-consistent even for near-degenerate
//! data.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BetaLeaf {
    alpha: f64,
    mu_ema: f64,
    initialized: bool,
    n: usize,
    m2: f64,
    m1: f64,
}

impl BetaLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            mu_ema: 0.5,
            initialized: false,
            n: 0,
            m2: 0.0,
            m1: 0.0,
        }
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            return 0.01;
        }
        (self.m2 / (self.n as f64 - 1.0)).max(1e-9)
    }
}

impl Leaf for BetaLeaf {
    fn name(&self) -> &'static str {
        "beta"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mu = self.mu_ema.clamp(1e-6, 1.0 - 1e-6);
        // Cap variance at the maximum-possible Beta variance μ(1−μ).
        let var_cap = mu * (1.0 - mu);
        let var = self.variance().min(var_cap * 0.99);
        (1..=horizon)
            .map(|h| Gaussian::new(mu, (var * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let y = y.clamp(0.0, 1.0);
        if !self.initialized {
            self.mu_ema = y;
            self.initialized = true;
        } else {
            self.mu_ema = self.alpha * y + (1.0 - self.alpha) * self.mu_ema;
        }
        self.n += 1;
        let delta = y - self.m1;
        self.m1 += delta / self.n as f64;
        self.m2 += delta * (y - self.m1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_on_to_bounded_rate() {
        let mut leaf = BetaLeaf::new(0.05);
        // Rate oscillating around 0.3.
        for i in 0..500 {
            let y = 0.3 + ((i as f64 * 0.1).sin() * 0.1);
            leaf.observe(y);
        }
        let preds = leaf.predict(3);
        assert!(
            (preds[0].mean - 0.3).abs() < 0.05,
            "expected ~0.3, got {}",
            preds[0].mean
        );
        // Output should stay in valid range.
        assert!(preds[0].mean >= 0.0 && preds[0].mean <= 1.0);
        assert!(preds[0].std > 0.0);
    }

    #[test]
    fn observations_outside_unit_interval_are_clamped() {
        let mut leaf = BetaLeaf::new(0.1);
        leaf.observe(-0.5);
        leaf.observe(1.5);
        leaf.observe(0.3);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite() && (0.0..=1.0).contains(&p.mean));
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
