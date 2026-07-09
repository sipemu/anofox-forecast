//! Discrete-Uniform leaf — moment-matched Gaussian output.
//!
//! For bounded small-count series where the demand takes values in
//! `{0, 1, ..., K}` and any value is roughly equally likely. Rare in
//! retail (heavily-skewed to zero) but useful for:
//!
//! - promo-count series (0..N promotions per period)
//! - capacity-limited demand (K = shelf capacity)
//! - service tickets per period with hard cap
//!
//! Discrete-Uniform on `{0, ..., K}` has:
//!
//! ```text
//!   E[Y]   = K / 2
//!   Var[Y] = (K² + 2K) / 12  =  (K + 1)² · 1/12  −  1/12
//! ```
//!
//! We estimate `K` as `max(observed) + safety margin` via a running max,
//! then output the theoretical mean and variance. Output is
//! `Gaussian(K/2, √(Var · h))`.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscreteUniformLeaf {
    /// Running max over observed values; used to infer K.
    k_estimate: f64,
    n: usize,
}

impl DiscreteUniformLeaf {
    pub fn new() -> Self {
        Self {
            k_estimate: 0.0,
            n: 0,
        }
    }
}

impl Default for DiscreteUniformLeaf {
    fn default() -> Self {
        Self::new()
    }
}

impl Leaf for DiscreteUniformLeaf {
    fn name(&self) -> &'static str {
        "discrete_uniform"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let k = self.k_estimate.max(1.0);
        let mean = k / 2.0;
        let var = (k * k + 2.0 * k) / 12.0;
        let var = var.max(1e-9);
        (1..=horizon)
            .map(|h| Gaussian::new(mean, (var * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let y_pos = y.max(0.0);
        if y_pos > self.k_estimate {
            self.k_estimate = y_pos;
        }
        self.n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_uniform_moments_at_known_k() {
        let mut leaf = DiscreteUniformLeaf::new();
        // Uniform 0..5.
        for i in 0..500 {
            leaf.observe((i % 6) as f64);
        }
        let preds = leaf.predict(3);
        // Expected: mean = 5/2 = 2.5, var = (25 + 10)/12 ≈ 2.92.
        assert!(
            (preds[0].mean - 2.5).abs() < 0.5,
            "expected ~2.5, got {}",
            preds[0].mean
        );
        assert!((preds[0].std - 1.71).abs() < 0.5);
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = DiscreteUniformLeaf::new();
        leaf.observe(2.0);
        leaf.observe(4.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
