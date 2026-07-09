//! Poisson leaf — moment-matched Gaussian output.
//!
//! Small-count series with `variance ≈ mean` are best modeled by a
//! Poisson distribution. This leaf tracks a rate estimate `λ_ema` and
//! outputs `Gaussian(λ, √(λ·h))` — the moment-match of Poisson's
//! variance = mean property, with the `√h` growth convention shared by
//! all other leaves in the shell.
//!
//! Softmax weighting: when `logpdf` sees an observation `y = 0` under
//! this leaf's prediction `Gaussian(0.5, √0.5)`, the score is
//! `-0.5·(0/√0.5)² - ln(√0.5) - ln(√(2π)) ≈ -0.573` — a plausible density
//! for a Poisson with `λ = 0.5`, unlike a random-walk leaf that would
//! see the zero as extreme tail.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoissonLeaf {
    alpha: f64,
    lambda_ema: f64,
    initialized: bool,
}

impl PoissonLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            lambda_ema: 0.0,
            initialized: false,
        }
    }
}

impl Leaf for PoissonLeaf {
    fn name(&self) -> &'static str {
        "poisson"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let lambda = self.lambda_ema.max(0.0);
        let base_var = lambda.max(1e-9);
        (1..=horizon)
            .map(|h| Gaussian::new(lambda, (base_var * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let y = y.max(0.0); // Poisson has non-negative support.
        if !self.initialized {
            self.lambda_ema = y;
            self.initialized = true;
        } else {
            self.lambda_ema = self.alpha * y + (1.0 - self.alpha) * self.lambda_ema;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_on_to_a_low_rate() {
        let mut leaf = PoissonLeaf::new(0.1);
        // λ ≈ 0.5 series.
        for i in 0..500 {
            leaf.observe(if i % 2 == 0 { 1.0 } else { 0.0 });
        }
        let preds = leaf.predict(5);
        assert!(
            (preds[0].mean - 0.5).abs() < 0.15,
            "expected ~0.5, got {}",
            preds[0].mean
        );
        // Std should grow with sqrt(h).
        assert!(preds[4].std > preds[0].std);
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = PoissonLeaf::new(0.1);
        leaf.observe(0.0);
        leaf.observe(3.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
