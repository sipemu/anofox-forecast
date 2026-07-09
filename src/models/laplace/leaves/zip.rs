//! Zero-Inflated Poisson (ZIP) leaf — moment-matched Gaussian output.
//!
//! Hurdle model on top of Poisson:
//!
//! ```text
//!   Y = 0        with probability p_zero (structural zero)
//!   Y ~ Poi(λ)   with probability 1 − p_zero
//! ```
//!
//! Useful when the observed zero fraction exceeds what a pure Poisson
//! would predict — the "excess zeros" retail-SKU regime where a stock
//! item is genuinely absent from the demand-generating process on some
//! days (out-of-assortment, promotional pause, seasonal item off-season)
//! rather than merely drawing a Poisson zero.
//!
//! Mixture moments:
//!
//! ```text
//!   E[Y]   = (1 − p_zero) · λ
//!   Var[Y] = (1 − p_zero) · λ · (1 + p_zero · λ)
//! ```
//!
//! Output is `Gaussian(E[Y], √(Var[Y] · h))` — the moment-match.
//! Compared to plain [`PoissonLeaf`](super::PoissonLeaf), ZIP correctly
//! accounts for the structural zeros in its variance estimate,
//! preventing softmax overconfidence on high-zero-fraction series.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const ZERO_TOL: f64 = 1e-9;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ZeroInflatedPoissonLeaf {
    alpha: f64,
    p_zero_ema: f64,
    p_zero_initialized: bool,
    lambda_ema: f64,
    lambda_initialized: bool,
}

impl ZeroInflatedPoissonLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            p_zero_ema: 0.0,
            p_zero_initialized: false,
            lambda_ema: 0.0,
            lambda_initialized: false,
        }
    }
}

impl Leaf for ZeroInflatedPoissonLeaf {
    fn name(&self) -> &'static str {
        "zip"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let p_zero = self.p_zero_ema.clamp(0.0, 1.0);
        let lambda = self.lambda_ema.max(0.0);
        let mean = (1.0 - p_zero) * lambda;
        let variance = ((1.0 - p_zero) * lambda * (1.0 + p_zero * lambda)).max(1e-9);
        (1..=horizon)
            .map(|h| Gaussian::new(mean, (variance * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let is_zero = y.abs() < ZERO_TOL;
        let z_indicator = if is_zero { 1.0 } else { 0.0 };
        if !self.p_zero_initialized {
            self.p_zero_ema = z_indicator;
            self.p_zero_initialized = true;
        } else {
            self.p_zero_ema = self.alpha * z_indicator + (1.0 - self.alpha) * self.p_zero_ema;
        }
        if !is_zero {
            let y_pos = y.max(0.0);
            if !self.lambda_initialized {
                self.lambda_ema = y_pos;
                self.lambda_initialized = true;
            } else {
                self.lambda_ema = self.alpha * y_pos + (1.0 - self.alpha) * self.lambda_ema;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_on_to_high_zero_fraction_low_rate() {
        // 70% zeros; when non-zero, Poisson-like with mean ≈ 3.
        let mut leaf = ZeroInflatedPoissonLeaf::new(0.05);
        for i in 0..1000 {
            let y = if i % 10 < 7 {
                0.0
            } else {
                2.0 + (i % 3) as f64
            };
            leaf.observe(y);
        }
        let preds = leaf.predict(3);
        // Expected mean ≈ 0.3 · 3 = 0.9.
        assert!(
            (preds[0].mean - 0.9).abs() < 0.5,
            "expected ~0.9, got {}",
            preds[0].mean
        );
        assert!(preds[0].std > 0.0);
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = ZeroInflatedPoissonLeaf::new(0.1);
        leaf.observe(0.0);
        leaf.observe(2.0);
        leaf.observe(0.0);
        let preds = leaf.predict(4);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
