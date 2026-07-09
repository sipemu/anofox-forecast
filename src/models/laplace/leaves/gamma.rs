//! Gamma leaf — moment-matched Gaussian output.
//!
//! Positive-skewed continuous data (e.g. lead times, waiting times,
//! sometimes retail sizes) fits Gamma with shape `k`, scale `θ`, mean
//! `kθ`, variance `kθ²`. This leaf tracks the mean via EMA and the
//! variance via Welford; the recovered `(k, θ) = (μ²/σ², σ²/μ)` are not
//! stored explicitly (moment-match at output time is equivalent for the
//! Gaussian projection) but *are* meaningful when Level 2 typed output
//! lands.
//!
//! Output: `Gaussian(μ, √(σ²·h))` — same shape as
//! [`NegativeBinomialLeaf`](super::NegativeBinomialLeaf) but on continuous
//! data. Non-negative clamp on `μ` guards near zero.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GammaLeaf {
    alpha: f64,
    mu_ema: f64,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_y: f64,
}

impl GammaLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            mu_ema: 0.0,
            initialized: false,
            n: 0,
            ss: 0.0,
            mean_y: 0.0,
        }
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).max(1e-9)
    }
}

impl Leaf for GammaLeaf {
    fn name(&self) -> &'static str {
        "gamma"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mu = self.mu_ema.max(0.0);
        let base_var = self.variance();
        (1..=horizon)
            .map(|h| Gaussian::new(mu, (base_var * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let y = y.max(0.0);
        if !self.initialized {
            self.mu_ema = y;
            self.initialized = true;
        } else {
            self.mu_ema = self.alpha * y + (1.0 - self.alpha) * self.mu_ema;
        }
        self.n += 1;
        let delta = y - self.mean_y;
        self.mean_y += delta / self.n as f64;
        self.ss += delta * (y - self.mean_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_gamma_like_positive_data() {
        let mut leaf = GammaLeaf::new(0.05);
        // Skewed positive data.
        let seq = [0.5, 1.0, 2.0, 4.0, 1.5, 0.8, 3.0, 1.2, 2.5, 0.7];
        for _ in 0..50 {
            for &y in &seq {
                leaf.observe(y);
            }
        }
        let preds = leaf.predict(1);
        assert!(preds[0].mean > 0.5 && preds[0].mean < 3.0);
        assert!(preds[0].std > 0.0);
    }

    #[test]
    fn negative_input_clamped_to_zero_domain() {
        let mut leaf = GammaLeaf::new(0.1);
        leaf.observe(-1.0);
        leaf.observe(2.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
