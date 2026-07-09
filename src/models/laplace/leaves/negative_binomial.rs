//! Negative-Binomial leaf — moment-matched Gaussian output.
//!
//! Overdispersed count series (variance > mean) are the retail-demand
//! norm and don't fit Poisson (which enforces `variance = mean`). NB has
//! two parameters: mean μ and dispersion r, with `variance = μ + μ²/r`.
//!
//! This leaf tracks the mean via EMA and the variance via a Welford
//! recursion on residuals against the running mean. At `predict` time it
//! recovers `r = μ² / (σ² - μ)` when `σ² > μ`, then outputs
//! `Gaussian(μ, √((μ + μ²/r) · h))`.
//!
//! When observed variance is close to or below the mean (Poisson-like),
//! `r → ∞` and the leaf collapses to `Gaussian(μ, √(μ·h))` — same as
//! `PoissonLeaf`. This is intentional; NB nests Poisson.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NegativeBinomialLeaf {
    alpha: f64,
    mu_ema: f64,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl NegativeBinomialLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            mu_ema: 0.0,
            initialized: false,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
        }
    }

    fn empirical_variance(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).max(1e-9)
    }
}

impl Leaf for NegativeBinomialLeaf {
    fn name(&self) -> &'static str {
        "neg_binomial"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mu = self.mu_ema.max(0.0);
        let sigma_sq = self.empirical_variance();
        // NB variance = μ + μ²/r. If observed variance ≤ mean, r → ∞ and
        // we fall back to Poisson-style variance = mean.
        let base_var = if sigma_sq > mu + 1e-9 {
            sigma_sq
        } else {
            mu.max(1e-9)
        };
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
        // Welford recursion on the observed value directly (approximating
        // the marginal variance of the process, not the residual variance).
        self.n += 1;
        let delta = y - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (y - self.mean_resid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overdispersed_variance_exceeds_mean() {
        let mut leaf = NegativeBinomialLeaf::new(0.05);
        // Sequence with mean ≈ 2, variance ≈ 5.
        let seq = [0.0, 5.0, 0.0, 3.0, 4.0, 0.0, 6.0, 0.0, 2.0, 0.0];
        for _ in 0..30 {
            for &y in &seq {
                leaf.observe(y);
            }
        }
        let preds = leaf.predict(1);
        assert!(preds[0].mean > 1.0 && preds[0].mean < 3.0);
        // NB std should exceed Poisson-style √μ.
        let sqrt_mu = preds[0].mean.sqrt();
        assert!(
            preds[0].std > sqrt_mu,
            "std {} should exceed √μ {} for overdispersed data",
            preds[0].std,
            sqrt_mu
        );
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = NegativeBinomialLeaf::new(0.1);
        leaf.observe(0.0);
        leaf.observe(4.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
