//! Tweedie (compound Poisson-gamma) leaf — moment-matched Gaussian output.
//!
//! Aggregate retail (SKU × store × week) is characterized by:
//! - a point mass at 0 (no sales occurred),
//! - a positive-continuous branch (aggregate amount when sales did occur),
//! - overdispersion (aggregate variance > mean).
//!
//! The Tweedie compound Poisson-gamma distribution captures all three
//! naturally without a hurdle: `Y = ∑_{i=1..N} G_i` where `N ~ Poi(λ)`
//! and each `G_i ~ Gamma(k, θ)`. The power parameter `p ∈ (1, 2)`
//! interpolates between Poisson (`p = 1`) and Gamma (`p = 2`).
//!
//! Mean-variance relationship: `Var[Y] = φ · μ^p` where `φ` is the
//! dispersion. This leaf tracks `μ_ema` and empirical variance, then
//! recovers `φ` at forecast time via `φ = σ² / μ^p`. The power `p` is
//! set from the ratio of zero-mass to variance/mean — for retail
//! aggregates `p ≈ 1.5` is common; we fix `p = 1.5` unless overridden.
//!
//! Output is `Gaussian(μ, √(φ · μ^p · h))` — the moment-match. The
//! true Tweedie shape (zero-mass + Gamma continuous positive branch)
//! is not preserved end-to-end; a `TypedMixture` route (planned α-24
//! Level 2) would fix that.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TweedieLeaf {
    alpha: f64,
    p: f64,
    mu_ema: f64,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_y: f64,
}

impl TweedieLeaf {
    /// `p ∈ (1, 2)` — the Tweedie power. `p = 1.5` is a canonical retail-
    /// aggregate default. Values outside `(1.001, 1.999)` are clamped.
    pub fn new(alpha: f64, p: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            p: p.clamp(1.001, 1.999),
            mu_ema: 0.0,
            initialized: false,
            n: 0,
            ss: 0.0,
            mean_y: 0.0,
        }
    }

    /// Tweedie leaf with the canonical `p = 1.5`.
    pub fn with_default_p(alpha: f64) -> Self {
        Self::new(alpha, 1.5)
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).max(1e-9)
    }

    /// Recovered dispersion φ from observed variance and mean.
    fn dispersion(&self) -> f64 {
        let mu = self.mu_ema.max(1e-6);
        let mu_p = mu.powf(self.p);
        (self.variance() / mu_p).max(1e-9)
    }
}

impl Leaf for TweedieLeaf {
    fn name(&self) -> &'static str {
        "tweedie"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mu = self.mu_ema.max(0.0);
        let phi = self.dispersion();
        let base_var = phi * mu.max(1e-9).powf(self.p);
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
    fn locks_on_to_zero_inflated_positive_data() {
        let mut leaf = TweedieLeaf::with_default_p(0.05);
        // 30% zeros, 70% positive with variable magnitude.
        for i in 0..500 {
            let y = if i % 10 < 3 {
                0.0
            } else {
                3.0 + (i % 5) as f64
            };
            leaf.observe(y);
        }
        let preds = leaf.predict(3);
        assert!(preds[0].mean >= 0.0);
        assert!(preds[0].std > 0.0);
        assert!(preds[0].mean < 6.0);
    }

    #[test]
    fn p_clamped_to_open_interval() {
        let leaf_low = TweedieLeaf::new(0.1, 0.5);
        let leaf_high = TweedieLeaf::new(0.1, 2.5);
        assert!(leaf_low.p > 1.0 && leaf_low.p < 2.0);
        assert!(leaf_high.p > 1.0 && leaf_high.p < 2.0);
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = TweedieLeaf::with_default_p(0.1);
        leaf.observe(0.0);
        leaf.observe(5.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
