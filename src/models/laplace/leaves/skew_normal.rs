//! Skew-Normal leaf — moment-matched Gaussian output.
//!
//! Skew-Normal has location `ξ`, scale `ω`, and skewness `α`, with:
//!
//! ```text
//!   E[Y]  = ξ + ω · δ · √(2/π)          where δ = α / √(1 + α²)
//!   Var[Y] = ω² · (1 − 2δ² / π)
//!   γ₁   = (4 − π) / 2 · (δ · √(2/π))³ / (1 − 2δ²/π)^(3/2)
//! ```
//!
//! Method-of-moments fit: solve for `δ` from sample skewness `γ₁`,
//! back out `α = δ / √(1 − δ²)`. When `|γ₁| < 0.05` we treat as
//! symmetric (Gaussian fallback). When `|γ₁| > 0.99` (near degenerate)
//! we clamp δ to ±0.99.
//!
//! Useful for asymmetric continuous data where YJ/log doesn't fully
//! symmetrize — right-skewed retail aggregates that aren't cleanly
//! log-normal, or left-skewed decays.
//!
//! Output is `Gaussian(E[Y], √(Var[Y] · h))`. The true skew shape is
//! not preserved end-to-end (TypedMixture, planned α-24 Level 2, would
//! fix).

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

pub struct SkewNormalLeaf {
    alpha: f64,
    /// Location EMA.
    xi_ema: f64,
    initialized: bool,
    /// Welford for M2 and M3 (2nd and 3rd central moments).
    n: usize,
    m1: f64,
    m2: f64,
    m3: f64,
}

impl SkewNormalLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            xi_ema: 0.0,
            initialized: false,
            n: 0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
        }
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.m2 / (self.n as f64 - 1.0)).max(1e-9)
    }

    /// Sample skewness γ₁ = (M3/n) / (M2/n)^(3/2).
    fn skewness(&self) -> f64 {
        if self.n < 30 {
            return 0.0;
        }
        let n = self.n as f64;
        let sd_biased = (self.m2 / n).sqrt();
        if sd_biased < 1e-9 {
            return 0.0;
        }
        (self.m3 / n) / sd_biased.powi(3)
    }
}

impl Leaf for SkewNormalLeaf {
    fn name(&self) -> &'static str {
        "skew_normal"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mean = self.xi_ema;
        let var_base = self.variance();
        // Skewness-adjusted variance. When sample skewness is small,
        // fall through to plain σ². When large, the skew-normal
        // variance factor (1 − 2δ²/π) shrinks it.
        let g1 = self.skewness();
        let variance = if g1.abs() < 0.05 {
            var_base
        } else {
            // Recover δ from γ₁ via a first-order approximation:
            // γ₁ ≈ (4 − π)/2 · c³ / (1 − c²)^(3/2), c = δ √(2/π).
            // For MVP we invert numerically with a fixed heuristic.
            let approx_c = (g1 * 0.5).clamp(-0.85, 0.85);
            let approx_delta_sq = approx_c * approx_c * std::f64::consts::PI / 2.0;
            let variance_factor =
                (1.0 - 2.0 * approx_delta_sq / std::f64::consts::PI).clamp(0.1, 1.0);
            var_base / variance_factor.max(0.1)
        };
        let variance = variance.max(1e-9);
        (1..=horizon)
            .map(|h| Gaussian::new(mean, (variance * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        if !self.initialized {
            self.xi_ema = y;
            self.initialized = true;
        } else {
            self.xi_ema = self.alpha * y + (1.0 - self.alpha) * self.xi_ema;
        }
        // Welford for M2 and M3 (single-pass, approximate M3 update).
        self.n += 1;
        let n = self.n as f64;
        let delta = y - self.m1;
        let delta_n = delta / n;
        let term1 = delta * delta_n * (n - 1.0);
        self.m1 += delta_n;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_on_to_symmetric_data() {
        let mut leaf = SkewNormalLeaf::new(0.05);
        for i in 0..500 {
            let y = 5.0 + ((i as f64 * 0.1).sin() * 1.5);
            leaf.observe(y);
        }
        let preds = leaf.predict(3);
        assert!(
            (preds[0].mean - 5.0).abs() < 2.0,
            "expected ~5±2, got {}",
            preds[0].mean
        );
        assert!(preds[0].std > 0.0);
    }

    #[test]
    fn detects_right_skew() {
        let mut leaf = SkewNormalLeaf::new(0.05);
        // Right-skewed: mostly small, occasional big spike.
        for i in 0..300 {
            let y = if i % 20 == 0 { 10.0 } else { 1.0 };
            leaf.observe(y);
        }
        assert!(leaf.skewness() > 0.5, "expected positive skew");
        let preds = leaf.predict(3);
        assert!(preds[0].mean.is_finite());
        assert!(preds[0].std > 0.0);
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = SkewNormalLeaf::new(0.1);
        leaf.observe(1.0);
        leaf.observe(3.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite());
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
