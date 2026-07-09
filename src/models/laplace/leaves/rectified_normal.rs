//! Rectified-Normal leaf — hurdle model, moment-matched Gaussian output.
//!
//! Intermittent continuous demand — a positive continuous distribution
//! (approximated as normal on the positive branch) with a point mass at
//! zero. A hurdle representation is easier to fit than the true rectified
//! normal:
//!
//! ```text
//!   Y = 0                with probability p_zero
//!   Y ~ N(μ_pos, σ²_pos) with probability 1 - p_zero
//! ```
//!
//! Mixture moments:
//!
//! ```text
//!   E[Y]   = (1 - p_zero) · μ_pos
//!   Var[Y] = (1 - p_zero) · σ²_pos + p_zero · (1 - p_zero) · μ_pos²
//! ```
//!
//! Output is `Gaussian(E[Y], √(Var[Y] · h))`.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const ZERO_TOL: f64 = 1e-9;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RectifiedNormalLeaf {
    alpha: f64,
    /// EMA of the Bernoulli indicator for "y is zero".
    p_zero_ema: f64,
    p_zero_initialized: bool,
    /// EMA of the positive branch's mean.
    mu_pos_ema: f64,
    mu_pos_initialized: bool,
    /// Welford on positive observations (variance estimate for the
    /// positive branch).
    n_pos: usize,
    ss_pos: f64,
    mean_pos: f64,
}

impl RectifiedNormalLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            p_zero_ema: 0.0,
            p_zero_initialized: false,
            mu_pos_ema: 0.0,
            mu_pos_initialized: false,
            n_pos: 0,
            ss_pos: 0.0,
            mean_pos: 0.0,
        }
    }

    fn positive_variance(&self) -> f64 {
        if self.n_pos < 2 {
            // Use the running positive mean as a rough variance prior
            // (variance ≈ mean for Poisson-adjacent data).
            return self.mu_pos_ema.max(1e-6);
        }
        (self.ss_pos / (self.n_pos as f64 - 1.0)).max(1e-9)
    }
}

impl Leaf for RectifiedNormalLeaf {
    fn name(&self) -> &'static str {
        "rectified_normal"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let p_zero = self.p_zero_ema.clamp(0.0, 1.0);
        let mu_pos = self.mu_pos_ema.max(0.0);
        let sig_sq_pos = self.positive_variance();
        let expected = (1.0 - p_zero) * mu_pos;
        let variance = (1.0 - p_zero) * sig_sq_pos + p_zero * (1.0 - p_zero) * mu_pos * mu_pos;
        let variance = variance.max(1e-9);
        (1..=horizon)
            .map(|h| Gaussian::new(expected, (variance * h as f64).sqrt()))
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
            if !self.mu_pos_initialized {
                self.mu_pos_ema = y_pos;
                self.mu_pos_initialized = true;
            } else {
                self.mu_pos_ema = self.alpha * y_pos + (1.0 - self.alpha) * self.mu_pos_ema;
            }
            self.n_pos += 1;
            let delta = y_pos - self.mean_pos;
            self.mean_pos += delta / self.n_pos as f64;
            self.ss_pos += delta * (y_pos - self.mean_pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_hurdle_moments() {
        let mut leaf = RectifiedNormalLeaf::new(0.05);
        // 40% zeros, 60% Normal(10, 4).
        for i in 0..500 {
            let y = if i % 5 < 2 {
                0.0
            } else {
                10.0 + ((i as f64 * 0.7).sin() * 2.0)
            };
            leaf.observe(y);
        }
        let preds = leaf.predict(1);
        // Expected mean ≈ 0.6 · 10 = 6.
        assert!(
            (preds[0].mean - 6.0).abs() < 1.5,
            "expected ~6, got {}",
            preds[0].mean
        );
        assert!(preds[0].std > 0.0);
    }

    #[test]
    fn all_zeros_predicts_zero() {
        let mut leaf = RectifiedNormalLeaf::new(0.1);
        for _ in 0..30 {
            leaf.observe(0.0);
        }
        let preds = leaf.predict(3);
        assert!(preds[0].mean.abs() < 0.1);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
