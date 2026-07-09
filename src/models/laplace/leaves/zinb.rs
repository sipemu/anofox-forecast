//! Zero-Inflated Negative-Binomial (ZINB) leaf — moment-matched Gaussian.
//!
//! Hurdle model on top of Negative-Binomial:
//!
//! ```text
//!   Y = 0          with probability p_zero (structural zero)
//!   Y ~ NB(μ, r)   with probability 1 − p_zero
//! ```
//!
//! This is the canonical retail-SKU count model: **overdispersed** counts
//! (variance > mean) with **excess zeros** (out-of-assortment, off-season).
//!
//! Mixture moments:
//!
//! ```text
//!   E[Y]   = (1 − p_zero) · μ
//!   Var[Y] = (1 − p_zero) · (μ + μ²/r) + p_zero · (1 − p_zero) · μ²
//! ```
//!
//! Falls back to [`ZeroInflatedPoissonLeaf`](super::ZeroInflatedPoissonLeaf)
//! when observed variance ≤ mean (equidispersed positive branch).

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const ZERO_TOL: f64 = 1e-9;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ZeroInflatedNegativeBinomialLeaf {
    alpha: f64,
    p_zero_ema: f64,
    p_zero_initialized: bool,
    mu_pos_ema: f64,
    mu_pos_initialized: bool,
    n_pos: usize,
    ss_pos: f64,
    mean_pos: f64,
}

impl ZeroInflatedNegativeBinomialLeaf {
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
            return self.mu_pos_ema.max(1e-6);
        }
        (self.ss_pos / (self.n_pos as f64 - 1.0)).max(1e-9)
    }
}

impl Leaf for ZeroInflatedNegativeBinomialLeaf {
    fn name(&self) -> &'static str {
        "zinb"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let p_zero = self.p_zero_ema.clamp(0.0, 1.0);
        let mu = self.mu_pos_ema.max(0.0);
        let sigma_sq_pos = self.positive_variance();
        // NB variance on the positive branch (falls back to Poisson when
        // observed ≤ mean).
        let nb_var = if sigma_sq_pos > mu + 1e-9 {
            sigma_sq_pos
        } else {
            mu.max(1e-9)
        };
        let mean = (1.0 - p_zero) * mu;
        let variance = ((1.0 - p_zero) * nb_var + p_zero * (1.0 - p_zero) * mu * mu).max(1e-9);
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
    fn locks_on_to_zero_inflated_overdispersed_counts() {
        // 60% zeros; when non-zero, overdispersed values 2–8.
        let mut leaf = ZeroInflatedNegativeBinomialLeaf::new(0.05);
        let pos_pattern = [2.0, 8.0, 3.0, 7.0, 4.0];
        for i in 0..500 {
            let y = if i % 5 < 3 { 0.0 } else { pos_pattern[i % 5] };
            leaf.observe(y);
        }
        let preds = leaf.predict(3);
        // Positive branch mean ≈ 4.8; overall mean ≈ 0.4 · 4.8 = 1.92.
        assert!(
            preds[0].mean > 1.0 && preds[0].mean < 3.0,
            "expected 1-3, got {}",
            preds[0].mean
        );
        assert!(
            preds[0].std > preds[0].mean.sqrt(),
            "overdispersed std should exceed √μ"
        );
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = ZeroInflatedNegativeBinomialLeaf::new(0.1);
        leaf.observe(0.0);
        leaf.observe(4.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
