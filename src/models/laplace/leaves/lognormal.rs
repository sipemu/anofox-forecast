//! Log-normal leaf — moment-matched Gaussian output.
//!
//! Positive multiplicative processes (retail SKUs where each period's
//! demand is a fractional multiple of the last) fit `y ~ LogNormal(μ_ln,
//! σ_ln²)` well. This leaf works on `ln(y + 1)` — a standard trick that
//! extends log-normal to include zeros — tracking EMA + Welford of the
//! log-transformed series, then inverse-transforms via the log-normal
//! mean/variance formulas at forecast time:
//!
//! ```text
//!   E[Y] = exp(μ_ln + σ_ln²/2) - 1
//!   Var[Y] = (exp(σ_ln²) - 1) · exp(2μ_ln + σ_ln²)
//! ```
//!
//! Output is `Gaussian(E[Y], √(Var[Y] · h))` — a moment-match; the true
//! log-normal shape is not preserved end-to-end (Level 2 in the roadmap
//! would fix that).

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[inline]
fn log1p_nonneg(y: f64) -> f64 {
    y.max(0.0).ln_1p()
}

pub struct LogNormalLeaf {
    alpha: f64,
    mu_log_ema: f64,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_log: f64,
}

impl LogNormalLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            mu_log_ema: 0.0,
            initialized: false,
            n: 0,
            ss: 0.0,
            mean_log: 0.0,
        }
    }

    fn log_variance(&self) -> f64 {
        if self.n < 2 {
            return 0.25; // Reasonable prior — σ_ln ≈ 0.5 → e^0.25 ≈ 1.28x spread.
        }
        (self.ss / (self.n as f64 - 1.0)).max(1e-9)
    }
}

impl Leaf for LogNormalLeaf {
    fn name(&self) -> &'static str {
        "lognormal"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mu_ln = self.mu_log_ema;
        let sig_sq_ln = self.log_variance();
        let mean_y = (mu_ln + 0.5 * sig_sq_ln).exp() - 1.0;
        let mean_y = mean_y.max(0.0);
        // (exp(σ²) - 1) · exp(2μ + σ²)
        let var_y = (sig_sq_ln.exp() - 1.0) * (2.0 * mu_ln + sig_sq_ln).exp();
        let var_y = var_y.max(1e-9);
        (1..=horizon)
            .map(|h| Gaussian::new(mean_y, (var_y * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let log_y = log1p_nonneg(y);
        if !self.initialized {
            self.mu_log_ema = log_y;
            self.initialized = true;
        } else {
            self.mu_log_ema = self.alpha * log_y + (1.0 - self.alpha) * self.mu_log_ema;
        }
        self.n += 1;
        let delta = log_y - self.mean_log;
        self.mean_log += delta / self.n as f64;
        self.ss += delta * (log_y - self.mean_log);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_onto_multiplicative_process() {
        let mut leaf = LogNormalLeaf::new(0.05);
        // Series with geometric-mean roughly at 5.
        for i in 0..500 {
            let y = 5.0 * (1.0 + 0.3 * ((i as f64 * 0.1).sin()));
            leaf.observe(y);
        }
        let preds = leaf.predict(1);
        assert!(
            (preds[0].mean - 5.0).abs() < 2.0,
            "expected ~5, got {}",
            preds[0].mean
        );
        assert!(preds[0].mean >= 0.0);
    }

    #[test]
    fn zero_observations_dont_crash() {
        let mut leaf = LogNormalLeaf::new(0.1);
        for _ in 0..30 {
            leaf.observe(0.0);
            leaf.observe(3.0);
        }
        let preds = leaf.predict(5);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
