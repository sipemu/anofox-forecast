//! Student-t leaf — heavy-tailed continuous, moment-matched Gaussian.
//!
//! For financial/economic series with tail-events that Gaussian tails
//! underpredict. Student-t with degrees of freedom `ν` has:
//!
//! ```text
//!   E[Y] = μ
//!   Var[Y] = σ² · ν / (ν − 2)     when ν > 2
//! ```
//!
//! We fit ν via moment-matching between the sample kurtosis and the
//! t-distribution's excess kurtosis `6 / (ν − 4)`. The point forecast
//! is `μ_ema`; the variance is scaled from the sample estimate by
//! `ν / (ν − 2)` to reflect the true tail mass — softmax weighting then
//! sees a correctly heavy density around outliers instead of scoring
//! them as extreme-tail.
//!
//! When there's not enough data to estimate `ν` (n < 50) we default to
//! `ν = 5` — a middle-ground heavy-tail choice.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const NU_DEFAULT: f64 = 5.0;
const NU_MIN: f64 = 2.5;
const NU_MAX: f64 = 100.0;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StudentTLeaf {
    alpha: f64,
    mu_ema: f64,
    initialized: bool,
    /// Welford: mean, M2 (sum of squared deviations), M4 (sum of 4th
    /// power deviations — for kurtosis estimation).
    n: usize,
    m1: f64,
    m2: f64,
    m4: f64,
}

impl StudentTLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            mu_ema: 0.0,
            initialized: false,
            n: 0,
            m1: 0.0,
            m2: 0.0,
            m4: 0.0,
        }
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.m2 / (self.n as f64 - 1.0)).max(1e-9)
    }

    /// Estimated degrees of freedom from sample excess-kurtosis.
    /// `excess_kurt = 6 / (ν − 4)` for Student-t, so `ν = 6/excess + 4`.
    fn nu(&self) -> f64 {
        if self.n < 50 {
            return NU_DEFAULT;
        }
        let var = self.variance();
        if var < 1e-9 {
            return NU_DEFAULT;
        }
        // Sample excess kurtosis: (M4/n) / (M2/n)² − 3.
        let mean_m4 = self.m4 / self.n as f64;
        let var_biased = self.m2 / self.n as f64;
        let excess = mean_m4 / (var_biased * var_biased) - 3.0;
        if excess <= 0.1 {
            return NU_MAX; // Near-Gaussian → effectively unbounded ν.
        }
        (6.0 / excess + 4.0).clamp(NU_MIN, NU_MAX)
    }
}

impl Leaf for StudentTLeaf {
    fn name(&self) -> &'static str {
        "student_t"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mu = self.mu_ema;
        let var = self.variance();
        let nu = self.nu();
        // Student-t variance scaling: sample σ² underestimates the true
        // tail variance by ν / (ν − 2).
        let scaling = if nu > 2.0 { nu / (nu - 2.0) } else { 1.0 };
        let scaled_var = (var * scaling).max(1e-9);
        (1..=horizon)
            .map(|h| Gaussian::new(mu, (scaled_var * h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        if !self.initialized {
            self.mu_ema = y;
            self.initialized = true;
        } else {
            self.mu_ema = self.alpha * y + (1.0 - self.alpha) * self.mu_ema;
        }
        // Welford for M2 and M4 (higher-order moment tracking for ν).
        self.n += 1;
        let n = self.n as f64;
        let delta = y - self.m1;
        let delta_n = delta / n;
        let term1 = delta * delta_n * (n - 1.0);
        self.m1 += delta_n;
        self.m4 += term1 * delta_n * delta_n * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n * delta_n * self.m2
            - 4.0 * delta_n * self.m4_partial();
        self.m2 += term1;
    }
}

impl StudentTLeaf {
    // Placeholder for the M3-partial term in the Welford M4 update.
    // For this MVP we approximate as 0 (introduces a small kurtosis
    // bias but keeps the recursion single-pass and cheap).
    fn m4_partial(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_on_to_mean_and_variance() {
        let mut leaf = StudentTLeaf::new(0.05);
        // Symmetric series centered at 5, spread ~2.
        for i in 0..500 {
            let y = 5.0 + ((i as f64 * 0.13).sin() * 2.0);
            leaf.observe(y);
        }
        let preds = leaf.predict(3);
        // EMA over 500 samples with α=0.05 puts most weight on the last
        // few. Sine wave centered at 5 amplitude 2 → last-sample bias up
        // to ±2. Use a generous tolerance.
        assert!(
            (preds[0].mean - 5.0).abs() < 2.5,
            "expected ~5±2, got {}",
            preds[0].mean
        );
        assert!(preds[0].std > 0.5 && preds[0].std < 6.0);
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = StudentTLeaf::new(0.1);
        leaf.observe(1.0);
        leaf.observe(-1.0);
        leaf.observe(2.0);
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite());
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
