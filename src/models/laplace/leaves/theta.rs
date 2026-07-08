//! Theta-method leaf — port of skaters' `theta` transform.
//!
//! The Theta method (Assimakopoulos & Nikolopoulos 2000) is one of the
//! strongest simple univariate forecasters — best in M3, near-best in
//! M4. Skaters ships it as a `theta(alpha)` transform in three variants
//! `α ∈ {0.05, 0.1, 0.3}`. This is the streaming leaf port.
//!
//! **Model:** SES level with a half-OLS-slope drift correction. At each
//! step,
//!
//! ```text
//!   forecast_t = level_{t-1} + slope_{t-1} / 2
//!   level_t   = α y_t + (1 - α) level_{t-1}
//!   slope_t   = OLS slope of y on t through step t
//! ```
//!
//! At horizon `h` the mean is `level_t + h · slope_t / 2` (linear
//! extrapolation of the half-slope). Variance is tracked as an EWMA of
//! residual² and grows with `√h` across horizons.
//!
//! PR #3 of #180.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;

/// SES + half-OLS-slope leaf with EWMA residual variance.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThetaLeaf {
    alpha: f64,
    // SES level.
    level: f64,
    initialized: bool,
    // Running OLS accumulators for `y ~ a + b · t`.
    n: f64,
    sum_t: f64,
    sum_t2: f64,
    sum_y: f64,
    sum_ty: f64,
    // Current OLS slope estimate (updated after each observe).
    slope: f64,
    // EWMA of squared residual for the variance channel.
    var_alpha: f64,
    var: f64,
    n_obs: usize,
}

impl ThetaLeaf {
    /// Skaters ships α ∈ {0.05, 0.1, 0.3}. Use these for the standard
    /// pool; other values are legal.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-4, 0.999),
            level: 0.0,
            initialized: false,
            n: 0.0,
            sum_t: 0.0,
            sum_t2: 0.0,
            sum_y: 0.0,
            sum_ty: 0.0,
            slope: 0.0,
            // Residual-variance EWMA rate — matches the terminal leaf's
            // default, effective memory ~33 obs.
            var_alpha: 0.03,
            var: 0.0,
            n_obs: 0,
        }
    }
}

impl Leaf for ThetaLeaf {
    fn name(&self) -> &'static str {
        "theta"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let sigma_one = if self.var.is_finite() && self.var > 0.0 {
            self.var.sqrt()
        } else {
            1.0
        };
        (1..=horizon)
            .map(|h| {
                let mean = self.level + (h as f64) * self.slope / 2.0;
                // Variance grows with √h for a random-walk residual assumption.
                let sigma = (sigma_one * (h as f64).sqrt()).max(1e-9);
                Gaussian::new(mean, sigma)
            })
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let sigma_one = if self.var.is_finite() && self.var > 0.0 {
            self.var.sqrt()
        } else {
            1.0
        };
        Gaussian::new(self.level + self.slope / 2.0, sigma_one.max(1e-9))
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        if !self.initialized {
            self.level = y;
            self.initialized = true;
            self.n = 1.0;
            self.sum_t = 1.0;
            self.sum_t2 = 1.0;
            self.sum_y = y;
            self.sum_ty = y;
            self.slope = 0.0;
            self.n_obs = 1;
            return;
        }

        // Compute the one-step forecast made BEFORE folding y into the
        // state — its residual drives the variance EWMA.
        let forecast = self.level + self.slope / 2.0;
        let residual = y - forecast;
        if residual.is_finite() {
            self.n_obs += 1;
            let n = self.n_obs as f64;
            let a = self.var_alpha.max(1.0 / n);
            self.var = (1.0 - a) * self.var + a * residual * residual;
        }

        // SES update on level.
        self.level = self.alpha * y + (1.0 - self.alpha) * self.level;

        // Running OLS on (t, y). t = current step counter.
        self.n += 1.0;
        let t = self.n;
        self.sum_t += t;
        self.sum_t2 += t * t;
        self.sum_y += y;
        self.sum_ty += t * y;

        let denom = self.n * self.sum_t2 - self.sum_t * self.sum_t;
        self.slope = if denom.abs() > 1e-12 {
            (self.n * self.sum_ty - self.sum_t * self.sum_y) / denom
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_linear_trend_via_ols_slope() {
        let mut t = ThetaLeaf::new(0.1);
        // y = 3 * step + noise-free
        for step in 1..=200 {
            t.observe(step as f64 * 3.0);
        }
        // Half-slope drift is applied to the forecast: expect the slope
        // component to be very close to 3.0.
        assert!(
            (t.slope - 3.0).abs() < 0.2,
            "OLS slope {:.4} not near true 3.0",
            t.slope
        );
        // Level should track the last value tightly.
        assert!(
            (t.level - 600.0).abs() < 60.0,
            "SES level {:.2} not near 600 (last obs)",
            t.level
        );
    }

    #[test]
    fn multi_horizon_variance_grows_with_sqrt_h() {
        let mut t = ThetaLeaf::new(0.1);
        // Zero-mean random-walk-like path.
        for i in 1..=500 {
            let z = ((i as f64 * 3.111).sin() * 43758.5453).fract() - 0.5;
            t.observe(z);
        }
        let g = t.predict(4);
        // sqrt-h scaling: σ_4 ≈ 2 σ_1.
        let r = g[3].std / g[0].std;
        assert!(
            (r - 2.0).abs() < 0.01,
            "σ_4/σ_1 = {r:.3} not near 2.0 (√h scaling)"
        );
    }

    #[test]
    fn nan_is_ignored() {
        let mut t = ThetaLeaf::new(0.1);
        for step in 1..=50 {
            t.observe(step as f64);
        }
        let before = t.slope;
        t.observe(f64::NAN);
        t.observe(f64::INFINITY);
        assert_eq!(t.slope, before);
    }
}
