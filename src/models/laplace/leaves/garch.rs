//! GARCH(1,1) volatility wrapper — port of skaters' `garch` transform.
//!
//! Divides each input by the conditional standard deviation before
//! passing it to the inner leaf; scales the inner's predictive
//! distribution by the same σ on the way out.
//!
//! ```text
//!   σ_t² = ω + α y_{t-1}² + β σ_{t-1}²
//!   y'_t = y_t / σ_t             ← inner leaf sees this
//!   D_out = D_inner.scale(σ_t)   ← recover original-space distribution
//! ```
//!
//! Stationarity requires `α + β < 1`; the unconditional variance is
//! `ω / (1 - α - β)`. Defaults `(ω, α, β) = (0.01, 0.1, 0.85)` match
//! skaters' default and are typical for financial return series.
//!
//! PR #3 of #180.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;

pub struct GarchWrappedLeaf {
    inner: Box<dyn Leaf + Send>,
    omega: f64,
    alpha: f64,
    beta: f64,
    var: f64,
    last_y: f64,
    initialized: bool,
    label: String,
}

impl GarchWrappedLeaf {
    /// Skaters' default: `ω = 0.01, α = 0.1, β = 0.85`. Stationarity
    /// requires `α + β < 1`.
    pub fn new(inner: Box<dyn Leaf + Send>, omega: f64, alpha: f64, beta: f64) -> Self {
        let omega = omega.max(1e-9);
        let alpha = alpha.max(0.0);
        let beta = beta.max(0.0);
        let label = format!("{}@garch", inner.name());
        Self {
            inner,
            omega,
            alpha,
            beta,
            var: 0.0,
            last_y: 0.0,
            initialized: false,
            label,
        }
    }

    /// Skaters' `garch()` default constructor equivalent.
    pub fn with_defaults(inner: Box<dyn Leaf + Send>) -> Self {
        Self::new(inner, 0.01, 0.1, 0.85)
    }

    fn conditional_sigma(&self) -> f64 {
        if self.var.is_finite() && self.var > 1e-16 {
            self.var.sqrt().max(1e-8)
        } else {
            self.omega.sqrt().max(1e-8)
        }
    }
}

impl Leaf for GarchWrappedLeaf {
    fn name(&self) -> &'static str {
        Box::leak(self.label.clone().into_boxed_str())
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let sigma_t = self.conditional_sigma();
        // The inner leaf's predictions are in standardized space; scale
        // them back by σ_t. Skaters keeps σ_t fixed across horizons —
        // the GARCH inverse doesn't grow with h. That's a simplification
        // (true GARCH multi-step variance is a mean-reverting geometric
        // series), but matches skaters' port for parity.
        let inner = self.inner.predict(horizon);
        inner
            .into_iter()
            .map(|g| Gaussian::new(g.mean * sigma_t, (g.std * sigma_t).max(1e-9)))
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let sigma_t = self.conditional_sigma();
        let g = self.inner.predict_one();
        Gaussian::new(g.mean * sigma_t, (g.std * sigma_t).max(1e-9))
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        if !self.initialized {
            // Bootstrap: use unconditional variance if stationary.
            let persist = self.alpha + self.beta;
            self.var = if persist < 1.0 {
                self.omega / (1.0 - persist)
            } else {
                self.omega
            };
            self.last_y = y;
            self.initialized = true;
            let sigma = self.conditional_sigma();
            self.inner.observe(y / sigma);
            return;
        }
        // Update conditional variance BEFORE dividing y — skaters' order
        // (updates based on last_y², then standardizes current y).
        self.var = self.omega + self.alpha * self.last_y * self.last_y + self.beta * self.var;
        let sigma = self.conditional_sigma();
        self.last_y = y;
        self.inner.observe(y / sigma);
    }
}

#[cfg(test)]
mod tests {
    use super::super::EmaLeaf;
    use super::*;

    #[test]
    fn absorbs_volatility_clustering() {
        // On a series with time-varying volatility, GARCH-wrapped leaf's
        // conditional σ should be much larger during the high-vol regime
        // than during the low-vol regime.
        let mut w = GarchWrappedLeaf::with_defaults(Box::new(EmaLeaf::new(0.1)));
        // First 200: low vol (σ=0.1). Next 200: high vol (σ=2.0).
        let mut hi_sigmas = Vec::new();
        let mut lo_sigmas = Vec::new();
        for i in 1..=400 {
            let u = ((i as f64 * 3.111).sin() * 43758.5453).fract() - 0.5;
            let scale = if i <= 200 { 0.1 } else { 2.0 };
            w.observe(scale * u);
            if i > 100 && i <= 200 {
                lo_sigmas.push(w.conditional_sigma());
            } else if i > 300 {
                hi_sigmas.push(w.conditional_sigma());
            }
        }
        let mean_lo: f64 = lo_sigmas.iter().sum::<f64>() / lo_sigmas.len() as f64;
        let mean_hi: f64 = hi_sigmas.iter().sum::<f64>() / hi_sigmas.len() as f64;
        assert!(
            mean_hi > 3.0 * mean_lo,
            "GARCH σ didn't track vol regime: lo={mean_lo:.3} hi={mean_hi:.3}"
        );
    }

    #[test]
    fn survives_extreme_values() {
        let mut w = GarchWrappedLeaf::with_defaults(Box::new(EmaLeaf::new(0.1)));
        for i in 1..=100 {
            w.observe(if i == 50 { 1000.0 } else { 0.1 });
        }
        let g = w.predict(1)[0];
        assert!(g.mean.is_finite() && g.std.is_finite() && g.std > 0.0);
    }

    #[test]
    fn nan_is_ignored() {
        let mut w = GarchWrappedLeaf::with_defaults(Box::new(EmaLeaf::new(0.1)));
        for _ in 0..20 {
            w.observe(0.5);
        }
        let before_sigma = w.conditional_sigma();
        w.observe(f64::NAN);
        w.observe(f64::INFINITY);
        assert!((w.conditional_sigma() - before_sigma).abs() < 1e-9);
    }
}
