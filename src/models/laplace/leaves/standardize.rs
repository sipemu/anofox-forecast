//! Standardize wrapper — port of skaters' `standardize` transform.
//!
//! Wraps an inner leaf. On observe, applies the running-EWMA
//! standardization
//!
//! ```text
//!   diff = y - mu_prior
//!   y'   = diff / sigma_updated         ← inner leaf sees this
//!   mu  += alpha * diff
//!   var  = (1 - alpha) var + alpha diff²
//! ```
//!
//! On predict, applies the affine inverse `x -> mu + sigma * x` to
//! every horizon's Gaussian.
//!
//! Skaters ships this composed with EMA at `α ∈ {0.05, 0.1}` in the
//! depth-2 pool. The point is to give the inner leaf a stationary,
//! unit-variance view of the series so its density fits well
//! regardless of the raw scale — the "conform first, model last"
//! side of skaters' pool.
//!
//! PR #4 of #180.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;

pub struct StandardizeWrapper {
    inner: Box<dyn Leaf + Send>,
    alpha: f64,
    mu: f64,
    var: f64,
    initialized: bool,
    label: String,
}

impl StandardizeWrapper {
    /// `alpha` = EWMA rate for the running mean + variance. Skaters' default
    /// is `0.05`.
    pub fn new(inner: Box<dyn Leaf + Send>, alpha: f64) -> Self {
        let alpha = alpha.clamp(1e-4, 0.999);
        let label = format!("{}@std{:.2}", inner.name(), alpha);
        Self {
            inner,
            alpha,
            mu: 0.0,
            var: 0.0,
            initialized: false,
            label,
        }
    }

    fn sigma(&self) -> f64 {
        if self.var.is_finite() && self.var > 1e-16 {
            self.var.sqrt().max(1e-8)
        } else {
            1e-8
        }
    }
}

impl Leaf for StandardizeWrapper {
    fn name(&self) -> &'static str {
        Box::leak(self.label.clone().into_boxed_str())
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let sigma = self.sigma();
        let mu = self.mu;
        let inner = self.inner.predict(horizon);
        inner
            .into_iter()
            .map(|g| Gaussian::new(mu + sigma * g.mean, (sigma * g.std).max(1e-9)))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        if !self.initialized {
            self.mu = y;
            self.var = 0.0;
            self.initialized = true;
            // Feed a zero to the inner so its state initializes at the
            // "no residual" position.
            self.inner.observe(0.0);
            return;
        }
        let diff = y - self.mu;
        // Standardize against the PRIOR mean — centering by the post-
        // update mean would shrink the residual by (1 - alpha). Use the
        // updated variance (avoids cold-start divide-by-zero).
        self.mu += self.alpha * diff;
        self.var = (1.0 - self.alpha) * self.var + self.alpha * diff * diff;
        let sigma = self.sigma();
        let y_prime = diff / sigma;
        self.inner.observe(y_prime);
    }
}

#[cfg(test)]
mod tests {
    use super::super::EmaLeaf;
    use super::*;

    fn gauss_ll(y: f64, mu: f64, sigma: f64) -> f64 {
        -0.5 * ((y - mu) / sigma).powi(2)
            - sigma.max(1e-30).ln()
            - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    #[test]
    fn mean_tracks_series_after_warmup() {
        let mut w = StandardizeWrapper::new(Box::new(EmaLeaf::new(0.1)), 0.05);
        for _ in 0..500 {
            w.observe(42.0);
        }
        let g = w.predict(1)[0];
        assert!(
            (g.mean - 42.0).abs() < 1.0,
            "prediction mean {} not near 42",
            g.mean
        );
    }

    #[test]
    fn scale_tracks_residual_std() {
        let mut w = StandardizeWrapper::new(Box::new(EmaLeaf::new(0.1)), 0.05);
        let true_sigma = 3.0;
        for i in 1..=2000 {
            let u1 = ((i as f64 * 3.111).sin() * 43758.5453)
                .fract()
                .abs()
                .max(1e-9);
            let u2 = ((i as f64 * 5.777).cos() * 12345.6789)
                .fract()
                .abs()
                .max(1e-9);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            w.observe(true_sigma * z);
        }
        assert!(
            (w.sigma() - true_sigma).abs() < 1.0,
            "sigma {} not near true {}",
            w.sigma(),
            true_sigma
        );
    }

    /// On a series with a step-change in scale, standardize + EMA should
    /// score better than raw EMA on held-out log-likelihood because it
    /// re-scales quickly.
    #[test]
    fn beats_raw_ema_on_step_change() {
        let mk = || Box::new(EmaLeaf::new(0.1)) as Box<dyn Leaf + Send>;
        let mut raw = mk();
        let mut wrapped = StandardizeWrapper::new(mk(), 0.05);
        let mut ys = Vec::new();
        for i in 1..=1500 {
            let u1 = ((i as f64 * 3.111).sin() * 43758.5453)
                .fract()
                .abs()
                .max(1e-9);
            let u2 = ((i as f64 * 5.777).cos() * 12345.6789)
                .fract()
                .abs()
                .max(1e-9);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let scale = if i <= 500 { 1.0 } else { 5.0 };
            ys.push(scale * z);
        }
        let mut ll_raw = 0.0;
        let mut ll_wrap = 0.0;
        for (i, y) in ys.iter().enumerate() {
            let gr = raw.predict(1)[0];
            let gw = wrapped.predict(1)[0];
            if i > 1200 {
                ll_raw += gauss_ll(*y, gr.mean, gr.std);
                ll_wrap += gauss_ll(*y, gw.mean, gw.std);
            }
            raw.observe(*y);
            wrapped.observe(*y);
        }
        assert!(
            ll_wrap > ll_raw,
            "standardize LL {ll_wrap} did not beat raw LL {ll_raw} on step-change scale"
        );
    }
}
