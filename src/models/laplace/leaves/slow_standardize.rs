//! Slow-standardize wrapper — "thinking fast and slow" leaf composition.
//!
//! Ports skaters' fast-slow decomposition pattern:
//!
//! ```text
//! y --[fast tracker]--> residual --[slow standardize]--> z --> Gaussian
//! ```
//!
//! The idea: a *fast* mean tracker reacts to every observation (α ~ 0.3-0.5)
//! but its instantaneous variance estimate is unstable. Wrap it in a
//! *slow* variance EWMA (α ~ 0.02-0.05) so the emitted spread reflects
//! how much noise really is left after the tracker's mean forecast. The
//! slow EWMA's effective memory is an order of magnitude longer than the
//! tracker's, so short bursts of noise don't blow up the density.
//!
//! Empirically the softmax ensemble then picks the tracker whose *mean*
//! fits the series while the slow scale tames its *spread*. This is one
//! of skaters' larger contributions — a lot of what looks like heavy
//! tails in a Gaussian pool is really "the fast tracker's variance
//! estimate hasn't converged."
//!
//! PR #2 of #180.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;

/// Wraps an inner `Leaf` and replaces its emitted std with a slow-EWMA
/// tracked residual std. The inner leaf's mean is used verbatim.
pub struct SlowStandardizeWrapper {
    inner: Box<dyn Leaf + Send>,
    slow_alpha: f64,
    v_slow: f64,
    n_obs: usize,
}

impl SlowStandardizeWrapper {
    /// `slow_alpha` = residual-variance EWMA rate. Typical: 0.02 or 0.05.
    pub fn new(inner: Box<dyn Leaf + Send>, slow_alpha: f64) -> Self {
        Self {
            inner,
            slow_alpha: slow_alpha.clamp(1e-4, 1.0),
            v_slow: 0.0,
            n_obs: 0,
        }
    }
}

impl Leaf for SlowStandardizeWrapper {
    fn name(&self) -> &'static str {
        "slow_std"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let inner = self.inner.predict(horizon);
        let sigma = if self.v_slow.is_finite() && self.v_slow > 0.0 {
            self.v_slow.sqrt()
        } else {
            // Bootstrap fallback until we've seen a few residuals.
            inner.first().map(|g| g.std).unwrap_or(1.0)
        };
        // Multi-step: the slow scale is a one-step residual variance
        // estimate. For h>1 the true predictive variance grows with h;
        // the skaters port here keeps the same spread across h and lets
        // the softmax ensemble pick alternative candidates for long
        // horizons. Follow-up in PR #3 could sqrt(h)-scale.
        inner
            .into_iter()
            .map(|g| Gaussian::new(g.mean, sigma.max(1e-9)))
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let g = self.inner.predict_one();
        let sigma = if self.v_slow.is_finite() && self.v_slow > 0.0 {
            self.v_slow.sqrt()
        } else {
            g.std
        };
        Gaussian::new(g.mean, sigma.max(1e-9))
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        // Take the fast tracker's current one-step forecast, compute
        // residual, update slow-variance EWMA, then absorb.
        let mean_hat = self.inner.predict(1).first().map(|g| g.mean).unwrap_or(0.0);
        let r = y - mean_hat;
        if r.is_finite() {
            self.n_obs += 1;
            let n = self.n_obs as f64;
            // Bootstrap: 1/n dominates until we've built up an average.
            let a = self.slow_alpha.max(1.0 / n);
            self.v_slow = (1.0 - a) * self.v_slow + a * r * r;
        }
        self.inner.observe(y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::laplace::leaves::EmaLeaf;

    fn gauss_ll(y: f64, mu: f64, sigma: f64) -> f64 {
        -0.5 * ((y - mu) / sigma).powi(2)
            - sigma.max(1e-30).ln()
            - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// The wrapper should keep the inner leaf's mean and replace the
    /// std with a slow-EWMA-tracked residual std. On IID Gaussian
    /// residuals with a fast tracker, the wrapped std should approach
    /// the true residual σ.
    #[test]
    fn wrapper_tracks_slow_residual_variance() {
        let inner = EmaLeaf::new(0.3);
        let mut w = SlowStandardizeWrapper::new(Box::new(inner), 0.05);
        let sigma_true = 2.0;
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
            w.observe(sigma_true * z);
        }
        let g = w.predict(1)[0];
        // Slow EWMA on IID-Gaussian residuals should converge to ~σ_true.
        // The fast tracker introduces bias in the residual for its own
        // level, but the EMA(0.3) tracker on zero-mean noise stays near
        // 0, so residual variance ≈ σ_true². Wide tolerance (large-α
        // EWMA has meaningful sampling noise).
        assert!(
            g.std > 0.7 * sigma_true && g.std < 1.5 * sigma_true,
            "slow σ={} not near true σ={}",
            g.std,
            sigma_true
        );
    }

    /// On a step-change series, the slow-standardize wrapper should
    /// produce a higher LL than the raw inner leaf whose fast σ
    /// underestimates the residual scale during the transient.
    #[test]
    fn beats_raw_inner_on_step_change() {
        let mk = || Box::new(EmaLeaf::new(0.3)) as Box<dyn Leaf + Send>;
        let mut inner = mk();
        let mut wrapped = SlowStandardizeWrapper::new(mk(), 0.03);
        // First half: N(0, 1). Second half: N(0, 4) — step-change in scale.
        let mut ys = Vec::new();
        for i in 1..=1000 {
            let u1 = ((i as f64 * 3.111).sin() * 43758.5453)
                .fract()
                .abs()
                .max(1e-9);
            let u2 = ((i as f64 * 5.777).cos() * 12345.6789)
                .fract()
                .abs()
                .max(1e-9);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let sigma = if i < 500 { 1.0 } else { 4.0 };
            ys.push(sigma * z);
        }
        let mut ll_inner = 0.0;
        let mut ll_wrapped = 0.0;
        for (i, y) in ys.iter().enumerate() {
            // Score the h=1 predictive at *each* step BEFORE observing.
            let gi = inner.predict(1)[0];
            let gw = wrapped.predict(1)[0];
            if i > 800 {
                ll_inner += gauss_ll(*y, gi.mean, gi.std);
                ll_wrapped += gauss_ll(*y, gw.mean, gw.std);
            }
            inner.observe(*y);
            wrapped.observe(*y);
        }
        assert!(
            ll_wrapped > ll_inner,
            "wrapped LL {ll_wrapped} did not beat raw-inner LL {ll_inner} on step-change scale"
        );
    }
}
