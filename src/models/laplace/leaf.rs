//! `Leaf` trait — the online predictor unit that laplace ensembles over.
//!
//! A leaf is a small streaming model that consumes observations one at a
//! time and emits per-horizon `Gaussian` predictions. The ensemble in
//! [`ensemble`](super::ensemble) tracks each leaf's cumulative log-likelihood
//! and turns those into softmax weights.

use super::dist::Gaussian;

/// Streaming per-horizon predictor.
pub trait Leaf {
    fn name(&self) -> &'static str;

    /// Return the *current* h-step-ahead predictive distributions, one per
    /// horizon in `1..=horizon`. Called *before* the next observation is
    /// absorbed, so the returned distributions are honest one-step (and
    /// h-step) forecasts.
    fn predict(&self, horizon: usize) -> Vec<Gaussian>;

    /// One-step-only predict — no `Vec` allocation. Called on the fit
    /// hot path (per leaf, per observation). Default forwards to
    /// `predict(1)[0]`; hot leaves override with a direct return to
    /// skip the intermediate allocation.
    fn predict_one(&self) -> Gaussian {
        let v = self.predict(1);
        v.first()
            .copied()
            .unwrap_or_else(|| Gaussian::new(0.0, 1.0))
    }

    /// Absorb one observation and update internal state.
    fn observe(&mut self, y: f64);
}
