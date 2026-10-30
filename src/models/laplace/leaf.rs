//! `Leaf` trait — the online predictor unit that laplace ensembles over.
//!
//! A leaf is a small streaming model that consumes observations one at a
//! time and emits per-horizon `Gaussian` predictions. The ensemble in
//! [`ensemble`](super::ensemble) tracks each leaf's cumulative log-likelihood
//! and turns those into softmax weights.

use super::dist::Gaussian;

/// Interns a composite leaf label as a `&'static str` for [`Leaf::name`].
///
/// Each distinct label is allocated once and kept in a global table, so calling
/// `name()` repeatedly (once per forecast step) no longer leaks a fresh string
/// every time, and the strings stay reachable (no LeakSanitizer reports).
pub(crate) fn intern_label(label: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static LABELS: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut labels = LABELS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(&existing) = labels.get(label) {
        return existing;
    }
    let leaked: &'static str = Box::leak(label.to_owned().into_boxed_str());
    labels.insert(leaked);
    leaked
}

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

#[cfg(test)]
mod intern_tests {
    use super::intern_label;

    #[test]
    fn intern_label_returns_one_allocation_per_label() {
        let a = intern_label("ets@sd7");
        let b = intern_label(&String::from("ets@sd7"));
        assert_eq!(a, "ets@sd7");
        assert!(std::ptr::eq(a, b), "same label must reuse one allocation");
        assert_ne!(intern_label("ets@sd12"), a);
    }
}
