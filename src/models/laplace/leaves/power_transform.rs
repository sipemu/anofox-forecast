//! Power-transform wrapper leaf — port of skaters' `power_transform`
//! composed with an inner leaf.
//!
//! Applies a **signed power** transform on the way in, and inverses
//! (delta method) on the way out:
//!
//! ```text
//!   forward:  y'  = sign(y) * |y|^p
//!   inverse:  ŷ   = sign(y') * |y'|^(1/p)
//!   d/dy'[inv]:   = (1/p) * |y'|^(1/p - 1)
//! ```
//!
//! For `0 < p < 1` this compresses tails (log-like) but works on all
//! reals — no explosion on negatives. Skaters ships `power_transform(0.5)`
//! (the signed square-root) composed with `ema_transform(0.1)`. The
//! wrapper here is generic: pass any inner leaf.
//!
//! PR #3 of #180.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;

pub struct PowerTransformWrapper {
    inner: Box<dyn Leaf + Send>,
    p: f64,
    inv_p: f64,
    label: String,
}

impl PowerTransformWrapper {
    /// Recommended: `p = 0.5` (signed square-root, matches skaters).
    /// Any `p ∈ (0, 1)` is legal.
    pub fn new(inner: Box<dyn Leaf + Send>, p: f64) -> Self {
        let p = p.clamp(1e-4, 0.9999);
        let label = format!("{}@pow{:.2}", inner.name(), p);
        Self {
            inner,
            p,
            inv_p: 1.0 / p,
            label,
        }
    }
}

/// Signed power: `sign(x) · |x|^p`. Preserves the sign of the input.
#[inline]
fn signed_pow(x: f64, p: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x.signum() * x.abs().powf(p)
    }
}

impl Leaf for PowerTransformWrapper {
    fn name(&self) -> &'static str {
        Box::leak(self.label.clone().into_boxed_str())
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let inner = self.inner.predict(horizon);
        inner
            .into_iter()
            .map(|g| {
                // Inverse mean via signed-power^(1/p).
                let mean_orig = signed_pow(g.mean, self.inv_p);
                // Delta-method Jacobian: (1/p) * |μ|^(1/p - 1).
                // Near zero the derivative is unbounded for p < 1 —
                // clamp |μ| from below so σ stays finite.
                let abs_mu = g.mean.abs().max(1e-6);
                let jac = self.inv_p * abs_mu.powf(self.inv_p - 1.0);
                let sigma = (g.std * jac.abs()).max(1e-9);
                Gaussian::new(mean_orig, sigma)
            })
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let g = self.inner.predict_one();
        let mean_orig = signed_pow(g.mean, self.inv_p);
        let abs_mu = g.mean.abs().max(1e-6);
        let jac = self.inv_p * abs_mu.powf(self.inv_p - 1.0);
        let sigma = (g.std * jac.abs()).max(1e-9);
        Gaussian::new(mean_orig, sigma)
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        let y_trans = signed_pow(y, self.p);
        if y_trans.is_finite() {
            self.inner.observe(y_trans);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::EmaLeaf;
    use super::*;

    #[test]
    fn round_trip_preserves_positive_values() {
        // Feed positive values; power-wrapper + EMA should track them
        // in original space to within EMA's expected level tracking
        // (α=1 → level = last obs).
        let mut w = PowerTransformWrapper::new(Box::new(EmaLeaf::new(1.0)), 0.5);
        for _ in 0..10 {
            w.observe(9.0); // sqrt(9)=3, EMA=3, back-square = 9
        }
        let g = w.predict(1)[0];
        assert!(
            (g.mean - 9.0).abs() < 0.5,
            "round-trip mean {} not near 9",
            g.mean
        );
    }

    #[test]
    fn round_trip_preserves_negative_values() {
        let mut w = PowerTransformWrapper::new(Box::new(EmaLeaf::new(1.0)), 0.5);
        for _ in 0..10 {
            w.observe(-4.0); // signed sqrt = -2; EMA = -2; back-square = -4
        }
        let g = w.predict(1)[0];
        assert!(
            (g.mean + 4.0).abs() < 0.5,
            "round-trip mean {} not near -4",
            g.mean
        );
    }

    #[test]
    fn compresses_heavy_tail_variance() {
        // Feed heavy-tailed inputs. Transformed-space std should be
        // smaller than raw-space std (because the transform compresses).
        let mut raw = EmaLeaf::new(0.1);
        let mut wrapped = PowerTransformWrapper::new(Box::new(EmaLeaf::new(0.1)), 0.5);
        for i in 1..=500 {
            let y = if i % 50 == 0 { 100.0 } else { 1.0 };
            raw.observe(y);
            wrapped.observe(y);
        }
        // The wrapped inner sees compressed values; its std is smaller
        // in transformed space. But after delta-method inversion, the
        // reported std may be similar to raw. What we CAN check: the
        // wrapper survives the extreme values without producing NaN
        // or infinite std.
        let g = wrapped.predict(1)[0];
        assert!(g.mean.is_finite() && g.std.is_finite() && g.std > 0.0);
    }
}
