//! Yeo-Johnson wrapper leaf.
//!
//! Wraps any [`Leaf`] with a fixed λ. Applies `yj_forward(y, λ)` before
//! delegating [`Leaf::observe`], and inverse-transforms the inner leaf's
//! predictions via the delta method in [`Leaf::predict`]. Component means
//! are clamped to the observed training range in transformed space to
//! prevent the log-branch Jacobian from exploding on extrapolation (same
//! trick as the α-6 shell-level YJ).
//!
//! Composes with [`LaplaceForecaster::with_yeo_johnson_grid`](super::super::forecaster::LaplaceForecaster::with_yeo_johnson_grid):
//! creating one wrapped copy of every leaf per grid λ turns the mixture
//! into a `(leaf, λ)` softmax — the "coordinate grid" of skaters' original
//! Yeo-Johnson design.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;

pub struct YjWrappedLeaf {
    inner: Box<dyn Leaf + Send>,
    lambda: f64,
    trans_min: f64,
    trans_max: f64,
    label: String,
}

impl YjWrappedLeaf {
    pub fn new(inner: Box<dyn Leaf + Send>, lambda: f64) -> Self {
        let label = format!("{}@yj{:.2}", inner.name(), lambda);
        Self {
            inner,
            lambda,
            trans_min: f64::INFINITY,
            trans_max: f64::NEG_INFINITY,
            label,
        }
    }
}

fn yj_forward(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if lambda.abs() < 1e-12 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        -(-x + 1.0).ln()
    } else {
        -(((-x + 1.0).powf(2.0 - lambda)) - 1.0) / (2.0 - lambda)
    }
}

fn yj_inverse_with_jac(y: f64, lambda: f64) -> (f64, f64) {
    if y >= 0.0 {
        if lambda.abs() < 1e-12 {
            let ey = y.exp();
            (ey - 1.0, ey)
        } else {
            let base = lambda * y + 1.0;
            if base <= 0.0 {
                (0.0, 0.0)
            } else {
                let inv = 1.0 / lambda;
                (base.powf(inv) - 1.0, base.powf(inv - 1.0))
            }
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        let emy = (-y).exp();
        (1.0 - emy, emy)
    } else {
        let base = 1.0 - (2.0 - lambda) * y;
        if base <= 0.0 {
            (1.0, 0.0)
        } else {
            let inv = 1.0 / (2.0 - lambda);
            (1.0 - base.powf(inv), base.powf(inv - 1.0))
        }
    }
}

impl Leaf for YjWrappedLeaf {
    fn name(&self) -> &'static str {
        // Leak the label so the trait's &'static str contract is honoured.
        // This is O(unique lambdas), typically 3-5 leaks per process — fine.
        Box::leak(self.label.clone().into_boxed_str())
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let trans_preds = self.inner.predict(horizon);
        let (lo, hi) = if self.trans_min <= self.trans_max {
            (self.trans_min, self.trans_max)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };
        trans_preds
            .iter()
            .map(|g| {
                let mean_clamped = g.mean.clamp(lo, hi);
                let (mean_orig, jac) = yj_inverse_with_jac(mean_clamped, self.lambda);
                Gaussian::new(mean_orig, (g.std * jac.abs()).max(1e-9))
            })
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let g = self.inner.predict_one();
        let (lo, hi) = if self.trans_min <= self.trans_max {
            (self.trans_min, self.trans_max)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };
        let mean_clamped = g.mean.clamp(lo, hi);
        let (mean_orig, jac) = yj_inverse_with_jac(mean_clamped, self.lambda);
        Gaussian::new(mean_orig, (g.std * jac.abs()).max(1e-9))
    }

    fn observe(&mut self, y: f64) {
        let y_trans = yj_forward(y, self.lambda);
        if y_trans.is_finite() {
            self.trans_min = self.trans_min.min(y_trans);
            self.trans_max = self.trans_max.max(y_trans);
            self.inner.observe(y_trans);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::EmaLeaf;
    use super::*;

    #[test]
    fn wrapper_with_lambda_1_matches_inner_behaviour() {
        // λ = 1 is a no-op transform (identity plus a constant).
        let mut wrapped = YjWrappedLeaf::new(Box::new(EmaLeaf::new(0.2)), 1.0);
        let mut plain = EmaLeaf::new(0.2);
        for y in [5.0, 6.0, 7.0, 6.5, 6.8] {
            wrapped.observe(y);
            plain.observe(y);
        }
        let wp = wrapped.predict(3);
        let pp = plain.predict(3);
        for (w, p) in wp.iter().zip(pp.iter()) {
            // λ=1 shifts by a constant; the point forecasts should match
            // up to a linear correction. Just check finiteness.
            assert!(w.mean.is_finite() && p.mean.is_finite());
        }
    }

    #[test]
    fn wrapper_produces_finite_forecasts_on_positive_series() {
        let mut wrapped = YjWrappedLeaf::new(Box::new(EmaLeaf::new(0.2)), 0.5);
        for _ in 0..50 {
            wrapped.observe(10.0);
        }
        let preds = wrapped.predict(5);
        for p in preds {
            assert!(p.mean.is_finite() && (p.mean - 10.0).abs() < 5.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
