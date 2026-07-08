//! Distributional primitives for the laplace forecaster.
//!
//! [`Gaussian`] is a single-component normal; [`GaussianMixture`] is a
//! weighted sum of them. Every horizon of a [`LaplaceForecaster`](super::LaplaceForecaster)
//! forecast is a `GaussianMixture` — its `.mean()` is the point forecast,
//! `.quantile()` powers intervals, `.logpdf()` powers the likelihood-based
//! leaf weighting.

use std::f64::consts::{PI, SQRT_2};

const SQRT_2PI: f64 = 2.506_628_274_631_000_7;
/// Precomputed `0.5 · ln(2π)` — appears in `Gaussian::logpdf` and
/// `GaussianMixture::logpdf`. Const so the compiler doesn't recompute
/// via `(2·π).ln()` every call.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_7;

/// Single normal component.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gaussian {
    pub mean: f64,
    pub std: f64,
}

impl Gaussian {
    pub const fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    pub fn variance(&self) -> f64 {
        self.std * self.std
    }

    /// Standard-normal PDF evaluated at `z`.
    fn phi(z: f64) -> f64 {
        (-0.5 * z * z).exp() / SQRT_2PI
    }

    /// Standard-normal CDF via `erf`.
    fn big_phi(z: f64) -> f64 {
        0.5 * (1.0 + erf(z / SQRT_2))
    }

    pub fn logpdf(&self, y: f64) -> f64 {
        let z = (y - self.mean) / self.std;
        -0.5 * z * z - self.std.ln() - HALF_LN_2PI
    }

    pub fn pdf(&self, y: f64) -> f64 {
        Self::phi((y - self.mean) / self.std) / self.std
    }

    pub fn cdf(&self, y: f64) -> f64 {
        Self::big_phi((y - self.mean) / self.std)
    }

    /// Inverse CDF (quantile). `p` must be in `(0, 1)`.
    pub fn quantile(&self, p: f64) -> f64 {
        self.mean + self.std * SQRT_2 * inv_erf(2.0 * p - 1.0)
    }
}

/// Weighted mixture of gaussians. Weights are normalised on construction.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianMixture {
    /// `(weight, component)` pairs. Weights sum to 1.
    pub components: Vec<(f64, Gaussian)>,
}

impl GaussianMixture {
    /// Build from `(weight, component)` pairs. Non-finite or non-positive
    /// weights are dropped; the surviving weights are re-normalised. Returns
    /// an empty mixture if nothing survives.
    pub fn new(pairs: impl IntoIterator<Item = (f64, Gaussian)>) -> Self {
        let mut kept: Vec<(f64, Gaussian)> = pairs
            .into_iter()
            .filter(|(w, _)| w.is_finite() && *w > 0.0)
            .collect();
        let sum: f64 = kept.iter().map(|(w, _)| *w).sum();
        if sum > 0.0 {
            for (w, _) in &mut kept {
                *w /= sum;
            }
        }
        Self { components: kept }
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Mixture mean: `Σ w_i · μ_i`.
    pub fn mean(&self) -> f64 {
        self.components.iter().map(|(w, g)| w * g.mean).sum()
    }

    /// Mixture variance: `Σ w_i (σ_i² + (μ_i − μ_mix)²)`.
    pub fn variance(&self) -> f64 {
        let mu = self.mean();
        self.components
            .iter()
            .map(|(w, g)| w * (g.variance() + (g.mean - mu).powi(2)))
            .sum()
    }

    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn pdf(&self, y: f64) -> f64 {
        self.components.iter().map(|(w, g)| w * g.pdf(y)).sum()
    }

    /// log-sum-exp of component log-densities.
    pub fn logpdf(&self, y: f64) -> f64 {
        if self.components.is_empty() {
            return f64::NEG_INFINITY;
        }
        let logs: Vec<f64> = self
            .components
            .iter()
            .map(|(w, g)| w.ln() + g.logpdf(y))
            .collect();
        let m = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return m;
        }
        m + logs.iter().map(|l| (l - m).exp()).sum::<f64>().ln()
    }

    pub fn cdf(&self, y: f64) -> f64 {
        self.components.iter().map(|(w, g)| w * g.cdf(y)).sum()
    }

    /// Quantile via bisection over the mixture CDF.
    pub fn quantile(&self, p: f64) -> f64 {
        if self.components.is_empty() {
            return f64::NAN;
        }
        if self.components.len() == 1 {
            return self.components[0].1.quantile(p);
        }
        // Bracket using the widest per-component range at 1e-9 tails.
        let lo = self
            .components
            .iter()
            .map(|(_, g)| g.quantile(1e-9))
            .fold(f64::INFINITY, f64::min);
        let hi = self
            .components
            .iter()
            .map(|(_, g)| g.quantile(1.0 - 1e-9))
            .fold(f64::NEG_INFINITY, f64::max);
        bisect(lo, hi, |x| self.cdf(x) - p, 1e-10, 80)
    }
}

fn bisect(mut lo: f64, mut hi: f64, f: impl Fn(f64) -> f64, tol: f64, max_iter: usize) -> f64 {
    for _ in 0..max_iter {
        let mid = 0.5 * (lo + hi);
        let fm = f(mid);
        if fm.abs() < tol || (hi - lo) < tol {
            return mid;
        }
        if fm < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Abramowitz & Stegun 7.1.26 rational approximation to `erf`.
/// Absolute error ≤ 1.5e-7 — good enough for a shell (Gaussian tails).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Winitzki approximation to `erf⁻¹`; ~1e-4 accurate, refined by one Newton step.
fn inv_erf(x: f64) -> f64 {
    let clipped = x.clamp(-0.999_999_999, 0.999_999_999);
    let a = 0.147;
    let ln = (1.0 - clipped * clipped).ln();
    let inner = 2.0 / (PI * a) + 0.5 * ln;
    let mut y = clipped.signum() * (inner.mul_add(inner, -ln / a).sqrt() - inner).sqrt();
    // One Newton step against erf(y) − x = 0; derivative = 2/√π · e^{-y²}.
    let f = erf(y) - clipped;
    let df = 2.0 / PI.sqrt() * (-y * y).exp();
    if df.is_finite() && df > 0.0 {
        y -= f / df;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normal_pdf_cdf_quantile() {
        let g = Gaussian::new(0.0, 1.0);
        assert!((g.pdf(0.0) - 1.0 / SQRT_2PI).abs() < 1e-12);
        assert!((g.cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((g.quantile(0.5) - 0.0).abs() < 1e-6);
        assert!((g.quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-3);
    }

    #[test]
    fn logpdf_matches_pdf_ln() {
        let g = Gaussian::new(2.5, 0.7);
        for &y in &[-1.0, 0.0, 2.5, 4.0] {
            assert!((g.logpdf(y) - g.pdf(y).ln()).abs() < 1e-9);
        }
    }

    #[test]
    fn mixture_mean_is_weighted_sum() {
        let m = GaussianMixture::new([
            (0.3, Gaussian::new(1.0, 0.5)),
            (0.7, Gaussian::new(3.0, 0.5)),
        ]);
        let expected = 0.3 * 1.0 + 0.7 * 3.0;
        assert!((m.mean() - expected).abs() < 1e-12);
    }

    #[test]
    fn mixture_variance_includes_between_component_spread() {
        let m = GaussianMixture::new([
            (0.5, Gaussian::new(0.0, 1.0)),
            (0.5, Gaussian::new(4.0, 1.0)),
        ]);
        // Between = 0.5·(0-2)² + 0.5·(4-2)² = 4; within = 1 → total 5.
        assert!((m.variance() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn mixture_weights_normalise() {
        let m = GaussianMixture::new([
            (2.0, Gaussian::new(0.0, 1.0)),
            (3.0, Gaussian::new(0.0, 1.0)),
        ]);
        let sum: f64 = m.components.iter().map(|(w, _)| w).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mixture_quantile_monotone() {
        let m = GaussianMixture::new([
            (0.4, Gaussian::new(-2.0, 0.6)),
            (0.6, Gaussian::new(1.5, 1.2)),
        ]);
        let ps = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
        let mut prev = f64::NEG_INFINITY;
        for &p in &ps {
            let q = m.quantile(p);
            assert!(q > prev, "quantile not monotone at p={p}: {q} <= {prev}");
            prev = q;
        }
    }

    #[test]
    fn empty_mixture_reports_neg_inf_logpdf() {
        let m = GaussianMixture::new(std::iter::empty::<(f64, Gaussian)>());
        assert!(m.is_empty());
        assert_eq!(m.logpdf(0.0), f64::NEG_INFINITY);
    }
}
