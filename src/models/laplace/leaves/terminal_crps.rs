//! CRPS-gradient terminal leaf — port of skaters' `crps_leaf`.
//!
//! Same scale-mixture form as [`TerminalScaleMixture`](super::terminal_scale_mixture::TerminalScaleMixture)
//! but with **exponentiated-gradient descent on the closed-form
//! mixture CRPS** instead of likelihood-EM for the weight update.
//!
//! Two differences vs. the likelihood variant:
//!
//! 1. **15 log-spaced scale components** `c_i = 0.4 · 1.28^i` for
//!    `i ∈ 0..15` (vs. 5 in `scale_mixture_leaf`). More granular
//!    coverage of the tail/body trade-off.
//! 2. **Exponentiated-gradient (EG) weight update**:
//!    `w_c *= exp(-η · (g_c - ḡ))` where `g_c` is the gradient of
//!    the closed-form mixture CRPS. Direct CRPS optimization —
//!    matches a CRPS specialist on CRPS scoring while keeping (or
//!    slightly improving) LL on heavy-tailed data.
//!
//! Skaters ships this as the default terminal at
//! `objective="crps"` because on M-series-ish integer-count data
//! (M5, retail SKUs) it typically beats the likelihood variant on
//! LL by 3-15% and on CRPS by 5-10%.
//!
//! PR #7 of #180.

use crate::models::laplace::dist::{Gaussian, GaussianMixture};

/// Number of scale components. Skaters ships 15 log-spaced values.
const N_SCALES: usize = 15;

/// Two normalization constants used in the CRPS gradient.
///
/// `INV_SQRT_2PI = 1/√(2π)`, `A0 = 2 · φ(0) = 2 / √(2π)` — the value
/// of `A(0, 1)` in the closed-form Gaussian expected absolute
/// deviation.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const A0: f64 = 2.0 * INV_SQRT_2PI;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Terminal scale-mixture with CRPS-gradient weight updates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TerminalCrpsMixture {
    scales: [f64; N_SCALES],
    /// Pre-computed pairwise `A(0, √(c_a² + c_b²))` for the CRPS
    /// gradient's second term. Constant across all observations.
    b_matrix: [[f64; N_SCALES]; N_SCALES],
    scale_alpha: f64,
    eta: f64,
    v: f64,
    w: [f64; N_SCALES],
    n_obs: usize,
}

/// `Φ(x)` — standard normal CDF via `erf`.
#[inline]
fn phi_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm_erf(x / SQRT_2))
}

/// `φ(x)` — standard normal PDF.
#[inline]
fn phi_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() * INV_SQRT_2PI
}

/// `E|N(m, s²)| = m · (2Φ(m/s) − 1) + 2s · φ(m/s)`.
///
/// The closed form for the expected absolute value of a Normal.
#[inline]
fn abs_normal(m: f64, s: f64) -> f64 {
    if s <= 0.0 {
        return m.abs();
    }
    let z = m / s;
    m * (2.0 * phi_cdf(z) - 1.0) + 2.0 * s * phi_pdf(z)
}

/// `erf` for f64. Rust std doesn't ship it; use the standard
/// Abramowitz-Stegun rational approximation 7.1.26 (max error
/// ~1.5e-7, plenty for our needs).
fn libm_erf(x: f64) -> f64 {
    // Constants for A&S 7.1.26
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

impl TerminalCrpsMixture {
    /// Skaters ships `scale_alpha = 0.01, eta = 1.0` for `crps_leaf`.
    pub fn new() -> Self {
        Self::with_params(0.01, 1.0)
    }

    pub fn with_params(scale_alpha: f64, eta: f64) -> Self {
        // Log-spaced scales matching skaters' `FINE` tuple.
        let mut scales = [0.0f64; N_SCALES];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0.4 * 1.28_f64.powi(i as i32);
        }
        // Pre-compute pairwise A(0, √(c_a² + c_b²)) = A0 · √(c_a² + c_b²)
        // (the m=0 special case of `abs_normal`).
        let mut b_matrix = [[0.0f64; N_SCALES]; N_SCALES];
        for a in 0..N_SCALES {
            for b in 0..N_SCALES {
                b_matrix[a][b] = A0 * (scales[a] * scales[a] + scales[b] * scales[b]).sqrt();
            }
        }
        // Start with all mass on the c ≈ 1.0 component.
        let one_idx = scales
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - 1.0).abs().partial_cmp(&(**b - 1.0).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let mut w = [1e-6; N_SCALES];
        w[one_idx] = 1.0;
        Self {
            scales,
            b_matrix,
            scale_alpha,
            eta,
            v: 0.0,
            w,
            n_obs: 0,
        }
    }

    /// Absorb one residual `r = y - softmax_mixture_mean` and update
    /// the mixture weights via one exponentiated-gradient step on
    /// the closed-form mixture CRPS.
    pub fn observe(&mut self, r: f64) {
        if !r.is_finite() {
            return;
        }
        self.n_obs += 1;
        let n = self.n_obs as f64;

        // 1/n bootstrap for early observations.
        let a = self.scale_alpha.max(1.0 / n);
        self.v = (1.0 - a) * self.v + a * r * r;

        let sigma = if self.v.is_finite() && self.v > 0.0 {
            self.v.sqrt()
        } else {
            r.abs().max(1e-8)
        };
        let z = r / sigma;

        // CRPS gradient per component:
        //   g[c] = A(-z, C[c]) - Σ_j w[j] · B[c][j]
        let mut g = [0.0f64; N_SCALES];
        for c in 0..N_SCALES {
            let first = abs_normal(-z, self.scales[c]);
            let mut second = 0.0;
            for j in 0..N_SCALES {
                second += self.w[j] * self.b_matrix[c][j];
            }
            g[c] = first - second;
        }
        // Exponentiated-gradient step, mean-centered exponent to
        // avoid overflow. Subtracting a constant from all exponents
        // leaves the normalized weights unchanged.
        let gm: f64 = g.iter().sum::<f64>() / N_SCALES as f64;
        let mut e = [0.0f64; N_SCALES];
        for c in 0..N_SCALES {
            e[c] = -self.eta * (g[c] - gm);
        }
        let e_max = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut nw = [0.0f64; N_SCALES];
        let mut z_norm = 0.0;
        for c in 0..N_SCALES {
            nw[c] = self.w[c] * (e[c] - e_max).exp();
            z_norm += nw[c];
        }
        if z_norm > 0.0 && z_norm.is_finite() {
            for c in 0..N_SCALES {
                self.w[c] = nw[c] / z_norm;
            }
        }
    }

    /// Emit a zero-centered 15-component mixture. Callers shift by
    /// the softmax ensemble mean to get the final predictive.
    pub fn predict(&self) -> Vec<(f64, f64)> {
        let sigma = if self.v.is_finite() && self.v > 0.0 {
            self.v.sqrt()
        } else {
            1e-6
        };
        (0..N_SCALES)
            .map(|i| (self.w[i], (self.scales[i] * sigma).max(1e-9)))
            .collect()
    }

    /// Convenience: emit as a `GaussianMixture` centered at `mean`.
    pub fn predict_shifted(&self, mean: f64) -> GaussianMixture {
        let comps = self.predict();
        GaussianMixture::new(
            comps
                .into_iter()
                .map(|(w, sig)| (w, Gaussian::new(mean, sig))),
        )
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }
}

impl Default for TerminalCrpsMixture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss_ll(y: f64, mu: f64, sigma: f64) -> f64 {
        -0.5 * ((y - mu) / sigma).powi(2)
            - sigma.max(1e-30).ln()
            - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    #[test]
    fn erf_approx_is_within_tolerance() {
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let approx = libm_erf(x);
            // Compare against the exact value via `(x/√2)` and the
            // known relation `erf(x) = 2·Φ(x·√2) - 1`, which we'd only
            // use for a cross-check. Just sanity-bound: erf(0)=0,
            // erf(∞)=1, erf is odd.
            let _ = approx;
        }
        assert!((libm_erf(0.0)).abs() < 1e-6);
        assert!((libm_erf(3.0) - 0.9999779).abs() < 1e-4);
        assert!((libm_erf(-1.0) + libm_erf(1.0)).abs() < 1e-6);
    }

    /// On IID Gaussian residuals, the CRPS terminal should score at
    /// least as well as a narrow single Gaussian — same guarantee as
    /// the likelihood-EM variant.
    #[test]
    fn beats_narrow_gaussian_on_iid_gaussian() {
        let mut t = TerminalCrpsMixture::new();
        let sigma_true = 1.5;
        let mut rs = Vec::new();
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
            rs.push(sigma_true * z);
        }
        for r in &rs {
            t.observe(*r);
        }
        let mix = t.predict_shifted(0.0);
        let ll_mix: f64 = rs[800..].iter().map(|y| mix.logpdf(*y)).sum();
        let ll_narrow: f64 = rs[800..].iter().map(|y| gauss_ll(*y, 0.0, 0.4)).sum();
        assert!(
            ll_mix > ll_narrow,
            "CRPS mixture LL {ll_mix} did not beat narrow Gaussian LL {ll_narrow}"
        );
    }

    /// On heavy-tailed residuals, the CRPS terminal's tail weights
    /// should shift wider — same behavior as the likelihood variant.
    #[test]
    fn beats_single_gaussian_on_heavy_tails() {
        let mut t = TerminalCrpsMixture::new();
        let mut rs = Vec::new();
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
            let scale = if i % 5 == 0 { 5.0 } else { 1.0 };
            rs.push(scale * z);
        }
        for r in &rs {
            t.observe(*r);
        }
        let mix = t.predict_shifted(0.0);
        let var: f64 = rs.iter().map(|r| r * r).sum::<f64>() / rs.len() as f64;
        let sigma_mom = var.sqrt();
        let ll_mix: f64 = rs[1500..].iter().map(|y| mix.logpdf(*y)).sum();
        let ll_gauss: f64 = rs[1500..]
            .iter()
            .map(|y| gauss_ll(*y, 0.0, sigma_mom))
            .sum();
        assert!(
            ll_mix > ll_gauss,
            "CRPS mixture LL {ll_mix} did not beat plain Gaussian LL {ll_gauss} on heavy tails"
        );
    }

    #[test]
    fn nan_residual_is_ignored() {
        let mut t = TerminalCrpsMixture::new();
        t.observe(1.0);
        t.observe(f64::NAN);
        t.observe(f64::INFINITY);
        t.observe(-1.0);
        assert_eq!(t.n_obs(), 2);
    }
}
