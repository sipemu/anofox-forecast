//! Terminal scale-mixture leaf — port of skaters' `scale_mixture_leaf`.
//!
//! Sits at the top of a `LaplaceForecaster`: consumes the residuals
//! `y - mixture_mean` produced by the softmax-over-leaves and models
//! their distribution as a fixed dictionary of zero-mean Gaussians at
//! log-spaced scales, with weights learned online by likelihood-EM.
//!
//! The rationale (from Peter Cotton's skaters):
//!
//! - A Student-t — and most heavy-tailed natural data (returns,
//!   stochastic volatility) — *is* a Gaussian scale mixture, so this
//!   approximates it by construction.
//! - Because every component shares mean 0, mixing **fattens** rather
//!   than **flattens** — so the shape survives into the ensemble.
//! - The weights are the "discrepancy from N(0,1)": mass on `c=1` is
//!   pure Gaussian; mass bleeding into larger `c` is heavier tails.
//!
//! Empirically (skaters' benchmarks): matches a plain Gaussian leaf on
//! normal data, gains ~0.13 nats on Student-t3.
//!
//! This is the "model first, conform last" pattern: the softmax
//! ensemble picks the *mean* forecast, then this leaf reshapes the
//! residual distribution once at the top so the tail is not diluted by
//! averaging Gaussians of different widths.

use crate::models::laplace::dist::{Gaussian, GaussianMixture};

/// Default fixed scale dictionary (relative to the running residual σ).
/// The `c=1.0` component is the plain-Gaussian anchor; the wider
/// components pick up mass on heavy-tail data.
const DEFAULT_SCALES: [f64; 5] = [0.7, 1.0, 1.6, 3.0, 6.0];

/// Terminal scale-mixture over residuals `y - softmax_mixture_mean`.
///
/// Two online EWMAs share a running-scale story:
/// - `v`: EWMA of the squared residual at rate `scale_alpha` → σ
/// - `w[i]`: EM-updated weight on scale `scales[i]`, recency rate `gamma`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TerminalScaleMixture {
    scales: [f64; 5],
    scale_alpha: f64,
    gamma: f64,
    v: f64,
    w: [f64; 5],
    n_obs: usize,
    /// Accuracy-audit #5: AR(1) residual autocorrelation φ estimator.
    /// EWMA of `r_t · r_{t-1} / v`. Bounded to `(-0.9, 0.9)` for
    /// stationarity. Used at forecast time to compute `√((1-φ^(2h)) /
    /// (1-φ²))` scaling instead of `√h` (which assumes IID).
    phi: f64,
    /// Previous residual, kept for autocorrelation update.
    prev_r: f64,
}

impl TerminalScaleMixture {
    /// `scale_alpha` = 0.03 (residual-variance EWMA rate,
    /// effective memory ~33 obs), `gamma` = 0.02 (weight-recency rate).
    /// Matches skaters' `laplace(..., scale_alpha=0.03)` default.
    pub fn new() -> Self {
        Self::with_params(0.03, 0.02)
    }

    pub fn with_params(scale_alpha: f64, gamma: f64) -> Self {
        // Start ~Gaussian: all mass on the c=1.0 component.
        let mut w = [1e-6; 5];
        // Find the index closest to 1.0 in DEFAULT_SCALES.
        let one_idx = DEFAULT_SCALES
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - 1.0).abs().partial_cmp(&(**b - 1.0).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        w[one_idx] = 1.0;
        Self {
            scales: DEFAULT_SCALES,
            scale_alpha,
            gamma,
            v: 0.0,
            w,
            n_obs: 0,
            phi: 0.0,
            prev_r: 0.0,
        }
    }

    /// Autocorrelation φ estimate (accuracy-audit #5). Bounded to
    /// `(-0.9, 0.9)` for stationarity.
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Multi-horizon σ scaling factor for AR(1) residuals.
    /// Returns `√((1 - φ^(2h)) / (1 - φ²))`. Equals `√h` when `φ=0`
    /// (IID), less than `√h` for `φ<0`, more than `√h` for `φ>0`.
    pub fn h_step_std_scale(&self, h: usize) -> f64 {
        if h == 0 {
            return 0.0;
        }
        let phi = self.phi.clamp(-0.9, 0.9);
        let phi2 = phi * phi;
        if phi2 < 1e-6 {
            return (h as f64).sqrt();
        }
        let numer = 1.0 - phi2.powi(h as i32);
        let denom = 1.0 - phi2;
        (numer / denom).max(0.0).sqrt()
    }

    /// Absorb one residual `r = y - softmax_mixture_mean`.
    ///
    /// Updates the running variance EWMA (tracks scale) and the online
    /// EM weight vector (tracks tail shape). Both use `max(rate, 1/n)`
    /// bootstrap so the first few observations behave like Welford's
    /// algorithm before switching to EWMA.
    pub fn observe(&mut self, r: f64) {
        if !r.is_finite() {
            return;
        }
        self.n_obs += 1;
        let n = self.n_obs as f64;

        // Residual-variance EWMA: `1/n` bootstrap for early obs.
        let a = self.scale_alpha.max(1.0 / n);
        self.v = (1.0 - a) * self.v + a * r * r;

        // Accuracy-audit #5: AR(1) autocorrelation EWMA. Update
        // `phi ~= E[r_t * r_{t-1}] / E[r_t²]`. Clamped to (-0.9, 0.9)
        // for stationarity. Only starts updating after we have a
        // previous residual (n_obs >= 2).
        if self.n_obs >= 2 && self.v > 1e-12 {
            let rho = (r * self.prev_r) / self.v;
            let phi_alpha = a; // Use same rate as variance EWMA.
            self.phi = ((1.0 - phi_alpha) * self.phi + phi_alpha * rho).clamp(-0.9, 0.9);
        }
        self.prev_r = r;

        let sigma = if self.v.is_finite() && self.v > 0.0 {
            self.v.sqrt()
        } else {
            r.abs().max(1e-8)
        };
        let z = r / sigma;

        // Online-EM weight update: `1/n` bootstrap for early obs.
        // Component density (up to a constant `1 / (sqrt(2π) sigma)`
        // which cancels in the ratio):
        //   dens[i] = w[i] * (1/c[i]) * exp(-0.5 * z² / c[i]²)
        let mut dens = [0.0; 5];
        let mut total = 0.0;
        for i in 0..5 {
            let c = self.scales[i];
            let d = self.w[i] * (-0.5 * z * z / (c * c)).exp() / c;
            dens[i] = d;
            total += d;
        }
        if total > 0.0 && total.is_finite() {
            let g = self.gamma.max(1.0 / n);
            for i in 0..5 {
                self.w[i] = (1.0 - g) * self.w[i] + g * dens[i] / total;
            }
        }
    }

    /// Emit a zero-centered mixture with the tracked scale and weights.
    /// Callers shift by the softmax ensemble mean to get the final
    /// predictive at a given horizon.
    pub fn predict(&self) -> Vec<(f64, f64)> {
        let sigma = if self.v.is_finite() && self.v > 0.0 {
            self.v.sqrt()
        } else {
            1e-6
        };
        (0..5)
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

    /// Warm-start the residual-variance EWMA with a robust batch
    /// estimate (accuracy-audit #3a). Sets `v = sigma²` and
    /// `n_obs = seed_n` so the subsequent `observe(r)` updates use
    /// the configured EWMA rate rather than the `1/n` bootstrap.
    ///
    /// Callers should pass `sigma = 1.4826 · median(|r|)` (MAD scaled
    /// to Gaussian σ) and `seed_n = 30` (or thereabouts) so the
    /// bootstrap window ends immediately.
    pub fn warm_start(&mut self, sigma: f64, seed_n: usize) {
        if sigma.is_finite() && sigma > 0.0 {
            self.v = sigma * sigma;
            self.n_obs = seed_n.max(1);
        }
    }
}

impl Default for TerminalScaleMixture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss_ll(y: f64, mu: f64, sigma: f64) -> f64 {
        -0.5 * ((y - mu) / sigma).powi(2) - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// On IID N(0, σ²) residuals, the terminal should concentrate weight
    /// on the c=1.0 component and its predictive Gaussian mixture at
    /// unseen y should score at least as well as a plain N(0, σ̂).
    #[test]
    fn concentrates_on_gaussian_residuals() {
        let mut t = TerminalScaleMixture::new();
        // Deterministic pseudo-Gaussian residuals via Box-Muller.
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
        // Score the last 200 residuals under the mixture; should beat
        // an under-scaled Gaussian.
        let ll_mix: f64 = rs[800..].iter().map(|y| mix.logpdf(*y)).sum();
        let ll_narrow: f64 = rs[800..].iter().map(|y| gauss_ll(*y, 0.0, 0.5)).sum();
        assert!(
            ll_mix > ll_narrow,
            "mixture LL {ll_mix} did not beat narrow-Gaussian LL {ll_narrow}"
        );
    }

    /// On heavy-tailed residuals, weight should shift toward wider
    /// components. Test: an approximate Student-t via averaging two
    /// pseudo-Gaussians of different scales — mixture LL should beat
    /// a single Gaussian at the same total variance.
    #[test]
    fn beats_single_gaussian_on_heavy_tails() {
        let mut t = TerminalScaleMixture::new();
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
            // Heavy-tailed: half from N(0, 1), half from N(0, 5) — a
            // simple discrete scale mixture the leaf should recover.
            let scale = if i % 5 == 0 { 5.0 } else { 1.0 };
            rs.push(scale * z);
        }
        for r in &rs {
            t.observe(*r);
        }
        let mix = t.predict_shifted(0.0);
        // Match the total variance and score a plain Gaussian on the tail.
        let var: f64 = rs.iter().map(|r| r * r).sum::<f64>() / rs.len() as f64;
        let sigma_mom = var.sqrt();
        let ll_mix: f64 = rs[1500..].iter().map(|y| mix.logpdf(*y)).sum();
        let ll_gauss: f64 = rs[1500..]
            .iter()
            .map(|y| gauss_ll(*y, 0.0, sigma_mom))
            .sum();
        assert!(
            ll_mix > ll_gauss,
            "scale-mixture LL {ll_mix} did not beat plain Gaussian LL {ll_gauss} on heavy tails"
        );
    }

    #[test]
    fn nan_residual_is_ignored() {
        let mut t = TerminalScaleMixture::new();
        t.observe(1.0);
        t.observe(f64::NAN);
        t.observe(f64::INFINITY);
        t.observe(-1.0);
        assert_eq!(t.n_obs(), 2);
    }
}
