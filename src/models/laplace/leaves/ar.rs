//! AR(1) leaf with online mean & OLS-updated φ.
//!
//! Fits `y_t − μ = φ · (y_{t−1} − μ) + ε`. Point forecast at horizon `h`:
//! `μ + φ^h · (y_t − μ)`. Predictive std at horizon `h`:
//! `σ · √(Σ_{i=0..h} φ^{2i})`.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ar1Leaf {
    alpha_mean: f64,
    mean: Option<f64>,
    last: Option<f64>,
    // Sufficient stats on centred products for online OLS on φ.
    s_xx: f64, // Σ (y_{t−1} − μ)²
    s_xy: f64, // Σ (y_{t−1} − μ)(y_t − μ)
    phi: f64,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl Ar1Leaf {
    pub fn new(alpha_mean: f64) -> Self {
        Self {
            alpha_mean: alpha_mean.clamp(1e-3, 1.0 - 1e-3),
            mean: None,
            last: None,
            s_xx: 0.0,
            s_xy: 0.0,
            phi: 0.0,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
        }
    }

    fn sigma(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).sqrt().max(1e-9)
    }
}

impl Leaf for Ar1Leaf {
    fn name(&self) -> &'static str {
        "ar1"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let mu = self.mean.unwrap_or(0.0);
        let last = self.last.unwrap_or(mu);
        let sigma = self.sigma();
        let phi = self.phi.clamp(-0.999, 0.999);
        (1..=horizon)
            .map(|h| {
                let phi_h = phi.powi(h as i32);
                let mean = mu + phi_h * (last - mu);
                // Var = σ² · Σ_{i=0..h−1} φ^{2i}
                let phi2 = phi * phi;
                let var_scale = if (1.0 - phi2).abs() < 1e-12 {
                    h as f64
                } else {
                    (1.0 - phi2.powi(h as i32)) / (1.0 - phi2)
                };
                Gaussian::new(mean, sigma * var_scale.sqrt())
            })
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let mu = self.mean.unwrap_or(0.0);
        let last = self.last.unwrap_or(mu);
        let phi = self.phi.clamp(-0.999, 0.999);
        Gaussian::new(mu + phi * (last - mu), self.sigma())
    }

    fn observe(&mut self, y: f64) {
        let mu_before = self.mean.unwrap_or(y);
        let last = self.last.unwrap_or(mu_before);

        let predicted = mu_before + self.phi * (last - mu_before);
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        // Sufficient stats update — use the pre-update mean so both sides are
        // centred with the same μ (small bias vs. true centred OLS but stable
        // in a streaming setting).
        let x = last - mu_before;
        let z = y - mu_before;
        self.s_xx += x * x;
        self.s_xy += x * z;
        if self.s_xx > 1e-12 {
            self.phi = (self.s_xy / self.s_xx).clamp(-0.999, 0.999);
        }

        self.mean = Some(match self.mean {
            Some(m) => self.alpha_mean * y + (1.0 - self.alpha_mean) * m,
            None => y,
        });
        self.last = Some(y);
    }
}
