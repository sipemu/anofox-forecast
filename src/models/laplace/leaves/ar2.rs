//! AR(2) leaf with Yule-Walker coefficient estimation over EMA-based
//! autocovariance estimates.
//!
//! Fits `y_t − μ = φ₁·(y_{t-1} − μ) + φ₂·(y_{t-2} − μ) + ε`.
//!
//! Under stationarity the two Yule-Walker equations give
//!
//! ```text
//!     γ₁ = φ₁·γ₀ + φ₂·γ₁
//!     γ₂ = φ₁·γ₁ + φ₂·γ₀
//! ```
//!
//! with closed-form solutions
//!
//! ```text
//!     φ₁ = γ₁·(γ₀ − γ₂) / (γ₀² − γ₁²)
//!     φ₂ = (γ₀·γ₂ − γ₁²) / (γ₀² − γ₁²)
//! ```
//!
//! `γ₀`, `γ₁`, `γ₂` are tracked as EMAs of `y²`, `y·y_{t-1}`, `y·y_{t-2}`
//! (converted from raw to centered via `E[y²] − μ²` at query time).
//! This is more numerically robust than accumulating running centred
//! products because `μ` drifts over the fit.
//!
//! h-step forecast: recursive substitution into the AR(2) recursion.
//! Predictive std uses `σ · √h` — same convention as the other leaves.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ar2Leaf {
    alpha: f64, // EMA rate for all first- and second-moment estimates
    last: Option<f64>,
    last2: Option<f64>,
    e_y: f64,    // E[y]
    e_y2: f64,   // E[y²]
    e_y_y1: f64, // E[y_t · y_{t-1}]
    e_y_y2: f64, // E[y_t · y_{t-2}]
    phi1: f64,
    phi2: f64,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl Ar2Leaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            last: None,
            last2: None,
            e_y: 0.0,
            e_y2: 0.0,
            e_y_y1: 0.0,
            e_y_y2: 0.0,
            phi1: 0.0,
            phi2: 0.0,
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

    /// Solve for `(phi1, phi2)` via Yule-Walker on the current moments.
    /// Requires at least a rough estimate of `γ₀` — returns `false` if
    /// the system is degenerate.
    fn recompute_phis(&mut self) -> bool {
        let mu = self.e_y;
        let g0 = self.e_y2 - mu * mu;
        let g1 = self.e_y_y1 - mu * mu;
        let g2 = self.e_y_y2 - mu * mu;

        let det = g0 * g0 - g1 * g1;
        if g0 <= 1e-12 || det.abs() <= 1e-12 {
            return false;
        }
        let phi1 = g1 * (g0 - g2) / det;
        let phi2 = (g0 * g2 - g1 * g1) / det;
        let (phi1, phi2) = project_to_stationary(phi1, phi2);
        self.phi1 = phi1;
        self.phi2 = phi2;
        true
    }
}

/// Project `(φ₁, φ₂)` onto the AR(2) stationary triangle with a small
/// safety margin. The triangle is defined by
/// `|φ₂| < 1`, `φ₁ + φ₂ < 1`, `φ₂ − φ₁ < 1`. Without this projection, a
/// near-unit-root AR(2) (common on trending series where MoM autocovariance
/// estimates push `φ₁ + φ₂ → 1`) produces recursive h-step forecasts that
/// diverge exponentially — the M4 daily benchmark showed mean MAE of 787
/// vs. 158 for plain Laplace when the projection was absent.
fn project_to_stationary(phi1: f64, phi2: f64) -> (f64, f64) {
    const MARGIN: f64 = 0.02;
    let phi2 = phi2.clamp(-1.0 + MARGIN, 1.0 - MARGIN);
    let mut phi1 = phi1;
    let sum = phi1 + phi2;
    if sum > 1.0 - MARGIN {
        phi1 = 1.0 - MARGIN - phi2;
    }
    let diff = phi2 - phi1;
    if diff > 1.0 - MARGIN {
        phi1 = phi2 - (1.0 - MARGIN);
    }
    (phi1.clamp(-2.0 + MARGIN, 2.0 - MARGIN), phi2)
}

impl Leaf for Ar2Leaf {
    fn name(&self) -> &'static str {
        "ar2"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        // Multi-step forecast: mean is the recursive AR(2) projection,
        // variance is the correct MA(∞) form σ² · Σ_{i=0..h-1} ψ_i²
        // where the ψ coefficients satisfy the same AR(2) recurrence
        // as the mean: ψ_0 = 1, ψ_1 = φ_1, ψ_i = φ_1·ψ_{i-1} + φ_2·ψ_{i-2}.
        //
        // Pre-fix (skaters #157 in upstream): used σ · √h — the
        // random-walk formula. That assumes independent innovations
        // at each step, but for a *stationary* AR(2) the multi-step
        // variance approaches the unconditional bound σ²/(1-φ_1²-φ_2²)
        // (approx), not σ²·h. The pre-fix formula:
        //   - overstates h-step uncertainty for stationary AR(2),
        //     hurting probabilistic-metric calibration (WQL, CRPS);
        //   - is scale-independent of φ, so a near-white AR(2)
        //     (φ_1=φ_2=0) got the same variance as a strongly
        //     autocorrelated one — physically wrong.
        let mu = self.e_y;
        let last = self.last.unwrap_or(mu);
        let last2 = self.last2.unwrap_or(mu);
        let sigma = self.sigma();
        let mut y_prev2 = last2 - mu;
        let mut y_prev1 = last - mu;
        // Rolling MA(∞) coefficients ψ_i, initialised at ψ_0=1, ψ_{-1}=0.
        let mut psi_prev2: f64 = 0.0;
        let mut psi_prev1: f64 = 1.0;
        let mut var_scale: f64 = 1.0; // Σ ψ_i² accumulated so far = ψ_0² = 1
        (1..=horizon)
            .map(|h| {
                let y_h = self.phi1 * y_prev1 + self.phi2 * y_prev2;
                let mean = mu + y_h;
                y_prev2 = y_prev1;
                y_prev1 = y_h;
                let g = Gaussian::new(mean, sigma * var_scale.sqrt());
                // Roll the MA(∞) coefficient for the NEXT step.
                let psi_next = self.phi1 * psi_prev1 + self.phi2 * psi_prev2;
                psi_prev2 = psi_prev1;
                psi_prev1 = psi_next;
                var_scale += psi_next * psi_next;
                let _ = h;
                g
            })
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let mu = self.e_y;
        let last = self.last.unwrap_or(mu);
        let last2 = self.last2.unwrap_or(mu);
        let y_h = self.phi1 * (last - mu) + self.phi2 * (last2 - mu);
        Gaussian::new(mu + y_h, self.sigma())
    }

    fn observe(&mut self, y: f64) {
        // Prediction from the *pre-update* coefficients, for residual tracking.
        let mu_pre = if self.n == 0 { y } else { self.e_y };
        let last_val = self.last.unwrap_or(mu_pre);
        let last2_val = self.last2.unwrap_or(mu_pre);
        let predicted = mu_pre + self.phi1 * (last_val - mu_pre) + self.phi2 * (last2_val - mu_pre);
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        // EMA updates on all first- and second-order moments.
        let a = self.alpha;
        if self.n == 1 {
            self.e_y = y;
            self.e_y2 = y * y;
            // e_y_y1 and e_y_y2 are not yet definable — leave at 0.
        } else {
            self.e_y = a * y + (1.0 - a) * self.e_y;
            self.e_y2 = a * y * y + (1.0 - a) * self.e_y2;
            if let Some(y1) = self.last {
                self.e_y_y1 = a * y * y1 + (1.0 - a) * self.e_y_y1;
            }
            if let Some(y2) = self.last2 {
                self.e_y_y2 = a * y * y2 + (1.0 - a) * self.e_y_y2;
            }
        }

        self.recompute_phis();

        self.last2 = self.last;
        self.last = Some(y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple linear congruential PRNG so tests are deterministic without
    /// depending on the crate's own RNG feature.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_f64(&mut self) -> f64 {
            // Numerical Recipes LCG.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (self.0 >> 32) as u32;
            x as f64 / (u32::MAX as f64 + 1.0)
        }
        /// Approximate standard normal via central limit averaging.
        fn next_normal(&mut self) -> f64 {
            let mut s = 0.0;
            for _ in 0..12 {
                s += self.next_f64();
            }
            s - 6.0
        }
    }

    fn ar2_series(phi1: f64, phi2: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(n);
        let (mut y1, mut y2) = (0.0, 0.0);
        for _ in 0..n {
            let y = phi1 * y1 + phi2 * y2 + rng.next_normal();
            out.push(y);
            y2 = y1;
            y1 = y;
        }
        out
    }

    #[test]
    fn recovers_ar2_coefficients_within_tolerance() {
        let series = ar2_series(0.6, -0.25, 2000, 42);
        let mut leaf = Ar2Leaf::new(0.02);
        for y in series {
            leaf.observe(y);
        }
        assert!(
            (leaf.phi1 - 0.6).abs() < 0.10,
            "phi1 = {} (target 0.6)",
            leaf.phi1
        );
        assert!(
            (leaf.phi2 - (-0.25)).abs() < 0.10,
            "phi2 = {} (target -0.25)",
            leaf.phi2
        );
    }

    #[test]
    fn pure_ar1_series_gives_small_phi2() {
        let series = ar2_series(0.7, 0.0, 2000, 43);
        let mut leaf = Ar2Leaf::new(0.02);
        for y in series {
            leaf.observe(y);
        }
        assert!(
            (leaf.phi1 - 0.7).abs() < 0.10,
            "phi1 = {} (target 0.7)",
            leaf.phi1
        );
        assert!(leaf.phi2.abs() < 0.15, "phi2 = {} (target ~0)", leaf.phi2);
    }

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = Ar2Leaf::new(0.1);
        leaf.observe(3.0);
        leaf.observe(4.0);
        let preds = leaf.predict(5);
        for (h, p) in preds.iter().enumerate() {
            assert!(p.mean.is_finite(), "h={}: mean not finite", h + 1);
            assert!(p.std.is_finite() && p.std > 0.0, "h={}: std invalid", h + 1);
        }
    }
}
