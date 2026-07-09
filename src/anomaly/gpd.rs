//! Generalized Pareto Distribution.
//!
//! Two functions ported from `microprediction/timemachines/heads/mahalanobis.py`:
//!
//! - [`gpd_fit_pwm`] — Probability-Weighted-Moments fit (Hosking &
//!   Wallis 1987). Not method-of-moments: MOM needs finite variance
//!   (shape < 1/2) and on the heavy tails d² actually produces
//!   (Hill estimates ~0.7) MOM understates the shape by half. PWM is
//!   valid for shape < 1.
//! - [`gpd_sf`] — GPD survival function at excess `e ≥ 0`.
//!
//! Used by the Mahalanobis wrapper's POT (peaks-over-threshold) tail:
//! d² excesses over the empirical-null quantile fit a GPD; deep-tail
//! p-values come from `p = zeta · gpd_sf(d² − t_pot)`.

/// Fit a Generalized Pareto Distribution to a list of exceedances by
/// probability-weighted moments (Hosking & Wallis 1987).
///
/// Returns `(gamma, sigma)` where `gamma` is the shape and `sigma` is
/// the scale. `gamma` is clamped to `[-0.5, 0.95]` for stability;
/// `sigma` is floored at `1e-12`.
///
/// # Panics
/// Debug-panics if `excesses.len() < 2` — the caller must gate this.
pub fn gpd_fit_pwm(excesses: &[f64]) -> (f64, f64) {
    debug_assert!(
        excesses.len() >= 2,
        "gpd_fit_pwm requires >= 2 excesses (divide-by-zero otherwise)"
    );
    let n = excesses.len();
    let n_f = n as f64;
    let mut x: Vec<f64> = excesses.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let a0: f64 = x.iter().sum::<f64>() / n_f;
    let mut a1: f64 = 0.0;
    for i in 0..n {
        a1 += ((n as f64 - 1.0 - i as f64) / (n_f - 1.0)) * x[i];
    }
    a1 /= n_f;
    let denom = a0 - 2.0 * a1;
    if denom <= 1e-12 {
        // Heavier than shape ~1: cap at the clamp.
        return (0.95, a0.max(1e-12));
    }
    let gamma_raw = 2.0 - a0 / denom;
    let gamma = gamma_raw.clamp(-0.5, 0.95);
    let sigma = (2.0 * a0 * a1 / denom).max(1e-12);
    (gamma, sigma)
}

/// GPD survival function at excess `e ≥ 0`.
///
/// - `gamma == 0` (approximately): degenerates to the exponential
///   distribution, `exp(-e/σ)`.
/// - `gamma != 0`: `(1 + γ·e/σ)^(-1/γ)`. Returns 0 beyond the finite
///   support when `γ < 0`.
pub fn gpd_sf(e: f64, gamma: f64, sigma: f64) -> f64 {
    if gamma.abs() < 1e-9 {
        return (-(e / sigma).min(700.0)).exp();
    }
    let arg = 1.0 + gamma * e / sigma;
    if arg <= 0.0 {
        return 0.0;
    }
    arg.powf(-1.0 / gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_at_gamma_zero() {
        // gpd_sf(e, γ=0, σ=1) should equal exp(-e)
        for &e in &[0.5, 1.0, 2.5, 10.0] {
            let sf = gpd_sf(e, 0.0, 1.0);
            let expected = (-e).exp();
            assert!(
                (sf - expected).abs() < 1e-12,
                "e={e}: got {sf}, expected {expected}"
            );
        }
    }

    #[test]
    fn heavy_tail_gamma_positive() {
        // With γ > 0 the tail is heavier than exponential
        let e = 5.0;
        let sf_exp = gpd_sf(e, 0.0, 1.0);
        let sf_heavy = gpd_sf(e, 0.5, 1.0);
        assert!(sf_heavy > sf_exp, "heavy tail should exceed exponential");
    }

    #[test]
    fn short_tail_gamma_negative() {
        // With γ < 0 the tail is lighter than exponential (and has
        // finite support).
        let e = 3.0;
        let sf_short = gpd_sf(e, -0.3, 1.0);
        let sf_exp = gpd_sf(e, 0.0, 1.0);
        assert!(sf_short < sf_exp);
        // Beyond finite support σ/|γ| = 1/0.3 ≈ 3.33: SF must be 0.
        let sf_beyond = gpd_sf(10.0, -0.3, 1.0);
        assert_eq!(sf_beyond, 0.0);
    }

    #[test]
    fn pwm_recovers_known_shape() {
        // Generate GPD(γ=0.3, σ=1) samples via inverse CDF and check the
        // PWM estimator recovers γ approximately. Deterministic seed
        // via LCG so the test is reproducible.
        let true_gamma = 0.3;
        let true_sigma = 1.0;
        let n = 1000;
        let mut seed = 12345u64;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (seed >> 33) as f64 / (1u64 << 31) as f64;
            let u = u.clamp(1e-12, 1.0 - 1e-12);
            // GPD inverse CDF at level u: σ/γ · ((1-u)^(-γ) - 1)
            let x = true_sigma / true_gamma * ((1.0 - u).powf(-true_gamma) - 1.0);
            samples.push(x);
        }
        let (gamma_hat, sigma_hat) = gpd_fit_pwm(&samples);
        assert!(
            (gamma_hat - true_gamma).abs() < 0.15,
            "γ estimate {gamma_hat} vs true {true_gamma}",
        );
        assert!(
            (sigma_hat - true_sigma).abs() < 0.3,
            "σ estimate {sigma_hat} vs true {true_sigma}",
        );
    }

    #[test]
    fn pwm_clamps_gamma() {
        // Constant excesses → denom ≈ 0 → clamp branch fires.
        let samples = vec![1.0; 20];
        let (gamma, sigma) = gpd_fit_pwm(&samples);
        assert!(gamma <= 0.95 && gamma >= -0.5);
        assert!(sigma > 0.0);
    }
}
