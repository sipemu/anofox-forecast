//! Yeo-Johnson power transform.
//!
//! Generalization of Box-Cox to handle non-positive values:
//!
//! ```text
//! y(x; λ) = ((x+1)^λ - 1) / λ                if x ≥ 0, λ ≠ 0
//!         = log(x+1)                         if x ≥ 0, λ = 0
//!         = -((-x+1)^(2-λ) - 1) / (2-λ)      if x < 0, λ ≠ 2
//!         = -log(-x+1)                       if x < 0, λ = 2
//! ```
//!
//! Reference: Yeo, I.-K., & Johnson, R. A. (2000). *A new family of power
//! transformations to improve normality or symmetry.* Biometrika, 87(4),
//! 954-959.

use crate::error::{ForecastError, Result};

/// Apply Yeo-Johnson with a fixed lambda.
pub fn yeo_johnson(values: &[f64], lambda: f64) -> Vec<f64> {
    values.iter().map(|&x| yj_forward(x, lambda)).collect()
}

/// Inverse Yeo-Johnson transform with a fixed lambda.
pub fn inv_yeo_johnson(values: &[f64], lambda: f64) -> Vec<f64> {
    values.iter().map(|&y| yj_inverse(y, lambda)).collect()
}

/// Forward transform of a single scalar.
#[inline]
fn yj_forward(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if (lambda).abs() < 1e-12 {
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

/// Inverse transform of a single scalar.
#[inline]
fn yj_inverse(y: f64, lambda: f64) -> f64 {
    if y >= 0.0 {
        if lambda.abs() < 1e-12 {
            y.exp() - 1.0
        } else {
            (lambda * y + 1.0).powf(1.0 / lambda) - 1.0
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        1.0 - (-y).exp()
    } else {
        1.0 - (1.0 - (2.0 - lambda) * y).powf(1.0 / (2.0 - lambda))
    }
}

/// Maximum-likelihood estimate of the Yeo-Johnson lambda over `[-2, 2]`.
///
/// Uses a coarse grid search (Δ = 0.01) followed by a finer pass (Δ = 0.001)
/// around the best λ — same shape as `boxcox_lambda`. Returns `1.0` (no-op)
/// when the series is too short or all values are equal.
pub fn yeo_johnson_lambda(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }
    let mut best_lambda = 1.0;
    let mut best_llf = f64::NEG_INFINITY;
    for i in -200..=200 {
        let lambda = i as f64 / 100.0;
        let llf = yj_llf(values, lambda);
        if llf > best_llf {
            best_llf = llf;
            best_lambda = lambda;
        }
    }
    let start = (best_lambda - 0.1).max(-2.0);
    let end = (best_lambda + 0.1).min(2.0);
    for i in 0..=100 {
        let lambda = start + (end - start) * i as f64 / 100.0;
        let llf = yj_llf(values, lambda);
        if llf > best_llf {
            best_llf = llf;
            best_lambda = lambda;
        }
    }
    best_lambda
}

/// Yeo-Johnson concentrated log-likelihood.
///
/// `ℓ(λ) = -n/2 · log(σ²) + (λ-1) · Σ sign(x) · log(|x|+1)`
fn yj_llf(values: &[f64], lambda: f64) -> f64 {
    let n = values.len();
    let transformed = yeo_johnson(values, lambda);
    if transformed.iter().any(|x| !x.is_finite()) {
        return f64::NEG_INFINITY;
    }
    let mean = transformed.iter().sum::<f64>() / n as f64;
    let variance = transformed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if variance <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let signed_log_sum: f64 = values
        .iter()
        .map(|&x| x.signum() * (x.abs() + 1.0).ln())
        .sum();
    -0.5 * n as f64 * variance.ln() + (lambda - 1.0) * signed_log_sum
}

/// Convenience wrapper: auto-fit lambda then transform.
pub fn yeo_johnson_auto(values: &[f64]) -> Result<(Vec<f64>, f64)> {
    if values.is_empty() {
        return Err(ForecastError::EmptyData);
    }
    let lambda = yeo_johnson_lambda(values);
    Ok((yeo_johnson(values, lambda), lambda))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn forward_inverse_round_trip_positive() {
        let xs = vec![0.0, 0.5, 1.0, 2.5, 10.0];
        for &lambda in &[-1.5_f64, -0.5, 0.0, 0.7, 1.5, 2.0] {
            let y = yeo_johnson(&xs, lambda);
            let back = inv_yeo_johnson(&y, lambda);
            for (x, b) in xs.iter().zip(&back) {
                assert_relative_eq!(*x, *b, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn forward_inverse_round_trip_mixed() {
        // Negatives, zero, positives all in one go.
        let xs = vec![-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0];
        for &lambda in &[-1.0, -0.3, 0.0, 0.5, 1.2, 1.7, 2.0] {
            let y = yeo_johnson(&xs, lambda);
            let back = inv_yeo_johnson(&y, lambda);
            for (x, b) in xs.iter().zip(&back) {
                assert_relative_eq!(*x, *b, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn lambda_one_is_identity_shift() {
        // λ = 1: y = x for x ≥ 0; y = -((1-x)^1 - 1) = x for x < 0 → identity
        let xs = vec![-2.0, -0.5, 0.0, 1.0, 5.0];
        let y = yeo_johnson(&xs, 1.0);
        for (x, yi) in xs.iter().zip(&y) {
            assert_relative_eq!(*x, *yi, epsilon = 1e-10);
        }
    }

    #[test]
    fn lambda_zero_log1p_for_positives() {
        let xs = vec![0.0, 1.0, 2.0, 5.0];
        let y = yeo_johnson(&xs, 0.0);
        for (x, yi) in xs.iter().zip(&y) {
            assert_relative_eq!((x + 1.0_f64).ln(), *yi, epsilon = 1e-12);
        }
    }

    #[test]
    fn lambda_two_neg_log_for_negatives() {
        let xs = vec![-3.0_f64, -1.5, -0.5];
        let y = yeo_johnson(&xs, 2.0);
        for (x, yi) in xs.iter().zip(&y) {
            assert_relative_eq!(-(-x + 1.0).ln(), *yi, epsilon = 1e-12);
        }
    }

    #[test]
    fn auto_lambda_handles_zero_inclusive_series() {
        // Heavy right tail with zeros — Yeo-Johnson should pick λ < 1 to compress.
        let xs: Vec<f64> = (0..50)
            .map(|i| {
                if i % 5 == 0 {
                    0.0
                } else {
                    (i as f64).powf(2.0)
                }
            })
            .collect();
        let (transformed, lambda) = yeo_johnson_auto(&xs).unwrap();
        assert!(lambda < 1.0, "expected compressing lambda, got {}", lambda);
        // Round-trip recovers original
        let back = inv_yeo_johnson(&transformed, lambda);
        for (x, b) in xs.iter().zip(&back) {
            assert_relative_eq!(*x, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn auto_lambda_handles_negatives_only() {
        let xs: Vec<f64> = (1..30).map(|i| -(i as f64).sqrt()).collect();
        let (_transformed, lambda) = yeo_johnson_auto(&xs).unwrap();
        assert!(lambda.is_finite());
        assert!((-2.0..=2.0).contains(&lambda));
    }
}
