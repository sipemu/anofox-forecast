//! Box-Cox power transformation.
//!
//! Transforms data to be more normally distributed.

/// Result of Box-Cox transformation.
#[derive(Debug, Clone)]
pub struct BoxCoxResult {
    /// Transformed data
    pub data: Vec<f64>,
    /// Lambda parameter used
    pub lambda: f64,
}

impl BoxCoxResult {
    /// Inverse transform to recover original scale.
    pub fn inverse(&self) -> Vec<f64> {
        inv_boxcox(&self.data, self.lambda)
    }
}

/// Apply Box-Cox transformation with a given lambda.
///
/// For lambda != 0: y = (x^lambda - 1) / lambda
/// For lambda == 0: y = ln(x)
///
/// # Arguments
/// * `series` - Input data (must be positive)
/// * `lambda` - Transformation parameter
///
/// # Panics
/// Does not panic, but returns NaN for non-positive values.
pub fn boxcox(series: &[f64], lambda: f64) -> Vec<f64> {
    series
        .iter()
        .map(|&x| {
            if x <= 0.0 {
                f64::NAN
            } else if lambda.abs() < 1e-10 {
                x.ln()
            } else {
                (x.powf(lambda) - 1.0) / lambda
            }
        })
        .collect()
}

/// Apply Box-Cox transformation with automatic lambda selection.
///
/// Uses maximum likelihood estimation to find optimal lambda.
pub fn boxcox_auto(series: &[f64]) -> BoxCoxResult {
    let lambda = boxcox_lambda(series);
    let data = boxcox(series, lambda);
    BoxCoxResult { data, lambda }
}

/// Inverse Box-Cox transformation.
///
/// For lambda != 0: x = (lambda * y + 1)^(1/lambda)
/// For lambda == 0: x = exp(y)
pub fn inv_boxcox(transformed: &[f64], lambda: f64) -> Vec<f64> {
    transformed
        .iter()
        .map(|&y| {
            if lambda.abs() < 1e-10 {
                y.exp()
            } else {
                let val = lambda * y + 1.0;
                if val <= 0.0 {
                    f64::NAN
                } else {
                    val.powf(1.0 / lambda)
                }
            }
        })
        .collect()
}

/// Find optimal Box-Cox lambda using maximum likelihood estimation.
///
/// Searches over a range of lambda values to minimize the negative
/// log-likelihood of the transformed data being normally distributed.
pub fn boxcox_lambda(series: &[f64]) -> f64 {
    // Filter to positive values only
    let positive: Vec<f64> = series.iter().copied().filter(|&x| x > 0.0).collect();

    if positive.is_empty() {
        return 1.0; // Default
    }

    // Search over lambda range [-2, 2]
    let mut best_lambda = 1.0;
    let mut best_llf = f64::NEG_INFINITY;

    for i in -200..=200 {
        let lambda = i as f64 / 100.0;
        let llf = boxcox_llf(&positive, lambda);

        if llf > best_llf {
            best_llf = llf;
            best_lambda = lambda;
        }
    }

    // Refine with finer search around best
    let start = (best_lambda - 0.1).max(-2.0);
    let end = (best_lambda + 0.1).min(2.0);

    for i in 0..=100 {
        let lambda = start + (end - start) * i as f64 / 100.0;
        let llf = boxcox_llf(&positive, lambda);

        if llf > best_llf {
            best_llf = llf;
            best_lambda = lambda;
        }
    }

    best_lambda
}

/// Compute log-likelihood for Box-Cox transformation.
fn boxcox_llf(series: &[f64], lambda: f64) -> f64 {
    let n = series.len();
    if n < 2 {
        return f64::NEG_INFINITY;
    }

    let transformed = boxcox(series, lambda);

    // Check for NaN values
    if transformed.iter().any(|x| x.is_nan()) {
        return f64::NEG_INFINITY;
    }

    let mean = transformed.iter().sum::<f64>() / n as f64;
    let variance = transformed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Log-likelihood (ignoring constant terms)
    // llf = -n/2 * ln(variance) + (lambda - 1) * sum(ln(x))
    let log_sum: f64 = series.iter().map(|x| x.ln()).sum();

    -0.5 * n as f64 * variance.ln() + (lambda - 1.0) * log_sum
}

/// Check if data is suitable for Box-Cox transformation.
///
/// Returns true if all values are positive.
pub fn is_boxcox_suitable(series: &[f64]) -> bool {
    !series.is_empty() && series.iter().all(|&x| x > 0.0)
}

/// Apply shifted Box-Cox for data with non-positive values.
///
/// Adds a shift to make all values positive before transformation.
pub fn boxcox_shifted(series: &[f64], lambda: f64) -> BoxCoxResult {
    let min_val = series.iter().copied().fold(f64::INFINITY, f64::min);
    let shift = if min_val <= 0.0 { -min_val + 1.0 } else { 0.0 };

    let shifted: Vec<f64> = series.iter().map(|&x| x + shift).collect();
    let data = boxcox(&shifted, lambda);

    BoxCoxResult { data, lambda }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== boxcox ====================

    #[test]
    fn boxcox_lambda_1() {
        // Lambda = 1: y = x - 1
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = boxcox(&series, 1.0);

        for (i, &x) in series.iter().enumerate() {
            assert_relative_eq!(result[i], x - 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn boxcox_lambda_0() {
        // Lambda = 0: y = ln(x)
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = boxcox(&series, 0.0);

        for (i, &x) in series.iter().enumerate() {
            assert_relative_eq!(result[i], x.ln(), epsilon = 1e-10);
        }
    }

    #[test]
    fn boxcox_lambda_2() {
        // Lambda = 2: y = (x^2 - 1) / 2
        let series = vec![1.0, 2.0, 3.0];
        let result = boxcox(&series, 2.0);

        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10); // (1-1)/2
        assert_relative_eq!(result[1], 1.5, epsilon = 1e-10); // (4-1)/2
        assert_relative_eq!(result[2], 4.0, epsilon = 1e-10); // (9-1)/2
    }

    #[test]
    fn boxcox_negative_values() {
        let series = vec![-1.0, 0.0, 1.0, 2.0];
        let result = boxcox(&series, 1.0);

        // Non-positive values should return NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(!result[2].is_nan());
        assert!(!result[3].is_nan());
    }

    #[test]
    fn boxcox_empty() {
        let result = boxcox(&[], 1.0);
        assert!(result.is_empty());
    }

    // ==================== inv_boxcox ====================

    #[test]
    fn inv_boxcox_roundtrip_lambda_1() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let transformed = boxcox(&series, 1.0);
        let recovered = inv_boxcox(&transformed, 1.0);

        for (orig, rec) in series.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    #[test]
    fn inv_boxcox_roundtrip_lambda_0() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let transformed = boxcox(&series, 0.0);
        let recovered = inv_boxcox(&transformed, 0.0);

        for (orig, rec) in series.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    #[test]
    fn inv_boxcox_roundtrip_lambda_05() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let transformed = boxcox(&series, 0.5);
        let recovered = inv_boxcox(&transformed, 0.5);

        for (orig, rec) in series.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    // ==================== boxcox_lambda ====================

    #[test]
    fn boxcox_lambda_finds_reasonable_value() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let lambda = boxcox_lambda(&series);

        // Should find a lambda in reasonable range
        assert!((-2.0..=2.0).contains(&lambda));
    }

    #[test]
    fn boxcox_lambda_exponential_data() {
        // Exponential data should have lambda close to 0 (log transform)
        let series: Vec<f64> = (1..=10).map(|i| (i as f64).exp()).collect();
        let lambda = boxcox_lambda(&series);

        // Lambda should be close to 0 for exponential data
        assert!(
            lambda.abs() < 0.5,
            "Expected lambda near 0 for exponential data, got {}",
            lambda
        );
    }

    // ==================== boxcox_auto ====================

    #[test]
    fn boxcox_auto_works() {
        let series = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // Quadratic
        let result = boxcox_auto(&series);

        assert!(!result.data.is_empty());
        assert!(result.lambda >= -2.0 && result.lambda <= 2.0);
    }

    #[test]
    fn boxcox_auto_inverse() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = boxcox_auto(&series);
        let recovered = result.inverse();

        for (orig, rec) in series.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-6);
        }
    }

    // ==================== is_boxcox_suitable ====================

    #[test]
    fn is_suitable_positive() {
        assert!(is_boxcox_suitable(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn is_suitable_with_zero() {
        assert!(!is_boxcox_suitable(&[0.0, 1.0, 2.0]));
    }

    #[test]
    fn is_suitable_with_negative() {
        assert!(!is_boxcox_suitable(&[-1.0, 1.0, 2.0]));
    }

    #[test]
    fn is_suitable_empty() {
        assert!(!is_boxcox_suitable(&[]));
    }

    // ==================== boxcox_shifted ====================

    #[test]
    fn boxcox_shifted_handles_negatives() {
        let series = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = boxcox_shifted(&series, 1.0);

        // Should not have NaN values after shifting
        assert!(result.data.iter().all(|x| !x.is_nan()));
    }

    #[test]
    fn boxcox_shifted_positive_unchanged() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = boxcox_shifted(&series, 1.0);
        let direct = boxcox(&series, 1.0);

        // For positive data, shifted and direct should be same
        for (s, d) in result.data.iter().zip(direct.iter()) {
            assert_relative_eq!(s, d, epsilon = 1e-10);
        }
    }
}
