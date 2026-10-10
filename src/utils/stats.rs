//! Statistical utility functions.

/// Quantile function (inverse CDF) for the standard normal distribution.
///
/// Uses Peter Acklam's rational approximation (relative error < 1.15e-9
/// across the full `(0,1)` domain), replacing the previous Abramowitz and
/// Stegun formula 26.2.23 (max absolute error ~4.5e-4 in `p`, which is
/// insufficient for the 1e-6-relative ETS prediction-interval parity this
/// crate now asserts against R `forecast.ets` -- D-08, plan 11-07).
///
/// # Arguments
/// * `p` - Probability value (0.0 to 1.0)
///
/// # Returns
/// The z-score corresponding to the given probability.
///
/// # Example
/// ```
/// use anofox_forecast::utils::quantile_normal;
///
/// // 95% confidence level -> z ≈ 1.96
/// let z = quantile_normal(0.975);
/// assert!((z - 1.96).abs() < 0.01);
/// ```
pub fn quantile_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Peter Acklam's rational approximation coefficients.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Calculate the mean of a slice.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    crate::simd::mean(values)
}

/// Calculate the variance of a slice (sample variance with n-1 denominator).
pub fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    crate::simd::variance_sample(values)
}

/// Calculate the standard deviation of a slice.
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Calculate the median of a slice.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Calculate the mean of finite values in a slice (skipping NaN/Inf).
pub fn nan_mean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Calculate the median of finite values in a slice (skipping NaN/Inf).
pub fn nan_median(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    if n % 2 == 0 {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    } else {
        finite[n / 2]
    }
}

/// Aggregate a slice of values using a named function.
///
/// Supports: "mean", "var", "std", "min", "max", "median".
/// Variance uses the sample (n-1) denominator.
pub fn aggregate(values: &[f64], func: &str) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    match func {
        "mean" => values.iter().sum::<f64>() / values.len() as f64,
        "var" => {
            if values.len() < 2 {
                return f64::NAN;
            }
            let m = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
        }
        "std" => {
            let var = aggregate(values, "var");
            var.sqrt()
        }
        "min" => values.iter().copied().fold(f64::INFINITY, f64::min),
        "max" => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "median" => median(values),
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn quantile_normal_known_values() {
        // Test known quantiles
        assert_relative_eq!(quantile_normal(0.5), 0.0, epsilon = 0.01);
        assert_relative_eq!(quantile_normal(0.975), 1.96, epsilon = 0.01);
        assert_relative_eq!(quantile_normal(0.025), -1.96, epsilon = 0.01);
        assert_relative_eq!(quantile_normal(0.995), 2.576, epsilon = 0.01);
    }

    #[test]
    fn quantile_normal_boundary_values() {
        assert_eq!(quantile_normal(0.0), f64::NEG_INFINITY);
        assert_eq!(quantile_normal(1.0), f64::INFINITY);
    }

    #[test]
    fn mean_calculates_correctly() {
        assert_relative_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0, epsilon = 1e-10);
        assert_relative_eq!(mean(&[10.0]), 10.0, epsilon = 1e-10);
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn variance_calculates_correctly() {
        // Sample variance of [1, 2, 3, 4, 5] = 2.5
        assert_relative_eq!(variance(&[1.0, 2.0, 3.0, 4.0, 5.0]), 2.5, epsilon = 1e-10);
        assert!(variance(&[1.0]).is_nan());
        assert!(variance(&[]).is_nan());
    }

    #[test]
    fn std_dev_calculates_correctly() {
        assert_relative_eq!(
            std_dev(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            2.5_f64.sqrt(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn median_calculates_correctly() {
        // Odd number of elements
        assert_relative_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0, epsilon = 1e-10);
        // Even number of elements
        assert_relative_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5, epsilon = 1e-10);
        // Unsorted input
        assert_relative_eq!(median(&[5.0, 1.0, 3.0, 2.0, 4.0]), 3.0, epsilon = 1e-10);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn nan_mean_skips_nan_and_inf() {
        assert_relative_eq!(
            nan_mean(&[1.0, f64::NAN, 3.0, f64::INFINITY, 5.0]),
            3.0,
            epsilon = 1e-10
        );
        assert!(nan_mean(&[f64::NAN, f64::NAN]).is_nan());
        assert!(nan_mean(&[]).is_nan());
    }

    #[test]
    fn nan_median_skips_nan_and_inf() {
        assert_relative_eq!(
            nan_median(&[1.0, f64::NAN, 3.0, f64::INFINITY, 5.0]),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(nan_median(&[1.0, f64::NAN, 4.0]), 2.5, epsilon = 1e-10);
        assert!(nan_median(&[f64::NAN]).is_nan());
        assert!(nan_median(&[]).is_nan());
    }
}
