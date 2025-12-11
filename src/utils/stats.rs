//! Statistical utility functions.

/// Approximate quantile function for standard normal distribution.
///
/// Uses the Abramowitz and Stegun approximation (formula 26.2.23).
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

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Abramowitz and Stegun coefficients
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

/// Calculate the mean of a slice.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate the variance of a slice (sample variance with n-1 denominator).
pub fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let m = mean(values);
    let sum_sq: f64 = values.iter().map(|x| (x - m).powi(2)).sum();
    sum_sq / (values.len() - 1) as f64
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
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Calculate the autocorrelation at a given lag.
pub fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    if values.len() <= lag {
        return f64::NAN;
    }
    let m = mean(values);
    let n = values.len();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        denominator += (values[i] - m).powi(2);
        if i >= lag {
            numerator += (values[i] - m) * (values[i - lag] - m);
        }
    }

    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator
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
    fn autocorrelation_lag_0_is_1() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(autocorrelation(&values, 0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn autocorrelation_known_pattern() {
        // For a simple linear trend, lag-1 autocorrelation should be high
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let acf1 = autocorrelation(&values, 1);
        // For a linear trend of length n, ACF(1) ≈ (n-2)/(n+1) ≈ 0.86 for n=20
        assert!(acf1 > 0.8);
    }
}
