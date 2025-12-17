//! Basic statistical features for time series.
//!
//! Provides fundamental statistics like mean, variance, energy, etc.

/// Returns the absolute energy of the time series (sum of squared values).
///
/// # Arguments
/// * `series` - Input time series
///
/// # Returns
/// Sum of x\[i\]^2 for all i
pub fn abs_energy(series: &[f64]) -> f64 {
    crate::simd::sum_of_squares(series)
}

/// Returns the highest absolute value of the time series.
pub fn absolute_maximum(series: &[f64]) -> f64 {
    series
        .iter()
        .map(|x| x.abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Returns the sum of absolute differences between consecutive values.
///
/// sum(|x\[i+1\] - x\[i\]|) for i = 0..n-1
pub fn absolute_sum_of_changes(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }
    series.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

/// Returns the number of elements in the time series.
pub fn length(series: &[f64]) -> f64 {
    series.len() as f64
}

/// Returns the maximum value.
pub fn maximum(series: &[f64]) -> f64 {
    series.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Returns the arithmetic mean.
pub fn mean(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    crate::simd::mean(series)
}

/// Returns the mean of absolute differences between consecutive values.
pub fn mean_abs_change(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return f64::NAN;
    }
    absolute_sum_of_changes(series) / (series.len() - 1) as f64
}

/// Returns the mean of differences between consecutive values.
pub fn mean_change(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return f64::NAN;
    }
    // Simplifies to (last - first) / (n - 1)
    (series[series.len() - 1] - series[0]) / (series.len() - 1) as f64
}

/// Returns the mean of the central approximation of the second derivative.
///
/// (x\[i+2\] - 2*x\[i+1\] + x\[i\]) / 2
pub fn mean_second_derivative_central(series: &[f64]) -> f64 {
    if series.len() < 3 {
        return f64::NAN;
    }
    let sum: f64 = series
        .windows(3)
        .map(|w| (w[2] - 2.0 * w[1] + w[0]) / 2.0)
        .sum();
    sum / (series.len() - 2) as f64
}

/// Returns the mean of the n largest absolute values.
///
/// # Arguments
/// * `series` - Input time series
/// * `n` - Number of largest values to average
pub fn mean_n_absolute_max(series: &[f64], n: usize) -> f64 {
    if series.is_empty() || n == 0 {
        return f64::NAN;
    }
    let mut abs_values: Vec<f64> = series.iter().map(|x| x.abs()).collect();
    abs_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let count = n.min(abs_values.len());
    abs_values[..count].iter().sum::<f64>() / count as f64
}

/// Returns the median value.
pub fn median(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Returns the minimum value.
pub fn minimum(series: &[f64]) -> f64 {
    series.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Returns the root mean square (quadratic mean).
pub fn root_mean_square(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    (abs_energy(series) / series.len() as f64).sqrt()
}

/// Returns the sample standard deviation.
pub fn standard_deviation(series: &[f64]) -> f64 {
    variance(series).sqrt()
}

/// Returns the sum of all values.
pub fn sum_values(series: &[f64]) -> f64 {
    crate::simd::sum(series)
}

/// Returns the population variance (with n denominator).
///
/// Note: This uses the population formula (n denominator) to match tsfresh.
/// For sample variance, use `variance_sample`.
pub fn variance(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    if series.len() == 1 {
        return 0.0;
    }
    crate::simd::variance(series)
}

/// Returns the sample variance (with n-1 denominator).
pub fn variance_sample(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return f64::NAN;
    }
    crate::simd::variance_sample(series)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== abs_energy ====================

    #[test]
    fn abs_energy_known_values() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // 1 + 4 + 9 + 16 + 25 = 55
        assert_relative_eq!(abs_energy(&series), 55.0, epsilon = 1e-10);
    }

    #[test]
    fn abs_energy_empty() {
        let series: Vec<f64> = vec![];
        assert_relative_eq!(abs_energy(&series), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn abs_energy_single() {
        assert_relative_eq!(abs_energy(&[3.0]), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn abs_energy_negative_values() {
        let series = vec![-1.0, -2.0, 3.0];
        // 1 + 4 + 9 = 14
        assert_relative_eq!(abs_energy(&series), 14.0, epsilon = 1e-10);
    }

    // ==================== absolute_maximum ====================

    #[test]
    fn absolute_maximum_positive() {
        let series = vec![1.0, 5.0, 3.0, 2.0];
        assert_relative_eq!(absolute_maximum(&series), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn absolute_maximum_negative_larger() {
        let series = vec![1.0, -10.0, 3.0, 2.0];
        assert_relative_eq!(absolute_maximum(&series), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn absolute_maximum_empty() {
        let series: Vec<f64> = vec![];
        assert_eq!(absolute_maximum(&series), f64::NEG_INFINITY);
    }

    // ==================== absolute_sum_of_changes ====================

    #[test]
    fn absolute_sum_of_changes_known() {
        let series = vec![1.0, 3.0, 7.0, 4.0];
        // |3-1| + |7-3| + |4-7| = 2 + 4 + 3 = 9
        assert_relative_eq!(absolute_sum_of_changes(&series), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn absolute_sum_of_changes_empty() {
        assert_relative_eq!(absolute_sum_of_changes(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn absolute_sum_of_changes_single() {
        assert_relative_eq!(absolute_sum_of_changes(&[5.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn absolute_sum_of_changes_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(absolute_sum_of_changes(&series), 0.0, epsilon = 1e-10);
    }

    // ==================== length ====================

    #[test]
    fn length_various() {
        assert_relative_eq!(length(&[1.0, 2.0, 3.0]), 3.0, epsilon = 1e-10);
        assert_relative_eq!(length(&[]), 0.0, epsilon = 1e-10);
        assert_relative_eq!(length(&[1.0]), 1.0, epsilon = 1e-10);
    }

    // ==================== maximum ====================

    #[test]
    fn maximum_known() {
        let series = vec![1.0, 5.0, 3.0, 2.0];
        assert_relative_eq!(maximum(&series), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn maximum_negative() {
        let series = vec![-5.0, -2.0, -10.0];
        assert_relative_eq!(maximum(&series), -2.0, epsilon = 1e-10);
    }

    #[test]
    fn maximum_empty() {
        assert_eq!(maximum(&[]), f64::NEG_INFINITY);
    }

    // ==================== mean ====================

    #[test]
    fn mean_known() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(mean(&series), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_empty() {
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn mean_single() {
        assert_relative_eq!(mean(&[7.0]), 7.0, epsilon = 1e-10);
    }

    // ==================== mean_abs_change ====================

    #[test]
    fn mean_abs_change_known() {
        let series = vec![1.0, 3.0, 7.0, 4.0];
        // (2 + 4 + 3) / 3 = 3
        assert_relative_eq!(mean_abs_change(&series), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_abs_change_short() {
        assert!(mean_abs_change(&[]).is_nan());
        assert!(mean_abs_change(&[1.0]).is_nan());
    }

    // ==================== mean_change ====================

    #[test]
    fn mean_change_known() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // (5 - 1) / 4 = 1
        assert_relative_eq!(mean_change(&series), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_change_decreasing() {
        let series = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        // (2 - 10) / 4 = -2
        assert_relative_eq!(mean_change(&series), -2.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_change_short() {
        assert!(mean_change(&[]).is_nan());
        assert!(mean_change(&[1.0]).is_nan());
    }

    // ==================== mean_second_derivative_central ====================

    #[test]
    fn mean_second_derivative_central_linear() {
        // Linear function: second derivative is 0
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(
            mean_second_derivative_central(&series),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn mean_second_derivative_central_quadratic() {
        // x^2: [0, 1, 4, 9, 16] -> second derivative = 2
        let series = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        // (4 - 2 + 0)/2 + (9 - 8 + 1)/2 + (16 - 18 + 4)/2 = 1 + 1 + 1 = 3
        // mean = 3 / 3 = 1 (central approx)
        let result = mean_second_derivative_central(&series);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_second_derivative_central_short() {
        assert!(mean_second_derivative_central(&[]).is_nan());
        assert!(mean_second_derivative_central(&[1.0]).is_nan());
        assert!(mean_second_derivative_central(&[1.0, 2.0]).is_nan());
    }

    // ==================== mean_n_absolute_max ====================

    #[test]
    fn mean_n_absolute_max_known() {
        let series = vec![1.0, -5.0, 3.0, -2.0, 4.0];
        // Absolute values: [1, 5, 3, 2, 4]
        // Top 3: [5, 4, 3] -> mean = 4
        assert_relative_eq!(mean_n_absolute_max(&series, 3), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_n_absolute_max_all() {
        let series = vec![1.0, 2.0, 3.0];
        // All 3: mean of [1, 2, 3] = 2
        assert_relative_eq!(mean_n_absolute_max(&series, 10), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_n_absolute_max_empty() {
        assert!(mean_n_absolute_max(&[], 3).is_nan());
    }

    #[test]
    fn mean_n_absolute_max_zero_n() {
        assert!(mean_n_absolute_max(&[1.0, 2.0], 0).is_nan());
    }

    // ==================== median ====================

    #[test]
    fn median_odd() {
        let series = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_relative_eq!(median(&series), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn median_even() {
        let series = vec![1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(median(&series), 2.5, epsilon = 1e-10);
    }

    #[test]
    fn median_empty() {
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn median_single() {
        assert_relative_eq!(median(&[7.0]), 7.0, epsilon = 1e-10);
    }

    // ==================== minimum ====================

    #[test]
    fn minimum_known() {
        let series = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_relative_eq!(minimum(&series), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn minimum_negative() {
        let series = vec![-5.0, -2.0, -10.0];
        assert_relative_eq!(minimum(&series), -10.0, epsilon = 1e-10);
    }

    #[test]
    fn minimum_empty() {
        assert_eq!(minimum(&[]), f64::INFINITY);
    }

    // ==================== root_mean_square ====================

    #[test]
    fn root_mean_square_known() {
        let series = vec![1.0, 2.0, 3.0];
        // sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.16
        let expected = (14.0_f64 / 3.0).sqrt();
        assert_relative_eq!(root_mean_square(&series), expected, epsilon = 1e-10);
    }

    #[test]
    fn root_mean_square_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(root_mean_square(&series), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn root_mean_square_empty() {
        assert!(root_mean_square(&[]).is_nan());
    }

    // ==================== standard_deviation (population) ====================

    #[test]
    fn standard_deviation_known() {
        // For population std dev with known values
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // mean = 3, population variance = 2.0, population std = sqrt(2.0)
        assert_relative_eq!(standard_deviation(&series), 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn standard_deviation_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(standard_deviation(&series), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn standard_deviation_short() {
        assert!(standard_deviation(&[]).is_nan());
    }

    #[test]
    fn standard_deviation_single() {
        // Single value should have 0 std
        assert_relative_eq!(standard_deviation(&[5.0]), 0.0, epsilon = 1e-10);
    }

    // ==================== sum_values ====================

    #[test]
    fn sum_values_known() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(sum_values(&series), 15.0, epsilon = 1e-10);
    }

    #[test]
    fn sum_values_empty() {
        assert_relative_eq!(sum_values(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn sum_values_negative() {
        let series = vec![-1.0, -2.0, 3.0];
        assert_relative_eq!(sum_values(&series), 0.0, epsilon = 1e-10);
    }

    // ==================== variance (population) ====================

    #[test]
    fn variance_known() {
        // Population variance with known values
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // mean = 3, sum_sq = (4+1+0+1+4) = 10, population var = 10/5 = 2.0
        assert_relative_eq!(variance(&series), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn variance_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(variance(&series), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn variance_two_values() {
        let series = vec![0.0, 2.0];
        // mean = 1, sum_sq = 1 + 1 = 2, population var = 2/2 = 1.0
        assert_relative_eq!(variance(&series), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn variance_short() {
        assert!(variance(&[]).is_nan());
    }

    #[test]
    fn variance_single() {
        // Single value should have 0 variance
        assert_relative_eq!(variance(&[5.0]), 0.0, epsilon = 1e-10);
    }

    // ==================== variance_sample ====================

    #[test]
    fn variance_sample_known() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // mean = 3, sum_sq = 10, sample var = 10/4 = 2.5
        assert_relative_eq!(variance_sample(&series), 2.5, epsilon = 1e-10);
    }

    #[test]
    fn variance_sample_short() {
        assert!(variance_sample(&[]).is_nan());
        assert!(variance_sample(&[1.0]).is_nan());
    }
}
