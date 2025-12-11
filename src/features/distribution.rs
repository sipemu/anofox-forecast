//! Distribution-related features for time series.
//!
//! Provides features based on the statistical distribution of values.

use super::basic::{mean, standard_deviation, variance};

/// Returns the skewness (third standardized moment).
///
/// Measures the asymmetry of the distribution.
pub fn skewness(series: &[f64]) -> f64 {
    if series.len() < 3 {
        return f64::NAN;
    }
    let n = series.len() as f64;
    let m = mean(series);
    let s = standard_deviation(series);

    if s < 1e-10 {
        return 0.0;
    }

    let sum_cubed: f64 = series.iter().map(|x| ((x - m) / s).powi(3)).sum();

    // Adjusted Fisher-Pearson standardized moment coefficient
    (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
}

/// Returns the kurtosis (fourth standardized moment).
///
/// Measures the "tailedness" of the distribution.
/// Returns excess kurtosis (normal distribution = 0).
pub fn kurtosis(series: &[f64]) -> f64 {
    if series.len() < 4 {
        return f64::NAN;
    }
    let n = series.len() as f64;
    let m = mean(series);
    let s = standard_deviation(series);

    if s < 1e-10 {
        return f64::NAN;
    }

    let sum_fourth: f64 = series.iter().map(|x| ((x - m) / s).powi(4)).sum();

    // Excess kurtosis formula
    let k = (n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum_fourth;
    k - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0))
}

/// Returns the value at the given quantile.
///
/// # Arguments
/// * `series` - Input time series
/// * `q` - Quantile (0.0 to 1.0)
pub fn quantile(series: &[f64], q: f64) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let pos = q * (n - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let frac = pos - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Returns whether the standard deviation is larger than r * (max - min).
///
/// # Arguments
/// * `series` - Input time series
/// * `r` - Ratio threshold (typically 0.25)
pub fn large_standard_deviation(series: &[f64], r: f64) -> bool {
    if series.len() < 2 {
        return false;
    }
    let std = standard_deviation(series);
    let range = series.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - series.iter().copied().fold(f64::INFINITY, f64::min);

    if range < 1e-10 {
        return false;
    }

    std > r * range
}

/// Returns whether the variance is larger than the standard deviation.
///
/// This is true when variance > 1 (since std = sqrt(var)).
pub fn variance_larger_than_standard_deviation(series: &[f64]) -> bool {
    let var = variance(series);
    if var.is_nan() {
        return false;
    }
    var > var.sqrt()
}

/// Returns the coefficient of variation (std / mean).
///
/// Also known as the relative standard deviation.
pub fn variation_coefficient(series: &[f64]) -> f64 {
    let m = mean(series);
    if m.abs() < 1e-10 {
        return f64::NAN;
    }
    standard_deviation(series) / m
}

/// Returns whether the distribution looks symmetric.
///
/// Compares the mean to the median and checks if the difference
/// is small relative to the range.
///
/// # Arguments
/// * `series` - Input time series
/// * `r` - Ratio threshold for symmetry (typically 0.05)
pub fn symmetry_looking(series: &[f64], r: f64) -> bool {
    if series.len() < 2 {
        return true;
    }

    let m = mean(series);
    let med = super::basic::median(series);
    let range = series.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - series.iter().copied().fold(f64::INFINITY, f64::min);

    if range < 1e-10 {
        return true;
    }

    ((m - med) / range).abs() < r
}

/// Returns the ratio of values beyond r standard deviations from the mean.
///
/// # Arguments
/// * `series` - Input time series
/// * `r` - Number of standard deviations (typically 2.0 or 2.5)
pub fn ratio_beyond_r_sigma(series: &[f64], r: f64) -> f64 {
    if series.len() < 2 {
        return f64::NAN;
    }

    let m = mean(series);
    let s = standard_deviation(series);

    if s < 1e-10 {
        return 0.0;
    }

    let threshold = r * s;
    let count = series.iter().filter(|&&x| (x - m).abs() > threshold).count();

    count as f64 / series.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== skewness ====================

    #[test]
    fn skewness_symmetric() {
        // Symmetric distribution should have skewness ≈ 0
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_relative_eq!(skewness(&series), 0.0, epsilon = 0.1);
    }

    #[test]
    fn skewness_right_skewed() {
        // Right-skewed: more mass on the left
        let series = vec![1.0, 1.0, 1.0, 2.0, 2.0, 10.0];
        let sk = skewness(&series);
        assert!(sk > 0.5, "Expected positive skewness, got {}", sk);
    }

    #[test]
    fn skewness_left_skewed() {
        // Left-skewed: more mass on the right
        let series = vec![1.0, 9.0, 9.0, 10.0, 10.0, 10.0];
        let sk = skewness(&series);
        assert!(sk < -0.5, "Expected negative skewness, got {}", sk);
    }

    #[test]
    fn skewness_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(skewness(&series), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn skewness_short() {
        assert!(skewness(&[]).is_nan());
        assert!(skewness(&[1.0]).is_nan());
        assert!(skewness(&[1.0, 2.0]).is_nan());
    }

    // ==================== kurtosis ====================

    #[test]
    fn kurtosis_uniform_like() {
        // Uniform distribution has negative excess kurtosis
        let series: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let k = kurtosis(&series);
        assert!(k < 0.0, "Uniform-like should have negative kurtosis, got {}", k);
    }

    #[test]
    fn kurtosis_heavy_tails() {
        // Heavy tails should have positive excess kurtosis
        let mut series = vec![0.0; 100];
        series[0] = -10.0;
        series[99] = 10.0;
        let k = kurtosis(&series);
        assert!(k > 0.0, "Heavy tails should have positive kurtosis, got {}", k);
    }

    #[test]
    fn kurtosis_constant() {
        let series = vec![5.0; 10];
        assert!(kurtosis(&series).is_nan());
    }

    #[test]
    fn kurtosis_short() {
        assert!(kurtosis(&[]).is_nan());
        assert!(kurtosis(&[1.0]).is_nan());
        assert!(kurtosis(&[1.0, 2.0]).is_nan());
        assert!(kurtosis(&[1.0, 2.0, 3.0]).is_nan());
    }

    // ==================== quantile ====================

    #[test]
    fn quantile_median() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(quantile(&series, 0.5), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn quantile_boundaries() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(quantile(&series, 0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(quantile(&series, 1.0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn quantile_quartiles() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(quantile(&series, 0.25), 2.0, epsilon = 1e-10);
        assert_relative_eq!(quantile(&series, 0.75), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn quantile_unsorted() {
        let series = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_relative_eq!(quantile(&series, 0.5), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn quantile_empty() {
        assert!(quantile(&[], 0.5).is_nan());
    }

    #[test]
    fn quantile_single() {
        assert_relative_eq!(quantile(&[7.0], 0.5), 7.0, epsilon = 1e-10);
    }

    #[test]
    fn quantile_clamped() {
        let series = vec![1.0, 2.0, 3.0];
        // Values outside [0, 1] should be clamped
        assert_relative_eq!(quantile(&series, -0.5), 1.0, epsilon = 1e-10);
        assert_relative_eq!(quantile(&series, 1.5), 3.0, epsilon = 1e-10);
    }

    // ==================== large_standard_deviation ====================

    #[test]
    fn large_standard_deviation_true() {
        // High variance data
        let series = vec![1.0, 10.0, 1.0, 10.0, 1.0, 10.0];
        assert!(large_standard_deviation(&series, 0.25));
    }

    #[test]
    fn large_standard_deviation_false() {
        // Series where most values are clustered but range is large
        // range = 100, most values around 50, so std << 0.25 * 100 = 25
        let series = vec![0.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 100.0];
        // std ≈ 26, which is > 0.25 * 100, so use r = 0.5
        // Actually let's use simpler data
        let series2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // range = 9, std ≈ 3.03, 0.25 * 9 = 2.25, so std > threshold - still true
        // Use r = 0.5: 0.5 * 9 = 4.5 > 3.03, so it should be false
        assert!(!large_standard_deviation(&series2, 0.5));
    }

    #[test]
    fn large_standard_deviation_constant() {
        let series = vec![5.0; 10];
        assert!(!large_standard_deviation(&series, 0.25));
    }

    #[test]
    fn large_standard_deviation_short() {
        assert!(!large_standard_deviation(&[], 0.25));
        assert!(!large_standard_deviation(&[1.0], 0.25));
    }

    // ==================== variance_larger_than_standard_deviation ====================

    #[test]
    fn variance_larger_than_std_true() {
        // Variance > 1 means var > std
        let series = vec![0.0, 5.0, 10.0]; // var ≈ 25
        assert!(variance_larger_than_standard_deviation(&series));
    }

    #[test]
    fn variance_larger_than_std_false() {
        // Variance < 1 means var < std
        let series = vec![0.0, 0.1, 0.2, 0.3]; // var ≈ 0.017
        assert!(!variance_larger_than_standard_deviation(&series));
    }

    #[test]
    fn variance_larger_than_std_short() {
        assert!(!variance_larger_than_standard_deviation(&[]));
        assert!(!variance_larger_than_standard_deviation(&[1.0]));
    }

    // ==================== variation_coefficient ====================

    #[test]
    fn variation_coefficient_known() {
        let series = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0];
        let cv = variation_coefficient(&series);
        // std ≈ 3.16, mean = 11, cv ≈ 0.287
        assert!(cv > 0.0);
        assert!(cv < 1.0);
    }

    #[test]
    fn variation_coefficient_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(variation_coefficient(&series), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn variation_coefficient_zero_mean() {
        let series = vec![-1.0, 0.0, 1.0];
        assert!(variation_coefficient(&series).is_nan());
    }

    // ==================== symmetry_looking ====================

    #[test]
    fn symmetry_looking_symmetric() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(symmetry_looking(&series, 0.05));
    }

    #[test]
    fn symmetry_looking_asymmetric() {
        let series = vec![1.0, 1.0, 1.0, 1.0, 100.0];
        assert!(!symmetry_looking(&series, 0.05));
    }

    #[test]
    fn symmetry_looking_constant() {
        let series = vec![5.0; 10];
        assert!(symmetry_looking(&series, 0.05));
    }

    #[test]
    fn symmetry_looking_short() {
        assert!(symmetry_looking(&[], 0.05));
        assert!(symmetry_looking(&[1.0], 0.05));
    }

    // ==================== ratio_beyond_r_sigma ====================

    #[test]
    fn ratio_beyond_r_sigma_normal_like() {
        // For normal distribution, about 5% should be beyond 2 sigma
        let series: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 20.0).collect();
        let ratio = ratio_beyond_r_sigma(&series, 2.0);
        assert!(ratio < 0.1); // Should be small
    }

    #[test]
    fn ratio_beyond_r_sigma_with_outliers() {
        let mut series = vec![0.0; 100];
        series[0] = 100.0; // Outlier
        series[99] = -100.0; // Outlier
        let ratio = ratio_beyond_r_sigma(&series, 2.0);
        assert!(ratio > 0.0);
    }

    #[test]
    fn ratio_beyond_r_sigma_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(ratio_beyond_r_sigma(&series, 2.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn ratio_beyond_r_sigma_short() {
        assert!(ratio_beyond_r_sigma(&[], 2.0).is_nan());
        assert!(ratio_beyond_r_sigma(&[1.0], 2.0).is_nan());
    }
}
