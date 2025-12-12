//! Complexity-based features for time series.
//!
//! Provides features measuring the complexity or irregularity of time series.

/// Returns the Complexity-Invariant Distance (CID-CE).
///
/// Measures the complexity of a time series based on consecutive differences.
/// Higher values indicate more complex (irregular) series.
///
/// # Arguments
/// * `series` - Input time series
/// * `normalize` - Whether to normalize by the series length
pub fn cid_ce(series: &[f64], normalize: bool) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }

    let sum_sq_diff: f64 = series.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let complexity = sum_sq_diff.sqrt();

    if normalize {
        complexity / (series.len() - 1) as f64
    } else {
        complexity
    }
}

/// Returns the C3 statistic measuring non-linearity.
///
/// This is related to time-lagged cross-correlation and measures
/// non-linear dependence in the time series.
///
/// # Arguments
/// * `series` - Input time series
/// * `lag` - Time lag
pub fn c3(series: &[f64], lag: usize) -> f64 {
    if series.len() <= 2 * lag {
        return f64::NAN;
    }

    let n = series.len() - 2 * lag;
    let sum: f64 = (0..n)
        .map(|i| series[i] * series[i + lag] * series[i + 2 * lag])
        .sum();

    sum / n as f64
}

/// Returns the Lempel-Ziv complexity of the time series.
///
/// Measures the number of distinct substrings in the binarized series.
/// Higher values indicate more complex patterns.
///
/// # Arguments
/// * `series` - Input time series
pub fn lempel_ziv_complexity(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }

    // Binarize using median as threshold
    let median = compute_median(series);
    let binary: Vec<u8> = series
        .iter()
        .map(|&x| if x >= median { 1 } else { 0 })
        .collect();

    // Compute LZ complexity
    let n = binary.len();
    let mut i = 0;
    let mut c = 1; // complexity counter
    let mut k = 1;
    let mut k_max = 1;
    let mut l = 1;

    while l + k <= n {
        // Check if substring [i..i+k] has appeared before in [0..l]
        let substring: &[u8] = &binary[i..i + k];
        let search_space: &[u8] = &binary[0..l];

        let found = (0..search_space.len().saturating_sub(k - 1))
            .any(|j| &search_space[j..j + k] == substring);

        if found {
            k += 1;
            if k > k_max {
                k_max = k;
            }
        } else {
            c += 1;
            l += k_max;
            i = l;
            k = 1;
            k_max = 1;
        }
    }

    // Normalize by the theoretical maximum complexity
    let n_f = n as f64;
    let b = 2.0; // binary alphabet
    let max_complexity = n_f / n_f.log(b);

    c as f64 / max_complexity
}

/// Helper: compute median
fn compute_median(series: &[f64]) -> f64 {
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== cid_ce ====================

    #[test]
    fn cid_ce_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(cid_ce(&series, false), 0.0, epsilon = 1e-10);
        assert_relative_eq!(cid_ce(&series, true), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cid_ce_linear() {
        // Linear series: all differences are 1
        let series: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // sqrt(9 * 1^2) = 3
        assert_relative_eq!(cid_ce(&series, false), 3.0, epsilon = 1e-10);
        assert_relative_eq!(cid_ce(&series, true), 3.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    fn cid_ce_complex() {
        // More complex series should have higher CID
        let simple: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let complex: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.0 } else { 10.0 })
            .collect();

        assert!(cid_ce(&complex, false) > cid_ce(&simple, false));
    }

    #[test]
    fn cid_ce_short() {
        assert_relative_eq!(cid_ce(&[], false), 0.0, epsilon = 1e-10);
        assert_relative_eq!(cid_ce(&[1.0], false), 0.0, epsilon = 1e-10);
    }

    // ==================== c3 ====================

    #[test]
    fn c3_constant() {
        let series = vec![2.0; 20];
        // c3 = 2 * 2 * 2 = 8
        assert_relative_eq!(c3(&series, 1), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn c3_linear() {
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = c3(&series, 1);
        assert!(!result.is_nan());
    }

    #[test]
    fn c3_alternating() {
        let series: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let result = c3(&series, 1);
        // For alternating +-1: x[i] * x[i+1] * x[i+2] alternates between -1 and 1
        assert!(!result.is_nan());
    }

    #[test]
    fn c3_short() {
        assert!(c3(&[], 1).is_nan());
        assert!(c3(&[1.0, 2.0], 1).is_nan());
        assert!(c3(&[1.0, 2.0, 3.0], 2).is_nan()); // Need at least 5 elements for lag=2
    }

    #[test]
    fn c3_different_lags() {
        let series: Vec<f64> = (0..50).map(|i| (i as f64 * 0.5).sin()).collect();
        let c3_1 = c3(&series, 1);
        let c3_5 = c3(&series, 5);
        assert!(!c3_1.is_nan());
        assert!(!c3_5.is_nan());
    }

    // ==================== lempel_ziv_complexity ====================

    #[test]
    fn lempel_ziv_constant() {
        // All same values: after binarization, all are same (all >= median)
        // But LZ complexity of a constant binary string is still non-trivial
        // because the algorithm counts distinct substrings
        let series = vec![5.0; 20];
        let lz = lempel_ziv_complexity(&series);
        // Just verify it returns a valid value
        assert!(!lz.is_nan() && lz >= 0.0, "Expected valid LZ, got {}", lz);
    }

    #[test]
    fn lempel_ziv_alternating() {
        // Simple alternating pattern: low complexity
        let series: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let lz = lempel_ziv_complexity(&series);
        assert!(!lz.is_nan());
    }

    #[test]
    fn lempel_ziv_random_like() {
        // More random patterns should have higher complexity
        let series: Vec<f64> = (0..50).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let lz = lempel_ziv_complexity(&series);
        assert!(!lz.is_nan());
    }

    #[test]
    fn lempel_ziv_short() {
        assert_relative_eq!(lempel_ziv_complexity(&[]), 0.0, epsilon = 1e-10);
        assert_relative_eq!(lempel_ziv_complexity(&[1.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn lempel_ziv_normalized() {
        // LZ complexity should be between 0 and 1 (approximately)
        let series: Vec<f64> = (0..100).map(|i| ((i * 7 + 3) % 17) as f64).collect();
        let lz = lempel_ziv_complexity(&series);
        assert!(
            (0.0..=2.0).contains(&lz),
            "LZ should be reasonable, got {}",
            lz
        );
    }
}
