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
/// * `normalize` - If true, z-score normalize the series before computing CID-CE (matches tsfresh)
pub fn cid_ce(series: &[f64], normalize: bool) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }

    if normalize {
        // tsfresh: z-score normalize the series first, then compute CID-CE
        let n = series.len() as f64;
        let m: f64 = series.iter().sum::<f64>() / n;
        let std: f64 = (series.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n).sqrt();

        if std < 1e-10 {
            return 0.0; // Constant series
        }

        let normalized: Vec<f64> = series.iter().map(|x| (x - m) / std).collect();
        let sum_sq_diff: f64 = normalized.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        sum_sq_diff.sqrt()
    } else {
        let sum_sq_diff: f64 = series.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        sum_sq_diff.sqrt()
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
/// Uses the LZ76 algorithm matching tsfresh implementation.
/// The series is discretized into bins using np.linspace/searchsorted approach.
///
/// # Arguments
/// * `series` - Input time series
/// * `bins` - Number of bins for discretization
pub fn lempel_ziv_complexity(series: &[f64], bins: usize) -> f64 {
    if series.len() < 2 || bins == 0 {
        return 0.0;
    }

    let n = series.len();

    // Find min/max for binning
    let min_val = series.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        // All same values -> minimal complexity
        return 1.0 / n as f64;
    }

    // Discretize using searchsorted approach (like np.searchsorted on linspace bins)
    // np.linspace creates bins+1 edges, we use bins[1:] for searchsorted
    let bin_width = (max_val - min_val) / bins as f64;
    let sequence: Vec<usize> = series
        .iter()
        .map(|&x| {
            // searchsorted finds insertion point; we clamp to valid range
            let bin = ((x - min_val) / bin_width).floor() as usize;
            bin.min(bins - 1)
        })
        .collect();

    // LZ76 greedy algorithm (matches tsfresh _lempel_ziv_complexity)
    let mut sub_strings: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    let mut ind = 0;
    let mut inc = 1;

    while ind + inc <= n {
        let sub_str: Vec<usize> = sequence[ind..ind + inc].to_vec();
        if sub_strings.contains(&sub_str) {
            inc += 1;
        } else {
            sub_strings.insert(sub_str);
            ind += inc;
            inc = 1;
        }
    }

    sub_strings.len() as f64 / n as f64
}

/// Returns the Lempel-Ziv complexity using median binarization.
///
/// This is an alternative implementation using binary alphabet.
pub fn lempel_ziv_complexity_binary(series: &[f64]) -> f64 {
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
    fn cid_ce_linear_unnormalized() {
        // Linear series: all differences are 1
        let series: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // sqrt(9 * 1^2) = 3
        assert_relative_eq!(cid_ce(&series, false), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn cid_ce_linear_normalized() {
        // Linear series normalized: z-score first, then compute CID-CE
        let series: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // After z-score normalization, differences between consecutive values are equal
        // For n=10, mean=4.5, std=sqrt((0-4.5)^2+...+(9-4.5)^2)/10) = sqrt(8.25) ≈ 2.872
        // Each normalized diff = 1/std ≈ 0.348, sum of 9 squared diffs ≈ 9 * (1/std)^2
        // CID-CE = sqrt(9 / var) = sqrt(9 / 8.25) ≈ 1.044
        let result = cid_ce(&series, true);
        assert!(
            result > 1.0 && result < 1.1,
            "Expected ~1.044, got {}",
            result
        );
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

    // ==================== lempel_ziv_complexity (tsfresh-compatible with bins) ====================

    #[test]
    fn lempel_ziv_constant() {
        // All same values: minimal complexity
        let series = vec![5.0; 20];
        let lz = lempel_ziv_complexity(&series, 10);
        // Constant series has complexity 1/n
        assert_relative_eq!(lz, 1.0 / 20.0, epsilon = 1e-10);
    }

    #[test]
    fn lempel_ziv_alternating() {
        // Simple alternating pattern
        let series: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let lz = lempel_ziv_complexity(&series, 10);
        assert!(!lz.is_nan());
        // Should have relatively low complexity due to repetitive pattern
        assert!(lz > 0.0 && lz < 0.5);
    }

    #[test]
    fn lempel_ziv_random_like() {
        // More random patterns should have higher complexity
        let series: Vec<f64> = (0..50).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let lz = lempel_ziv_complexity(&series, 10);
        assert!(!lz.is_nan());
        assert!(lz > 0.0);
    }

    #[test]
    fn lempel_ziv_short() {
        assert_relative_eq!(lempel_ziv_complexity(&[], 10), 0.0, epsilon = 1e-10);
        assert_relative_eq!(lempel_ziv_complexity(&[1.0], 10), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn lempel_ziv_zero_bins() {
        let series = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(lempel_ziv_complexity(&series, 0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn lempel_ziv_normalized_range() {
        // LZ complexity should be between 0 and 1
        let series: Vec<f64> = (0..100).map(|i| ((i * 7 + 3) % 17) as f64).collect();
        let lz = lempel_ziv_complexity(&series, 10);
        assert!(
            (0.0..=1.0).contains(&lz),
            "LZ should be in [0, 1], got {}",
            lz
        );
    }

    // ==================== lempel_ziv_complexity_binary ====================

    #[test]
    fn lempel_ziv_binary_works() {
        let series: Vec<f64> = (0..50).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let lz = lempel_ziv_complexity_binary(&series);
        assert!(!lz.is_nan());
        assert!(lz > 0.0);
    }
}
